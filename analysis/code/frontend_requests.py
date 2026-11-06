"""Bound request facts from static JS and explicitly bound local web entries.

No business-domain lists, label inputs, online page fetches or APK execution.
Unknown remote pages are navigation facts only, not inferred LLM functions.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit
import json
import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile

from bounded_literals import javascript_payload
from business_facts import scan_llm_literal
from app_scope import normalize_component_name
from smali_literals import decode_smali_string
from webview_fields import without_comments, register_normalizer, written_registers
from registry import (normalize_llm_model_id, extract_llm_model_ids, _is_llm_schema,
                      _GENERATION_PARAMETER_KEYS, is_custom_service_url, endpoint_noise_reason,
                      classify_url_signal, match_web_service)
from signal_types import RawSignal, SignalType, SignalRole, SignalScope, SignalRelation, SignalVerification


@dataclass(frozen=True)
class Script:
    module_id: str
    byte_offset: int
    text: str
    asset_root: str = ''


def response_protocol_paths(fact):
    """Distinctive response access, independent of the request's body/URL."""
    body = fact.get('body')
    if isinstance(body, dict) and (re.search(
            r'^(?:flux(?:\.|-|$)|stable[-_/ ]diffusion|sdxl|dall-e|imagen)',
            str(body.get('model') or '').lower()) or body.get('responseModalities') == ['IMAGE']):
        return []
    paths = [[p for p in items if not str(p).isdigit() and p not in {'json'}]
             for items in fact.get('response_paths', [])]
    return sorted({'.'.join(p[-3:]) for p in paths if p[-3:] in (
        ['choices', 'delta', 'content'], ['choices', 'message', 'content'])}
        | {'.'.join(p[-4:]) for p in paths if p[-4:] == ['candidates', 'content', 'parts', 'text']})


def request_evidence(fact):
    body = fact.get('body')
    if not isinstance(body, dict):
        return ()
    model = str(body.get('model') or '').lower()
    # Pure image-generation profiles cannot borrow an adjacent chat model.
    if re.search(r'^(?:flux(?:\.|-|$)|stable[-_/ ]diffusion|sdxl|dall-e|imagen)', model):
        return ()
    if body.get('responseModalities') == ['IMAGE']:
        return ()
    messages = body.get('messages')
    message_shape = (isinstance(messages, list) and bool(messages) and
                     all(isinstance(m, dict) and m.get('role') in {'system', 'user', 'assistant'}
                         and 'content' in m for m in messages))
    fields = set(body)
    if message_shape:
        fields |= {'role', 'content'}
    explicit_model = model in extract_llm_model_ids(model) and normalize_llm_model_id(model) == model
    if message_shape and explicit_model:
        return ('typed_messages_with_model_identity',)
    if message_shape and _is_llm_schema(fields):
        return ('bound_generation_request_schema',)
    if message_shape and response_protocol_paths(fact):
        return ('bound_message_request_text_response',)
    controls = set(body) & _GENERATION_PARAMETER_KEYS
    config = body.get('generationConfig') or body.get('generation_config')
    if isinstance(config, dict):
        controls |= {k for k in config if k.lower().replace('_', '') in
                     {'temperature', 'topp', 'topk', 'maxoutputtokens', 'maxtokens'}}
    prompt = body.get('prompt')
    paths = [[p for p in items if not str(p).isdigit() and p not in {'json'}]
             for items in fact.get('response_paths', [])]
    text_response = any(p[-3:] == ['choices', 'delta', 'content'] or
                        p[-3:] == ['choices', 'message', 'content'] or
                        p[-4:] == ['candidates', 'content', 'parts', 'text'] for p in paths)
    if isinstance(prompt, str) and controls and text_response and re.search(
        r'^(?:please\s+)?(?:write|generate|compose|draft|rewrite|summari[sz]e|translate)\b', prompt, re.I
    ) and re.search(r'\b(?:poem|story|text|email|paragraph|reply|response|summary|article|letter)s?\b', prompt, re.I):
        return ('bound_generation_instruction_response',)
    return ()


def request_signals(fact, source=None, *, layer='zip'):
    evidence = request_evidence(fact)
    if not evidence:
        # An unresolved body must not erase independently observed response
        # structure. This is a reference, never an endpoint or HTTP binding.
        return [RawSignal(SignalType.LLM_JSON_SCHEMA, path, layer, source or fact['file'],
                          role=SignalRole.REFERENCE, scope=SignalScope.METHOD,
                          source_kind='js_response_structure',
                          components=(f'response_path={path}', f"request_id={fact['id']}",
                                      'binding=response_access_only'))
                for path in response_protocol_paths(fact)]
    source = source or fact['file']
    components = (f"request_id={fact['id']}", 'binding=request_object',
                  f"http_invocation={fact['invocation']}", 'operation=text_generation',
                  'stage=static_invocation', f"source_line={fact['line']}",
                  f"char_offset={fact['start']}", 'offset_unit=utf16_code_units',
                  *(f'protocol_evidence={item}' for item in evidence))
    signals = [RawSignal(SignalType.LLM_HTTP_API_USAGE, fact['id'], layer, source,
                         role=SignalRole.RELATION, scope=SignalScope.METHOD,
                         relation=SignalRelation.HTTP_API_LLM_CONTEXT, components=components,
                         verification=SignalVerification.VERIFIED, source_kind='js_bound_request')]
    url = fact.get('url')
    if not url:
        return signals
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password:
            return signals
        endpoint = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, '', ''))
    except ValueError:
        return signals
    identities = list(classify_url_signal(source, endpoint, layer))
    signals.extend(s for s in identities if s.signal_type == SignalType.KNOWN_LLM_API_DOMAIN)
    if is_custom_service_url(endpoint) and not endpoint_noise_reason(endpoint):
        signals.append(RawSignal(SignalType.BUSINESS_AI_ENDPOINT, endpoint, layer, source,
                                 role=SignalRole.RELATION, scope=SignalScope.METHOD,
                                 relation=SignalRelation.CUSTOM_DOMAIN_LLM_PATH,
                                 components=(*components, f'custom_domain={parsed.hostname}', f'llm_path={parsed.path}'),
                                 verification=SignalVerification.VERIFIED, source_kind='js_bound_request'))
    return signals


def _asset_path(base, value):
    if not isinstance(value, str) or any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in value):
        return None
    try:
        parts = urlsplit(value)
    except ValueError:
        return None
    if parts.scheme or parts.netloc:
        return None
    value = unquote(parts.path)
    target = posixpath.normpath(posixpath.join(base, value))
    return target if target.startswith('assets/') and not target.startswith('assets/../') else None


def _material_issue(diagnostics, reason, source, *, line=None, exception=None):
    """Record locations and error types, never source literals or exception text."""
    diagnostics['status'] = 'partial'
    diagnostics.setdefault('reason', reason)
    diagnostics['error_count'] = diagnostics.get('error_count', 0) + 1
    errors = diagnostics.setdefault('errors', [])
    exception_type = type(exception).__name__ if exception is not None else None
    for item in errors:
        if (item['reason'], item['source'], item['exception_type']) == (reason, str(source), exception_type):
            item['count'] += 1
            if line is not None:
                item['last_line'] = line
            return
    errors.append({'reason': reason, 'source': str(source), 'exception_type': exception_type,
                   'count': 1, 'first_line': line, 'last_line': line})


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts, self.inline, self.frames, self.buffer = [], [], [], None
        self.base = None
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'base' and self.base is None:
            self.base = attrs.get('href')
        if tag == 'script':
            if attrs.get('src'):
                self.scripts.append(attrs['src'])
            elif attrs.get('type', '').lower() in {'', 'module', 'text/javascript', 'application/javascript'}:
                self.buffer = []
        if tag == 'iframe' and attrs.get('src'):
            self.frames.append(attrs['src'])
    def handle_data(self, data):
        if self.buffer is not None:
            self.buffer.append(data)
    def handle_endtag(self, tag):
        if tag == 'script' and self.buffer is not None:
            self.inline.append(''.join(self.buffer)); self.buffer = None


def local_entries(root, scope, metadata, diagnostics=None):
    """Read actual launcher framework configuration and typed local loadUrl."""
    entries = []
    info = diagnostics if diagnostics is not None else {}
    if not root or not root.exists():
        return entries
    launchers = metadata.get('manifest_launcher_components') or []
    package = metadata.get('manifest_package') or ''
    launcher_owners = {'L'+normalize_component_name(n, package).replace('.', '/')+';'
                       for n in launchers if isinstance(n, str) and n}
    # Resolve superclass declarations, including a custom intermediate activity.
    def class_text(owner):
        relative = owner[1:-1]+'.smali'
        for dex in sorted(root.glob('smali*')):
            file = dex/relative
            try:
                if file.is_file():
                    return file.read_text(encoding='utf-8')
            except (OSError, UnicodeError) as exc:
                _material_issue(info, 'unreadable_smali', file.relative_to(root), exception=exc)
        return ''
    for owner in sorted(launcher_owners):
        visited = set()
        current = owner
        while current and current not in visited:
            visited.add(current)
            if current == 'Lorg/apache/cordova/CordovaActivity;':
                config = root/'res/xml/config.xml'
                try:
                    if config.is_file():
                        xml = ET.fromstring(config.read_text(encoding='utf-8'))
                        for element in xml.iter():
                            if element.tag.rsplit('}',1)[-1] == 'content':
                                target = _asset_path('assets/www', element.get('src', ''))
                                if target:
                                    entries.append({'asset': target, 'binding': 'framework_entry_configuration', 'owner': owner})
                except (OSError, UnicodeError, ValueError, ET.ParseError) as exc:
                    _material_issue(info, 'invalid_framework_config', config.relative_to(root), exception=exc)
                break
            if current == 'Lcom/getcapacitor/BridgeActivity;':
                config = root/'assets/capacitor.config.json'
                try:
                    if config.is_file():
                        data = json.loads(config.read_text(encoding='utf-8-sig'))
                        if not isinstance(data, dict) or not isinstance(data.get('server', {}), dict):
                            raise ValueError('invalid_capacitor_config_shape')
                        if data.get('server', {}).get('url'):
                            break  # Remote configuration must not bind unused local assets.
                except (OSError, UnicodeError, ValueError) as exc:
                    _material_issue(info, 'invalid_framework_config', config.relative_to(root), exception=exc)
                    break
                entries.append({'asset':'assets/public/index.html','binding':'framework_entry_configuration','owner':owner})
                break
            parent = re.search(r'(?m)^\s*\.super\s+(L[^;\s]+;)', class_text(current))
            current = parent[1] if parent else ''
    files = (file for dex in sorted(root.glob('smali*')) if dex.is_dir() for file in sorted(dex.rglob('*.smali'))
             if scope.matches_smali_path(file, root))
    for file in files:
        try:
            text = without_comments(file.read_text(encoding='utf-8'))
        except (OSError, UnicodeError) as exc:
            _material_issue(info, 'unreadable_smali', file.relative_to(root), exception=exc)
            continue
        line_cursor, line_number = 0, 1
        for method in re.finditer(r'(?ms)^\s*\.method\s+([^\n]+)\n(.*?)^\s*\.end method', text):
            values = {}
            normalize = register_normalizer(method[1], method[2])
            line_number += text.count('\n', line_cursor, method.start(2))
            line_cursor = method.start(2)
            for line_no, line in enumerate(method[2].splitlines(), line_number):
                line = line.strip()
                if line.startswith((':','if-','goto','return','throw','packed-switch','sparse-switch','.catch')):
                    values.clear()
                literal = re.fullmatch(r'const-string(?:/jumbo)?\s+([vp]\d+),\s*("(?:\\.|[^"\\])*")', line)
                if literal:
                    target = normalize(literal[1])
                    values.pop(target, None)
                    try:
                        values[target] = decode_smali_string(literal[2])
                    except ValueError as exc:
                        _material_issue(info, 'invalid_smali_string', file.relative_to(root),
                                        line=line_no, exception=exc)
                    continue
                if re.match(r'const-string(?:/jumbo)?\b', line):
                    _material_issue(info, 'invalid_smali_string', file.relative_to(root), line=line_no)
                call = re.fullmatch(r'invoke-virtual\s+\{([vp]\d+),\s*([vp]\d+)\},\s*Landroid/webkit/WebView;->loadUrl\(Ljava/lang/String;\)V', line)
                if call:
                    url = values.get(normalize(call[2]), '')
                    if url.startswith('file:///android_asset/'):
                        target = _asset_path('assets', url[len('file:///android_asset/'):])
                        if target:
                            entries.append({'asset':target,'binding':'local_asset_argument','owner':str(file)})
                else:
                    for dest in written_registers(line):
                        values.pop(normalize(dest), None)
    return list({(e['asset'],e['binding'],e['owner']):e for e in entries}.values())


def extract_frontend_requests(apk, root, scope, metadata, diagnostics):
    diagnostics.update(status='not_needed', entries=[], scripts=0, requests=0,
                       coverage='static_const_object_requests_v3', unqualified_requests=0, remote_navigation=[],
                       error_count=0, errors=[])
    signals = []
    try:
        entries = local_entries(root, scope, metadata, diagnostics)
        diagnostics['entries'] = entries
        if not entries:
            return signals
        if diagnostics['status'] != 'partial':
            diagnostics['status'] = 'complete'
        with zipfile.ZipFile(apk) as archive:
            names = set(archive.namelist())
            for entry in entries:
                if entry['asset'] not in names:
                    _material_issue(diagnostics, 'missing_entry_asset', entry['asset']); continue
                asset_root = posixpath.dirname(entry['asset'])
                pages = [entry['asset']]; visited_pages=set(); scripts=[]; pending=[]
                def navigation(url, page_source, kind):
                    try:
                        parsed=urlsplit(url)
                    except ValueError as exc:
                        _material_issue(diagnostics, 'invalid_navigation_url', page_source, exception=exc)
                        return
                    if parsed.scheme not in {'http','https'} or not parsed.hostname or parsed.username or parsed.password:
                        return
                    clean=urlunsplit((parsed.scheme,parsed.netloc,parsed.path,'',''))
                    diagnostics['remote_navigation'].append({'source':page_source,'target':clean,'kind':kind})
                    if not match_web_service(clean):
                        return
                    signals.append(RawSignal(SignalType.WEBVIEW_LLM_TARGET,clean,'zip',page_source,
                        role=SignalRole.RELATION,scope=SignalScope.FILE,
                        relation=SignalRelation.WEBVIEW_KNOWN_LLM_TARGET,
                        components=('binding=frontend_entry_navigation','relevance_hint=known_llm_website',
                            f"entry_asset={entry['asset']}",f"entry_binding={entry['binding']}",f'navigation_kind={kind}'),
                        verification=SignalVerification.OBSERVED,source_kind='frontend_navigation'))
                while pages:
                    page_source=pages.pop()
                    if page_source in visited_pages:
                        continue
                    visited_pages.add(page_source)
                    try:
                        page=Page();page.feed(archive.read(page_source).decode('utf-8-sig'))
                    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
                        _material_issue(diagnostics, 'unreadable_html_asset', page_source, exception=exc)
                        continue
                    if page.base:
                        _material_issue(diagnostics, 'unsupported_html_base', page_source)
                        continue
                    parent=posixpath.dirname(page_source)
                    scripts.extend(Script(page_source+f'#inline-{i}',0,text,asset_root) for i,text in enumerate(page.inline))
                    for src in page.scripts:
                        try:
                            parsed = urlsplit(src)
                        except ValueError as exc:
                            _material_issue(diagnostics, 'invalid_script_url', page_source, exception=exc)
                            continue
                        if parsed.scheme or parsed.netloc:
                            _material_issue(diagnostics, 'remote_script_unavailable', page_source)
                        else:
                            pending.append(_asset_path(asset_root if src.startswith('/') else parent,src.lstrip('/')))
                    for src in page.frames:
                        try:
                            parsed = urlsplit(src)
                        except ValueError as exc:
                            _material_issue(diagnostics, 'invalid_frame_url', page_source, exception=exc)
                            continue
                        if parsed.scheme or parsed.netloc:
                            navigation(src,page_source,'iframe_src')
                        else:
                            target=_asset_path(asset_root if src.startswith('/') else parent,src.lstrip('/'))
                            if target in names:
                                pages.append(target)
                seen = set()
                while pending:
                    source = pending.pop()
                    if not source or source in seen:
                        continue
                    seen.add(source)
                    if source not in names:
                        _material_issue(diagnostics, 'missing_script_asset', source); continue
                    try:
                        text = archive.read(source).decode('utf-8-sig')
                    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
                        _material_issue(diagnostics, 'unreadable_script_asset', source, exception=exc)
                        continue
                    scripts.append(Script(source,0,text,asset_root))
                    # Discovery only. The AST resolver below validates actual
                    # imports/exports; regex matches never establish relations.
                    for match in re.finditer(r'''\b(?:from\s*|import\s*\(?\s*)["']([^"']+)["']''', text):
                        spec = match[1]
                        if spec.startswith('.'):
                            target = _asset_path(posixpath.dirname(source), spec)
                        elif spec.startswith('/'):
                            target = _asset_path(asset_root, spec.lstrip('/'))
                        else:
                            continue
                        candidates = [target, (target or '')+'.js', (target or '')+'.mjs']
                        pending.append(next((p for p in candidates if p in names), target))
                diagnostics['scripts'] += len(scripts)
                info = {}
                payload = javascript_payload('', info, modules=scripts, facts=True) if scripts else None
                diagnostics.setdefault('parsers', []).append(info)
                if scripts and info.get('status') != 'complete':
                    _material_issue(diagnostics, 'script_parser_partial', entry['asset'])
                # Parsed facts survive unsupported request shapes or incomplete
                # call/response binding. Only independently LLM-related values
                # are retained; unrelated requests never supply that relevance.
                for literal in (payload or {}).get('candidates', []):
                    module = literal['module_id']
                    source = module.split('#inline-', 1)[0]
                    for signal in scan_llm_literal(literal['value'], source, 'zip'):
                        signals.append(replace(signal, source_kind='source_string_literal',
                            components=(*signal.components, 'coverage=atomic_literal',
                                        f'script_asset={module}', f"source_line={literal['line']}",
                                        f"char_offset={literal['start']}", 'offset_unit=utf16_code_units')))
                for nav in (payload or {}).get('facts', {}).get('navigations', []):
                    navigation(nav['url'],nav['file'],'location_assignment')
                for fact in (payload or {}).get('facts', {}).get('requests', []):
                    found = request_signals(fact)
                    signals.extend(found)
                    if not request_evidence(fact):
                        diagnostics['unqualified_requests'] += 1
                        reason = ('request_body_unresolved' if not isinstance(fact.get('body'), dict)
                                  else 'no_supported_llm_request_profile')
                        reasons = diagnostics.setdefault('unqualified_request_reasons', {})
                        reasons[reason] = reasons.get(reason, 0) + 1
                        continue
                    diagnostics['requests'] += 1
                    if not (fact.get('ui_bound') and fact.get('input_bound') and fact.get('render_bound')):
                        continue
                    components = ('binding=frontend_entry_request_response', 'relevance_hint=bound_text_generation',
                                  f"entry_asset={entry['asset']}", f"entry_binding={entry['binding']}",
                                  f"request_id={fact['id']}", f"script_asset={fact['file']}",
                                  f"source_line={fact['line']}", 'input_binding=dom_event', 'response_binding=dom_text')
                    signals.append(RawSignal(SignalType.WEBVIEW_CHAT_PAGE, entry['asset'], 'zip', fact['file'],
                                             role=SignalRole.RELATION, scope=SignalScope.FILE,
                                             relation=SignalRelation.WEBVIEW_CUSTOM_CHAT_TARGET, components=components,
                                             verification=SignalVerification.VERIFIED, source_kind='frontend_entry_request'))
    except (OSError, ValueError, UnicodeError, zipfile.BadZipFile, ET.ParseError) as exc:
        _material_issue(diagnostics, 'frontend_material_error', 'frontend_scan', exception=exc)
    return signals
