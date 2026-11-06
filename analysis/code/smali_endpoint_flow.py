"""Bounded concrete string flow for custom LLM requests in Smali.

This is an additional recovery path, not a whole-program interpreter. It
follows explicit parameters, per-allocation captured fields and StringBuilder
assembly into URL/OkHttp destinations. Only explicitly scheduled Runnable
objects are executed. Unknown values and control-flow joins lose information.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re

from registry import scan_url_for_business_ai_endpoint, is_custom_service_url, _is_llm_schema
from signal_types import SignalVerification
from urllib.parse import urljoin, urlsplit
from webview_fields import written_registers, without_comments


_CALL = re.compile(r"invoke-([\w/-]+)\s+\{([^}]*)\},\s*(L[^;]+;->\S+)")
_STRING = re.compile(r'const-string(?:/jumbo)?\s+([vp]\d+),\s*"((?:\\.|[^"\\])*)"')
_MOVE = re.compile(r"move(?:-object|-wide)?(?:/from16|/16)?\s+([vp]\d+),\s*([vp]\d+)")
_FIELD = re.compile(r"(iget|iput)(?:-object)?\s+([vp]\d+),\s*([vp]\d+),\s*(L\S+)")


@dataclass(eq=False)
class _Object:
    owner: str
    fields: dict[str, object] = field(default_factory=dict)
    text: str | None = None
    runnable: _Object | None = None
    client_base: str | None = None
    api_owner: str | None = None
    json_fields: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class _ClassValue:
    owner: str


def _registers(spec: str) -> list[str]:
    if ".." in spec:
        first, last = (s.strip() for s in spec.split("..", 1))
        if first[:1] != last[:1] or not first[1:].isdigit() or not last[1:].isdigit():
            return []
        start, end = int(first[1:]), int(last[1:])
        return [first[0] + str(i) for i in range(start, min(end + 1, start + 256))]
    return [s.strip() for s in spec.split(",") if s.strip()]


def find_endpoint_flows(methods, tags, max_depth: int = 5, protocols=None, static_fields=None):
    """Return endpoint observations with source and concrete request provenance.

    Each source gets at most 512 method visits and 50,000 instructions; depth
    includes wrapper calls and callback execution. f48 still counts sources,
    while multiple proven endpoint values can enter f16/f17.
    """
    relations = []
    any_limited = False
    related_api = (set(protocols.api_methods) | {key for key, api in protocols.all_api_methods.items()
                                               if api.body_type == 'Lorg/json/JSONObject;'}) if protocols else set()
    for source in sorted(methods):
        tag = tags[source]
        if not methods[source].app_owned or not (
            tag.custom_route or tag.custom_source or tag.schema_source
            or any(call[3] in related_api for call in _CALL.finditer(methods[source].body))
        ):
            continue
        budget = [512, 50000]
        found: dict[str, tuple[str, ...]] = {}
        bound_fields: dict[str, tuple[str, ...]] = {}
        source_context = methods[source].body

        def observe(value, path):
            if not isinstance(value, str):
                return
            if any(s.verification != SignalVerification.CANDIDATE for s in scan_url_for_business_ai_endpoint(value, source, "smali", source_context)):
                found.setdefault(value, path)

        def run(key, args, depth, path, is_root=False):
            nonlocal any_limited
            method = methods.get(key)
            if method is None or not method.app_owned or key in path:
                return None
            if depth > max_depth or budget[0] <= 0:
                any_limited = True
                return None
            budget[0] -= 1
            path = (*path, key)
            regs = {f"p{i}": value for i, value in enumerate(args)}
            # Parameter aliases in .locals output (J/D consume two slots in
            # invoke lists, so the slot-wise mapping also handles wide args).
            local_count = re.search(r"(?m)^\s*\.locals\s+(\d+)", method.body)
            offset = int(local_count[1]) if local_count else None
            if offset is None:
                total = re.search(r"(?m)^\s*\.registers\s+(\d+)", method.body)
                if total and args:
                    offset = int(total[1]) - len(args)

            def reg(name):
                if offset is not None and name.startswith("v") and int(name[1:]) >= offset:
                    return "p" + str(int(name[1:]) - offset)
                return name

            def get(name):
                return regs.get(reg(name))

            def put(name, value):
                regs[reg(name)] = value

            def invalidate():
                # Captures are shared with callers. Kill them as well when a
                # branch makes a field assignment ambiguous.
                objects = [v for v in regs.values() if isinstance(v, _Object)]
                seen = set()
                while objects:
                    obj = objects.pop()
                    if id(obj) in seen:
                        continue
                    seen.add(id(obj))
                    objects.extend(v for v in obj.fields.values() if isinstance(v, _Object))
                    if obj.runnable:
                        objects.append(obj.runnable)
                    obj.fields.clear()
                    obj.text = None
                    obj.runnable = None
                    obj.client_base = None
                    obj.api_owner = None
                    obj.json_fields.clear()
                regs.clear()

            lines = without_comments(method.body).splitlines()
            # Keep only immutable String registers established before the first
            # forward branch and never written again. Objects and branch-local
            # values retain the existing conservative invalidation behavior.
            labels = {line.strip(): i for i, line in enumerate(lines) if line.strip().startswith(':')}
            safe_forward = not re.search(r'(?m)^\s*(?:\.catch|packed-switch|sparse-switch|throw|goto)', method.body)
            last_write = {}
            for i, raw in enumerate(lines):
                line = raw.strip()
                for dest in written_registers(line):
                    last_write[reg(dest)] = i
                if line.startswith(('if-', 'goto')):
                    target = line.split()[-1]
                    if labels.get(target, -1) <= i:
                        safe_forward = False
            stable_strings = None

            def invalidate_at_join(index):
                nonlocal stable_strings
                if stable_strings is None:
                    stable_strings = ({key: value for key, value in regs.items()
                                       if isinstance(value, str) and last_write.get(key, -1) < index}
                                      if safe_forward else {})
                invalidate()
                regs.update(stable_strings)

            pending = None
            for line_index, raw in enumerate(lines):
                budget[1] -= 1
                if budget[1] < 0:
                    any_limited = True
                    return None
                line = raw.strip()
                if not line or line.startswith((".", "#")):
                    continue
                # Exception-range labels aren't joins. No guessing values
                # across actual branches, switch labels or catch handlers.
                if line.startswith(":"):
                    if not line.startswith((":try_start", ":try_end")):
                        invalidate_at_join(line_index)
                        pending = None
                    continue
                if line.startswith(("if-", "goto", "packed-switch", "sparse-switch")):
                    invalidate_at_join(line_index)
                    pending = None
                    continue
                if m := _STRING.fullmatch(line):
                    put(m[1], m[2].replace(r"\/", "/"))
                elif m := re.fullmatch(r'sget-object\s+([vp]\d+),\s*(L\S+)', line):
                    put(m[1], (static_fields or {}).get(m[2]))
                elif m := re.fullmatch(r"const-class\s+([vp]\d+),\s*(L[^;]+;)", line):
                    put(m[1], _ClassValue(m[2]))
                elif m := _MOVE.fullmatch(line):
                    put(m[1], get(m[2]))
                elif line.startswith("move-result"):
                    put(line.split()[-1], pending)
                    pending = None
                elif m := re.fullmatch(r"new-instance\s+([vp]\d+),\s*(L[^;]+;)", line):
                    put(m[1], _Object(m[2]))
                elif m := _FIELD.fullmatch(line):
                    obj = get(m[3])
                    if m[1] == "iget":
                        put(m[2], obj.fields.get(m[4]) if isinstance(obj, _Object) else None)
                    elif isinstance(obj, _Object):
                        obj.fields[m[4]] = get(m[2])
                elif m := _CALL.fullmatch(line):
                    target = m[3]
                    values = [get(r) for r in _registers(m[2])]
                    receiver = values[0] if values else None
                    owner, signature = target.split("->", 1)
                    pending = None
                    if (protocols and target in protocols.all_api_methods and isinstance(receiver, _Object)
                            and receiver.api_owner == owner and receiver.client_base):
                        api = protocols.all_api_methods[target]
                        payload = values[api.body_slot] if 0 <= api.body_slot < len(values) else None
                        body_bound = (api.request_kind == 'body' and isinstance(payload, _Object)
                                      and payload.owner == api.body_type
                                      and (target in protocols.api_methods or _is_llm_schema(payload.json_fields)))
                        form_bound = (api.request_kind == 'form' and target in protocols.api_methods
                                      and all(slot < len(values) and (name not in {'model', 'messages', 'prompt'} or values[slot] is not None)
                                              for slot, name in api.parameter_fields))
                        if body_bound or form_bound:
                            base = receiver.client_base
                            route = api.route
                            if is_custom_service_url(base) and not route.startswith(("http:", "https:", "//")):
                                endpoint = urljoin(base, route)
                                if urlsplit(endpoint).netloc == urlsplit(base).netloc:
                                    endpoint = endpoint.split("?", 1)[0].split("#", 1)[0]
                                    found.setdefault(endpoint, (*path, target))
                                    bound_fields.setdefault(endpoint, tuple(sorted(set(api.fields) | (payload.json_fields if body_bound else set()))))
                    elif (owner == "Lretrofit2/Retrofit$Builder;" and signature.startswith("baseUrl(Ljava/lang/String;)")
                          and isinstance(receiver, _Object) and receiver.owner == owner):
                        base = values[1] if len(values) == 2 else None
                        receiver.client_base = base if isinstance(base, str) and base.startswith(("https://", "http://")) else None
                        pending = receiver
                    elif (owner == "Lretrofit2/Retrofit$Builder;" and signature.startswith("build()")
                          and isinstance(receiver, _Object) and receiver.owner == owner):
                        pending = _Object("Lretrofit2/Retrofit;", client_base=receiver.client_base)
                    elif (owner == "Lretrofit2/Retrofit$Builder;" and isinstance(receiver, _Object) and receiver.owner == owner
                          and signature.endswith(")Lretrofit2/Retrofit$Builder;")
                          and signature.split("(", 1)[0] in {"client", "addConverterFactory", "addCallAdapterFactory", "callbackExecutor", "validateEagerly"}):
                        pending = receiver
                    elif (protocols and owner in protocols.proxy_factories and signature.endswith("(Ljava/lang/Class;)Ljava/lang/Object;")
                          and isinstance(receiver, _Object) and len(values) == 2 and isinstance(values[1], _ClassValue)):
                        if receiver.client_base:
                            pending = _Object(values[1].owner, client_base=receiver.client_base, api_owner=values[1].owner)
                    elif (owner == "Lorg/json/JSONObject;" and signature.startswith("put(Ljava/lang/String;")
                          and isinstance(receiver, _Object) and len(values) >= 3 and isinstance(values[1], str)):
                        receiver.json_fields.add(values[1].lower())
                        pending = receiver
                    elif owner in {"Ljava/lang/StringBuilder;", "Ljava/lang/StringBuffer;"} and isinstance(receiver, _Object):
                        if signature.startswith("<init>("):
                            receiver.text = "" if len(values) == 1 else (
                                values[1] if isinstance(values[1], str) else None)
                        elif signature.startswith("append("):
                            fragment = values[1] if len(values) == 2 else None
                            receiver.text = receiver.text + fragment if (
                                isinstance(receiver.text, str) and isinstance(fragment, str)
                                and len(receiver.text) + len(fragment) <= 4096) else None
                            pending = receiver
                        elif signature == "toString()Ljava/lang/String;":
                            pending = receiver.text
                    elif target == "Ljava/lang/String;->concat(Ljava/lang/String;)Ljava/lang/String;":
                        if len(values) == 2 and all(isinstance(v, str) for v in values):
                            pending = "".join(values) if sum(map(len, values)) <= 4096 else None
                    elif target == "Ljava/net/URL;-><init>(Ljava/lang/String;)V" and isinstance(receiver, _Object):
                        receiver.text = values[1] if len(values) == 2 and isinstance(values[1], str) else None
                    elif owner == "Ljava/net/URL;" and signature.startswith("openConnection("):
                        if isinstance(receiver, _Object):
                            observe(receiver.text, path)
                    elif target in {
                        "Lokhttp3/Request$Builder;->url(Ljava/lang/String;)Lokhttp3/Request$Builder;",
                        "Lcom/squareup/okhttp/Request$Builder;->url(Ljava/lang/String;)Lcom/squareup/okhttp/Request$Builder;",
                    }:
                        # Request URL configuration, not proof of execution.
                        if isinstance(receiver, _Object):
                            receiver.text = values[1] if len(values) == 2 and isinstance(values[1], str) else None
                            pending = receiver
                    elif owner in {"Lokhttp3/Request$Builder;", "Lcom/squareup/okhttp/Request$Builder;"} and signature.startswith("build()"):
                        pending = receiver
                    elif owner in {"Lokhttp3/OkHttpClient;", "Lcom/squareup/okhttp/OkHttpClient;"} and signature.startswith("newCall("):
                        if len(values) == 2 and isinstance(values[1], _Object):
                            observe(values[1].text, path)
                    elif owner == "Ljava/lang/Thread;" and signature == "<init>(Ljava/lang/Runnable;)V" and isinstance(receiver, _Object):
                        receiver.runnable = values[1] if len(values) == 2 and isinstance(values[1], _Object) else None
                    elif target == "Ljava/lang/Thread;->start()V" and isinstance(receiver, _Object):
                        if receiver.runnable:
                            task = receiver.runnable
                            run(task.owner + "->run()V", [task], depth + 1, path)
                    elif target in {
                        "Ljava/util/concurrent/Executor;->execute(Ljava/lang/Runnable;)V",
                        "Ljava/util/concurrent/ExecutorService;->execute(Ljava/lang/Runnable;)V",
                        "Landroid/os/Handler;->post(Ljava/lang/Runnable;)Z",
                    }:
                        if len(values) == 2 and isinstance(values[1], _Object):
                            task = values[1]
                            run(task.owner + "->run()V", [task], depth + 1, path)
                    elif target in methods and (any(v is not None for v in values) or not values):
                        pending = run(target, values, depth + 1, path)
                elif line.startswith("return"):
                    if not is_root:
                        return get(line.split()[-1]) if line.startswith("return-object") else None
                    regs.clear()
                    stable_strings = {}
                    pending = None
                elif line.startswith("check-cast "):
                    pass
                else:
                    # Unknown writes kill previous constants, including sget,
                    # numeric const, array reads and arithmetic. Field writes
                    # not modeled here do not manufacture string values.
                    dest = re.match(r"[\w/-]+\s+([vp]\d+)(?:,|$)", line)
                    if dest and not line.startswith(("iput", "sput", "aput", "monitor-")):
                        put(dest[1], None)
            return None

        run(source, [], 0, (), is_root=True)
        for value, path in sorted(found.items()):
            api = protocols.all_api_methods.get(path[-1]) if protocols and path else None
            relations.append({
                "value": value,
                "source_method": source,
                "components": [f"request_url={value}", f"binding=client_interface_{api.request_kind}" if api else "binding=concrete_string_arguments",
                               *(f"flow_method={key}" for key in path),
                               *((f"serialized_field={name}" for name in bound_fields.get(value, api.fields)) if api else ()),
                               "http_sink=api_proxy_invocation" if api else "http_sink=url_request_construction"],
            })
    return relations, any_limited
