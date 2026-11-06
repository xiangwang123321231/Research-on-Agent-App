// Parse untrusted source as data. Never evaluate, import or execute APK code.
'use strict';
const fs = require('fs');
const parser = require('./vendor/babel_parser/index.cjs');
const MAX_LITERAL_CHARS = 16384;
// Python derives this transport prefilter from the shared evidence registry.
// A standalone invocation without a filter returns all bounded literals.
const hintIndex = process.argv.indexOf('--literal-hints');
const hint = hintIndex < 0 ? null : new RegExp(process.argv[hintIndex + 1], 'i');
const skippedKeys = new Set(['loc', 'extra', 'comments', 'tokens', 'errors']);
const output = {status: 'complete', parser: 'babel-7.26.7', syntax_status: 'pending',
  nodes: 0, literals: 0, candidates: [], facts: [], parsed_modules: 0, scanned_chars: 0, offset_unit: 'utf16_code_units'};
function partial(reason) { output.status = 'partial'; output.reason = reason; }
try {
  const bytes = fs.readFileSync(0);
  const modules = process.argv.includes('--modules') ? JSON.parse(bytes.toString('utf8')).modules
    : [{id: '', byte_offset: 0, source: bytes.toString('utf8')}];
  output.module_count = modules.length;
  const parsedForFacts = [];
  for (const module of modules) {
   try {
    const source = module.source;
    const ast = parser.parse(source, {sourceType: 'unambiguous', attachComment: false,
      errorRecovery: false});
    output.syntax_status = 'complete';
    const stack = [ast];
    const seen = new Set();
    while (stack.length) {
      const node = stack.pop();
      if (!node || typeof node !== 'object' || typeof node.type !== 'string') continue;
      ++output.nodes;
      let value;
      if (node.type === 'StringLiteral') value = node.value;
      else if (node.type === 'TemplateLiteral' && node.expressions.length === 0)
        value = node.quasis[0].value.cooked;
      if (typeof value === 'string') {
        ++output.literals;
        if (value.length <= MAX_LITERAL_CHARS && !seen.has(value) &&
            (!hint || hint.test(value) || /^https?:\/\//i.test(value))) {
          seen.add(value);
          output.candidates.push({value, module_id: module.id, module_byte_offset: module.byte_offset, start: node.start, end: node.end,
            line: node.loc.start.line, column: node.loc.start.column});
        }
      }
      // Reverse the push order to walk source-order arrays without recursive
      // traversal or a spread call that overflows on very large arrays.
      const keys = Object.keys(node);
      for (let k = keys.length - 1; k >= 0; --k) {
        if (skippedKeys.has(keys[k])) continue;
        const child = node[keys[k]];
        if (Array.isArray(child)) {
          for (let i = child.length - 1; i >= 0; --i) if (child[i] && child[i].type) stack.push(child[i]);
        } else if (child && typeof child === 'object' && child.type) stack.push(child);
      }
    }
    output.scanned_chars += source.length;
    ++output.parsed_modules;
    if (process.argv.includes('--facts')) {
      parsedForFacts.push({...module, ast});
    }
   } catch (error) {
     partial(error instanceof SyntaxError ? 'unsupported_or_invalid_syntax' : 'parser_failure');
     output.syntax_status = 'partial';
     (output.module_errors ||= []).push({module_id: module.id, error_offset: error.pos || 0});
   }
  }
  if (output.module_errors) output.syntax_status = 'partial';
  if (process.argv.includes('--facts')) {
    try {
      output.facts = require('./js_request_facts.cjs')(parsedForFacts);
      output.fact_status = 'complete';
    } catch (_) {
      // Unsupported relation analysis must not erase parsed atomic literals.
      partial('request_fact_analysis_failed');
      output.fact_status = 'failed';
      output.facts = {requests: [], imports: [], navigations: []};
    }
  }
} catch (error) {
  partial(error instanceof SyntaxError ? 'unsupported_or_invalid_syntax' : 'parser_failure');
  output.syntax_status = 'failed';
  if (Number.isInteger(error.pos)) output.error_offset = error.pos;
  // Error messages may contain source text; never expose them.
  output.candidates = [];
}
process.stdout.write(JSON.stringify(output));
