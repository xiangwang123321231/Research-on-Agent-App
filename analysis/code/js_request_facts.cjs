// Static AST summaries only. No source evaluation and no application identities.
'use strict';
const path = require('path').posix;
const UNKNOWN = null;
function children(node) {
  const out = [];
  for (const key of Object.keys(node)) {
    if (['loc','extra','comments','tokens','errors'].includes(key)) continue;
    const v = node[key];
    if (Array.isArray(v)) { for (const c of v) if (c && c.type) out.push(c); }
    else if (v && typeof v === 'object' && v.type) out.push(v);
  }
  return out;
}
const property = n => n && (n.type === 'Identifier' ? n.name : n.type === 'StringLiteral' ? n.value : null);
const memberName = n => !n.computed ? property(n.property) : n.property.type === 'StringLiteral' || n.property.type === 'NumericLiteral' ? String(n.property.value) : null;
function collectModules(modules) {
  const environments = new WeakMap(), functionScopes = new WeakMap();
  const moduleMap = new Map(), all = [], writes = [], calls = [], handlers = new Set();
  let serial = 0;
  function scope(parent, module) { return {parent, module, bindings: new Map(), id: ++serial}; }
  function define(env, name, value) {
    if (!name) return;
    const list = env.bindings.get(name) || [];
    list.push(value); env.bindings.set(name, list);
  }
  function lookup(env, name) {
    for (let e = env; e; e = e.parent) if (e.bindings.has(name)) return e.bindings.get(name);
    return null;
  }
  function walk(root, environment) {
    const pending = [[root, environment]];
    while (pending.length) {
      const [node, env] = pending.pop();
      environments.set(node, env); all.push(node);
      if (node.type === 'BlockStatement') {
        const block=scope(env,env.module);
        for (const child of children(node).reverse()) pending.push([child,block]);
        continue;
      }
      if (/^(FunctionDeclaration|FunctionExpression|ArrowFunctionExpression|ObjectMethod)$/.test(node.type)) {
        if (node.type === 'FunctionDeclaration') define(env, node.id && node.id.name, {node, function: true});
        const inner = scope(env, env.module); functionScopes.set(node, inner);
        for (const param of node.params) if (param.type === 'Identifier') define(inner, param.name, {param, fn: node});
        for (const child of children(node)) pending.push([child, inner]);
        continue;
      }
      if (node.type === 'VariableDeclarator' && node.id.type === 'Identifier') define(env, node.id.name, {node: node.init});
      if (node.type === 'ImportDeclaration') {
        for (const spec of node.specifiers) define(env, spec.local.name,
          {import: node.source.value, imported: spec.type === 'ImportDefaultSpecifier' ? 'default' : property(spec.imported)});
      }
      if (node.type === 'AssignmentExpression' || node.type === 'UpdateExpression') writes.push([node, env]);
      if (node.type === 'CallExpression' || node.type === 'OptionalCallExpression') calls.push(node);
      for (const child of children(node).reverse()) pending.push([child, env]);
    }
  }
  for (const m of modules) {
    const env = scope(null, m), exports = new Map();
    moduleMap.set(m.id, {m, env, exports}); walk(m.ast.program, env);
    for (const n of m.ast.program.body) {
      if (n.type === 'ExportDefaultDeclaration') exports.set('default', n.declaration);
      if (n.type !== 'ExportNamedDeclaration') continue;
      if (n.declaration) {
        if (n.declaration.id) exports.set(n.declaration.id.name, n.declaration);
        for (const d of n.declaration.declarations || []) if (d.id.type === 'Identifier') exports.set(d.id.name, d.init);
      }
      for (const spec of n.specifiers || []) exports.set(property(spec.exported), {local: property(spec.local)});
    }
  }
  // Reassignments invalidate a binding throughout its lexical scope. Keeping
  // an old value across an unsupported branch would invent a request target.
  const propertyWrites = [];
  for (const [node, env] of writes) {
    const left = node.left || node.argument;
    if (left.type === 'Identifier') {
      const binding = lookup(env, left.name); if (binding) binding.push({invalid: true});
    } else if (left.type === 'MemberExpression') propertyWrites.push([left, env]);
  }
  const invalidProperties = new WeakMap();
  function objectDeclaration(env, name, seen = new Set()) {
    const defs=lookup(env,name);
    if (!defs || defs.length !== 1 || seen.has(defs)) return null;
    seen.add(defs);
    const n=defs[0].node;
    if (!n) return null;
    if (n.type === 'Identifier') return objectDeclaration(environments.get(n) || env,n.name,seen);
    return n;
  }
  for (const [left,env] of propertyWrites) {
    let base=left.object, first=memberName(left);
    while(base.type === 'MemberExpression') { first=memberName(base); base=base.object; }
    if (base.type !== 'Identifier') continue;
    const declaration=objectDeclaration(env,base.name);
    if (!declaration) continue;
    const names=invalidProperties.get(declaration) || new Set();
    names.add(first); invalidProperties.set(declaration,names);
  }
  const readonlyGlobals = new Set(['fetch']);
  const safeFunctionCache = new WeakMap();
  function declaration(env, name, seen = new Set()) {
    const defs=lookup(env,name);
    if (!defs || defs.length !== 1 || seen.has(defs)) return null;
    seen.add(defs);
    const d=defs[0];
    if (d.import) {
      if (!d.import.startsWith('.') && !d.import.startsWith('/')) return {library:d.import};
      const target=path.normalize(d.import.startsWith('/') ? (env.module.root || path.dirname(env.module.id))+d.import
        : path.join(path.dirname(env.module.id),d.import));
      const found=[target,target+'.js',target+'.mjs',target+'/index.js'].map(p=>moduleMap.get(p)).find(Boolean);
      if (!found) return null;
      const value=found.exports.get(d.imported);
      if (value && value.local) return declaration(found.env,value.local,seen);
      return value ? {node:value,env:environments.get(value) || found.env} : null;
    }
    if (d.node && d.node.type === 'Identifier') return declaration(environments.get(d.node) || env,d.node.name,seen);
    return {node:d.node,env:environments.get(d.node) || env};
  }
  function safeFunction(fn, seen = new Set()) {
    if (!fn || seen.has(fn)) return false;
    if (safeFunctionCache.has(fn)) return safeFunctionCache.get(fn);
    seen.add(fn);
    const stack=[fn.body];
    while(stack.length) {
      const n=stack.pop();
      if (n.type === 'AssignmentExpression' || n.type === 'UpdateExpression'
          || n.type === 'CallExpression' && !safeCall(n, seen)) {safeFunctionCache.set(fn,false);return false;}
      for (const child of children(n)) stack.push(child);
    }
    safeFunctionCache.set(fn,true);
    return true;
  }
  function safeCall(call, seen = new Set()) {
    const c=call.callee, env=environments.get(call);
    if (c.type === 'Identifier') {
      const defs=lookup(env,c.name);
      if (!defs) return readonlyGlobals.has(c.name);
      if (defs.length !== 1) return false;
      if (defs[0].import && ['cross-fetch','node-fetch'].includes(defs[0].import)) return true;
      const def=declaration(env,c.name),fn=def && def.node;
      return fn && /Function/.test(fn.type) && safeFunction(fn,seen);
    }
    if (c.type !== 'MemberExpression' || c.object.type !== 'Identifier') return false;
    const name=c.object.name, method=memberName(c), defs=lookup(env,name),decl=declaration(env,name);
    let lib=!defs ? name : decl && decl.library;
    if (decl && decl.node && decl.node.type === 'CallExpression' && decl.node.callee.type === 'MemberExpression'
        && memberName(decl.node.callee) === 'create' && safeCall(decl.node)) lib='axios';
    return lib === 'JSON' && method === 'stringify'
      || lib === 'console' && ['log','warn','error'].includes(method)
      || lib === 'axios' && ['post','put','patch','create'].includes(method)
      || ['jquery','$','jQuery'].includes(lib) && method === 'ajax';
  }
  for (const call of calls) {
    if (safeCall(call)) continue;
    const env=environments.get(call);
    for (const arg of call.arguments) {
      const node=arg.type === 'Identifier' ? objectDeclaration(env,arg.name) : arg.type === 'ObjectExpression' ? arg : null;
      if (node && node.type === 'ObjectExpression') invalidProperties.set(node,new Set([null]));
    }
  }
  const active = new Set();
  function imported(env, spec, name, args) {
    if (['axios','jquery','cross-fetch','node-fetch'].includes(spec)) return {kind: 'library', name: spec, global: false};
    if (!spec.startsWith('.') && !spec.startsWith('/')) return UNKNOWN;
    const target = path.normalize(spec.startsWith('/') ? (env.module.root || path.dirname(env.module.id)) + spec : path.join(path.dirname(env.module.id), spec));
    const found = [target, target+'.js', target+'.mjs', target+'/index.js'].map(p => moduleMap.get(p)).find(Boolean);
    if (!found) return UNKNOWN;
    const value = found.exports.get(name);
    if (!value) return UNKNOWN;
    if (value.local) return identifier(found.env, value.local, args);
    return evaluate(value, found.env, args);
  }
  function identifier(env, name, args) {
    const defs = lookup(env, name);
    if (!defs) {
      if (['fetch','axios','$','jQuery','document','JSON'].includes(name)) return {kind:'library', name};
      return UNKNOWN;
    }
    if (defs.length !== 1 || defs[0].invalid) return UNKNOWN;
    const d = defs[0];
    if (d.param) return args.has(d.param) ? args.get(d.param) : handlers.has(d.fn) ? {kind:'event'} : {kind:'parameter'};
    if (d.import) return imported(env, d.import, d.imported, args);
    return evaluate(d.node, environments.get(d.node) || env, args);
  }
  function fnNode(node, env, args) { const v = evaluate(node, env, args); return v && v.kind === 'function' ? v.node : null; }
  const requestMap = new Map(), uses = [], uiCalls = new Set();
  function responseUse(value, node, kind) {
    if (value && value.kind === 'response') uses.push({request: value.id, path: value.path, kind,
      file: environments.get(node).module.id, start: node.start});
  }
  function request(node, env, url, options, args, client) {
    const opts = options && options.kind === 'object' ? options.props : {};
    let body = opts.body || opts.data || UNKNOWN;
    if (body && body.kind === 'json') body = body.value;
    const id = env.module.id+':'+node.start;
    if (typeof url === 'string' && client && typeof client.base === 'string' && !/^https?:\/\//i.test(url)) {
      try { url = new URL(url, client.base).href; } catch (_) { url = UNKNOWN; }
    }
    const fact = {id, file: env.module.id, start: node.start, end: node.end,
      line: node.loc.start.line, column: node.loc.start.column, url: typeof url === 'string' ? url : null,
      invocation: client ? client.name : 'fetch', method: typeof opts.method === 'string' ? opts.method : 'POST',
      body: shape(body), input_bound: containsInput(body), ui_bound: uiCalls.has(node), response_paths: []};
    // A parameterized helper can be visited with different arguments. Never
    // combine those contexts into a single stronger request proof.
    const context = JSON.stringify([fact.url, fact.body, fact.input_bound]);
    const key = id+':'+context;
    requestMap.set(key, fact);
    return {kind:'response', id: key, path: []};
  }
  function shape(value, seen = new Set()) {
    if (value == null) return null;
    if (typeof value !== 'object') return value;
    if (seen.has(value)) return null;
    seen.add(value);
    if (value.kind === 'object') {
      const result = {};
      for (const [key, item] of Object.entries(value.props)) {
        if (/key|token|secret|authorization|password/i.test(key)) {
          // Generation control field names remain useful, their values do not.
          result[key] = null;
        } else result[key] = shape(item, new Set(seen));
      }
      return result;
    }
    if (value.kind === 'array') return value.items.map(v => shape(v, new Set(seen)));
    return null;
  }
  function containsInput(value, seen = new Set()) {
    if (!value || typeof value !== 'object' || seen.has(value)) return false;
    if (value.kind === 'input') return true;
    seen.add(value);
    return value.kind === 'object' ? Object.values(value.props).some(v => containsInput(v, seen))
      : value.kind === 'array' ? value.items.some(v => containsInput(v, seen)) : false;
  }
  function evaluate(node, env, args = new Map()) {
    if (!node || active.has(node)) return UNKNOWN;
    active.add(node);
    try {
      const ev = child => evaluate(child, environments.get(child) || env, args);
      if (['StringLiteral','NumericLiteral','BooleanLiteral'].includes(node.type)) return node.value;
      if (node.type === 'Identifier') return identifier(env, node.name, args);
      if (node.type === 'AwaitExpression') return ev(node.argument);
      if (node.type === 'TemplateLiteral') {
        let s = node.quasis[0].value.cooked;
        for (let i=0;i<node.expressions.length;i++) { const v = ev(node.expressions[i]); if (typeof v !== 'string') return UNKNOWN; s += v+node.quasis[i+1].value.cooked; }
        return s;
      }
      if (node.type === 'BinaryExpression' && node.operator === '+') {
        const l=ev(node.left), r=ev(node.right); return typeof l === 'string' && typeof r === 'string' ? l+r : UNKNOWN;
      }
      if (/^(FunctionDeclaration|FunctionExpression|ArrowFunctionExpression|ObjectMethod)$/.test(node.type)) return {kind:'function', node};
      if (node.type === 'ArrayExpression') return {kind:'array', items: node.elements.map(ev)};
      if (node.type === 'ObjectExpression') {
        // Context-dependent objects cannot share a cached parameter value.
        const obj = {kind:'object', id: env.module.id+':'+node.start, props: {}, node};
        for (const prop of node.properties) {
          if (prop.type !== 'ObjectProperty' || prop.computed) return UNKNOWN;
          const name = property(prop.key); if (name == null || Object.hasOwn(obj.props, name)) return UNKNOWN;
          obj.props[name] = ev(prop.value);
        }
        for (const name of invalidProperties.get(node) || []) {
          if (name == null) return UNKNOWN;
          obj.props[name] = UNKNOWN;
        }
        return obj;
      }
      if (node.type === 'MemberExpression' || node.type === 'OptionalMemberExpression') {
        const obj=ev(node.object), name=memberName(node);
        if (!obj || name == null) return UNKNOWN;
        if (obj.kind === 'object') return obj.props[name] ?? UNKNOWN;
        if (obj.kind === 'array') return obj.items[Number(name)] ?? UNKNOWN;
        if (obj.kind === 'dom' && name === 'value') return {kind:'input'};
        if (obj.kind === 'event' && name === 'target') return {kind:'event_target'};
        if (obj.kind === 'event_target' && name === 'value') return {kind:'input'};
        if (obj.kind === 'response') return {...obj, path: [...obj.path,name]};
        return UNKNOWN;
      }
      if (node.type !== 'CallExpression' && node.type !== 'OptionalCallExpression') return UNKNOWN;
      const callee = node.callee, values = node.arguments.map(ev);
      const receiver = callee.type === 'MemberExpression' ? ev(callee.object) : null;
      const name = callee.type === 'MemberExpression' ? memberName(callee) : null;
      const fn = ev(callee);
      if (fn && fn.kind === 'library' && ['fetch','cross-fetch','node-fetch'].includes(fn.name)) return request(node,env,values[0],values[1],args);
      if (receiver && receiver.kind === 'library') {
        if (receiver.name === 'JSON' && name === 'stringify') return {kind:'json', value: values[0]};
        if (receiver.name === 'document' && ['getElementById','querySelector'].includes(name)) return {kind:'dom'};
        if (receiver.name === 'axios' && name === 'create') {
          if (invalidProperties.has(node)) return UNKNOWN;
          const props = values[0] && values[0].props;
          return props ? {kind:'client', name:'axios', base:props.baseURL} : UNKNOWN;
        }
        if (['$','jQuery','jquery'].includes(receiver.name) && name === 'ajax') {
          const opts = values[0];
          return request(node,env,opts && opts.props && opts.props.url,opts,args,{name:'jquery'});
        }
      }
      if (receiver && (receiver.kind === 'client' || receiver.kind === 'library' && receiver.name === 'axios')) {
        if (['post','put','patch'].includes(name)) return request(node,env,values[0],{kind:'object',props:{method:name.toUpperCase(),data:values[1]}},args,receiver);
      }
      if (receiver && receiver.kind === 'response') {
        if (name === 'then') {
          const callback = values[0];
          if (callback && callback.kind === 'function') return callFunction(callback.node,[receiver],args);
        }
        if (['json','text','pipeThrough','getReader','read'].includes(name)) return {...receiver,path:[...receiver.path,name]};
      }
      if (fn && fn.kind === 'function') return callFunction(fn.node,values,args);
      return UNKNOWN;
    } finally { active.delete(node); }
  }
  function callFunction(fn, values, args) {
    const inner=functionScopes.get(fn), bound=new Map(args);
    fn.params.forEach((p,i) => { if (p.type === 'Identifier') bound.set(p,values[i] ?? UNKNOWN); });
    if (fn.body.type !== 'BlockStatement') return evaluate(fn.body,inner,bound);
    // Only a single straight-line return is summarized; branches and unknown
    // writes do not silently become one of several possible return values.
    const body=fn.body.body;
    if (body.some(n => /^(If|For|While|DoWhile|Switch|Try|Throw)/.test(n.type))) return UNKNOWN;
    const returns=body.filter(n => n.type === 'ReturnStatement');
    return returns.length === 1 ? evaluate(returns[0].argument,environments.get(returns[0].argument) || inner,bound) : UNKNOWN;
  }
  // Mark concrete DOM event callbacks first, then analyze request arguments.
  for (const call of calls) {
    const env=environments.get(call), callee=call.callee;
    if (callee.type !== 'MemberExpression' || memberName(callee) !== 'addEventListener') continue;
    const element=evaluate(callee.object,env), event=evaluate(call.arguments[0],env);
    const fn=fnNode(call.arguments[1],env,new Map());
    if (element && element.kind === 'dom' && ['click','submit','input','change'].includes(event) && fn) handlers.add(fn);
  }
  const handlerScopes = new Set([...handlers].map(fn => functionScopes.get(fn)));
  for (const call of calls) {
    const env=environments.get(call);
    let e=env;
    while(e) { if (handlerScopes.has(e)) {uiCalls.add(call);break;} e=e.parent; }
    const callee=call.callee;
    const method=callee.type === 'MemberExpression' ? memberName(callee) : null;
    const defs=callee.type === 'Identifier' ? lookup(env,callee.name) : null;
    const publicImport=defs && defs.length === 1 && ['cross-fetch','node-fetch'].includes(defs[0].import);
    if (method && ['post','put','patch','ajax','then'].includes(method)
        || callee.type === 'Identifier' && (callee.name === 'fetch' || publicImport)) evaluate(call,env);
  }
  for (const node of all) {
    if (node.type !== 'AssignmentExpression' || node.left.type !== 'MemberExpression') continue;
    if (!['textContent','innerText','innerHTML'].includes(memberName(node.left))) continue;
    const env=environments.get(node), element=evaluate(node.left.object,env);
    if (element && element.kind === 'dom') responseUse(evaluate(node.right,env),node,'dom_text');
  }
  const requests=[];
  for (const [key,fact] of requestMap) {
    const matches=uses.filter(u => u.request === key);
    fact.response_paths=matches.map(u => u.path); fact.render_bound=matches.length > 0;
    requests.push(fact);
  }
  const navigations=[];
  for (const node of all) {
    if (node.type !== 'AssignmentExpression' || node.operator !== '=' || node.left.type !== 'MemberExpression') continue;
    const left=node.left,env=environments.get(node);
    const direct=memberName(left) === 'href' && left.object.type === 'Identifier' && left.object.name === 'location' && !lookup(env,'location');
    const windowLocation=memberName(left) === 'href' && left.object.type === 'MemberExpression'
      && memberName(left.object) === 'location' && left.object.object.type === 'Identifier'
      && left.object.object.name === 'window' && !lookup(env,'window');
    if (!direct && !windowLocation) continue;
    const url=evaluate(node.right,env);
    if (typeof url === 'string') navigations.push({file:env.module.id,url,start:node.start,line:node.loc.start.line});
  }
  const imports=[];
  for (const m of modules) for (const n of m.ast.program.body) if (n.type === 'ImportDeclaration') imports.push({file:m.id,source:n.source.value});
  return {requests,imports,navigations,coverage:'static_const_object_requests_v1'};
}
module.exports = collectModules;
