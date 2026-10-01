// Escape currency dollar signs ($2.50 -> \$2.50) in MDX prose so remark-math does not read them as math.
// Skips frontmatter and JSX {...} expressions (e.g. Quiz props).
// Usage: node scripts/fix-dollars.mjs src/content/lessons/<module>/*.mdx
import { readFileSync, writeFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

const CURRENCY = /(?<![\\$])\$(?=\d)/g;

function fix(text) {
  let head = '';
  let body = text;
  if (text.startsWith('---')) {
    const end = text.indexOf('\n---', 3);
    if (end > 0) { head = text.slice(0, end + 4); body = text.slice(end + 4); }
  }
  let out = '';
  let buf = '';
  let depth = 0;
  let fence = false;
  const lines = body.split('\n');
  for (let li = 0; li < lines.length; li++) {
    const line = lines[li] + (li < lines.length - 1 ? '\n' : '');
    if (/^\s*```/.test(line) && depth === 0) {
      out += buf.replace(CURRENCY, '\\$');
      buf = '';
      fence = !fence;
      out += line;
      continue;
    }
    if (fence) { out += line; continue; }
    for (const ch of line) {
      if (ch === '{') { if (depth === 0) { out += buf.replace(CURRENCY, '\\$'); buf = ''; } depth++; buf += ch; }
      else if (ch === '}' && depth > 0) { depth--; buf += ch; if (depth === 0) { out += buf; buf = ''; } }
      else buf += ch;
    }
  }
  out += depth === 0 ? buf.replace(CURRENCY, '\\$') : buf;
  return head + out;
}

function expand(p) {
  const st = statSync(p);
  if (st.isDirectory()) return readdirSync(p).flatMap((f) => expand(join(p, f)));
  return p.endsWith('.mdx') ? [p] : [];
}

for (const p of process.argv.slice(2).flatMap(expand)) {
  const s = readFileSync(p, 'utf8');
  const t = fix(s);
  if (t !== s) { writeFileSync(p, t); console.log('fixed', p); }
}
