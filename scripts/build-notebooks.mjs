// Converts notebooks-src/*.py (percent format: "# %%" code cells, "# %% [markdown]" text cells)
// into Jupyter notebooks at public/notebooks/*.ipynb, ready for "Open in Colab".
import { readdirSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join, basename } from 'node:path';

const SRC = 'notebooks-src';
const OUT = join('public', 'notebooks');
mkdirSync(OUT, { recursive: true });

function toCells(text) {
  const lines = text.replace(/\r\n/g, '\n').split('\n');
  const cells = [];
  let cur = null;
  const push = () => {
    if (!cur) return;
    while (cur.lines.length && cur.lines[cur.lines.length - 1].trim() === '') cur.lines.pop();
    while (cur.lines.length && cur.lines[0].trim() === '') cur.lines.shift();
    if (cur.lines.length) cells.push(cur);
  };
  for (const line of lines) {
    const m = line.match(/^# %%\s*(\[markdown\])?/);
    if (m) {
      push();
      cur = { type: m[1] ? 'markdown' : 'code', lines: [] };
      continue;
    }
    if (!cur) cur = { type: 'code', lines: [] };
    if (cur.type === 'markdown') cur.lines.push(line.replace(/^# ?/, ''));
    else cur.lines.push(line);
  }
  push();
  return cells.map((c) => {
    const source = c.lines.map((l, i) => (i < c.lines.length - 1 ? l + '\n' : l));
    return c.type === 'markdown'
      ? { cell_type: 'markdown', metadata: {}, source }
      : { cell_type: 'code', metadata: {}, execution_count: null, outputs: [], source };
  });
}

let n = 0;
for (const f of readdirSync(SRC).filter((f) => f.endsWith('.py'))) {
  const nb = {
    nbformat: 4,
    nbformat_minor: 5,
    metadata: {
      accelerator: 'GPU',
      colab: { provenance: [], gpuType: 'T4' },
      kernelspec: { name: 'python3', display_name: 'Python 3' },
      language_info: { name: 'python' },
    },
    cells: toCells(readFileSync(join(SRC, f), 'utf8')),
  };
  writeFileSync(join(OUT, basename(f, '.py') + '.ipynb'), JSON.stringify(nb, null, 1));
  n++;
}
console.log(`built ${n} notebooks into ${OUT}`);
