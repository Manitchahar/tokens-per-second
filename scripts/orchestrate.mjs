// Parallel course builder. Hands each unfinished module to an external coding agent
// (opencode or Command Code, both on the Space Bunny Alpha model), each in its own copy of the project.
// A module is merged into the main site only after it passes verification in a clean staging copy
// (main site + that module's files only, then a full `npm run build`).
//
// Usage: node scripts/orchestrate.mjs [--concurrency 6] [--only slug1,slug2] [--attempts 4]
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync, readdirSync, rmSync, statSync, appendFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

const MAIN = resolve('.');
const WORK = resolve(MAIN, '..', 'tps-workers');
const LOGS = join(MAIN, 'logs', 'orchestrator');
mkdirSync(WORK, { recursive: true });
mkdirSync(LOGS, { recursive: true });

const arg = (name, def) => { const i = process.argv.indexOf(`--${name}`); return i > 0 ? process.argv[i + 1] : def; };
const CONCURRENCY = Number(arg('concurrency', 6));
const ATTEMPTS = Number(arg('attempts', 4));
const ONLY = arg('only', '')?.split(',').filter(Boolean) ?? [];
const AGENT_TIMEOUT_MIN = Number(arg('timeout', 180));

// Command templates are re-read before every agent run from logs/orchestrator/tools.json (if present),
// so models can be switched (e.g. after a rate limit) without restarting the orchestrator.
// {ROOT} and {PROMPT} are substituted. Set "disabled": true on a tool to skip it.
const DEFAULT_TOOLS = {
  opencode: { cmd: 'opencode run --auto --dir "{ROOT}" -m opencode/space-bunny-free "{PROMPT}"' },
  cmdc: { cmd: 'cmdc -p "{PROMPT}" -m stealth/space-bunny-alpha --yolo --trust --skip-onboarding --max-turns 600' },
};
function toolConfig() {
  const f = join(LOGS, 'tools.json');
  if (!existsSync(f)) writeFileSync(f, JSON.stringify(DEFAULT_TOOLS, null, 2));
  try { return { ...DEFAULT_TOOLS, ...JSON.parse(readFileSync(f, 'utf8')) }; } catch { return DEFAULT_TOOLS; }
}
const TOOLS = new Proxy({}, {
  get: (_, name) => (prompt, root) => toolConfig()[name].cmd.replaceAll('{ROOT}', root).replaceAll('{PROMPT}', prompt),
});
/** First enabled tool in the preferred order, rotating by attempt. */
function pickTool(order, attempt) {
  const cfg = toolConfig();
  const enabled = order.filter((t) => cfg[t] && !cfg[t].disabled);
  const list = enabled.length ? enabled : order;
  return list[(attempt - 1) % list.length];
}

// ───────────── syllabus
function readModules() {
  const src = readFileSync(join(MAIN, 'src/data/course.ts'), 'utf8');
  const mods = [];
  const re = /num: '(\d+)', slug: '([^']+)', title: '([^']+)'[\s\S]*?tagline: '((?:[^'\\]|\\.)*)',\s*lessons: \[([\s\S]*?)\n    \],/g;
  let m;
  while ((m = re.exec(src))) {
    const lessons = [...m[5].matchAll(/(L|LAB|PROJECT)\('([^']+)', '((?:[^'\\]|\\.)*)'/g)].map((x) => ({
      kind: x[1] === 'L' ? 'lesson' : x[1] === 'LAB' ? 'lab' : 'project', slug: x[2], title: x[3].replace(/\\u2019/g, '’'),
    }));
    mods.push({ num: m[1], slug: m[2], title: m[3], tagline: m[4], lessons });
  }
  return mods;
}

const log = (msg) => { const line = `[${new Date().toISOString().slice(11, 19)}] ${msg}`; console.log(line); appendFileSync(join(LOGS, 'orchestrator.log'), line + '\n'); };
const status = {};
const saveStatus = () => writeFileSync(join(LOGS, 'status.json'), JSON.stringify(status, null, 2));

// ───────────── filesystem helpers
function robocopy(from, to, extra = []) {
  const r = spawnSync('robocopy', [from, to, ...extra, '/NFL', '/NDL', '/NJH', '/NJS', '/NP', '/R:2', '/W:1'], { encoding: 'utf8' });
  if (r.status >= 8) throw new Error(`robocopy failed ${from} -> ${to}: ${r.stdout}${r.stderr}`);
}
function junction(link, target) {
  if (existsSync(link)) return;
  const r = spawnSync('cmd', ['/c', 'mklink', '/J', link, target], { encoding: 'utf8' });
  if (r.status !== 0) throw new Error(`mklink failed: ${r.stdout}${r.stderr}`);
}
function removeDir(p) {
  if (!existsSync(p)) return;
  // remove the node_modules junction first so rm never follows it
  const nm = join(p, 'node_modules');
  if (existsSync(nm)) spawnSync('cmd', ['/c', 'rmdir', nm]);
  rmSync(p, { recursive: true, force: true });
}
const EXCLUDE_DIRS = ['/XD', 'node_modules', 'dist', '.astro', '.impeccable', '.claude', 'logs', '.cache'];

function snapshotMain(dest) {
  mkdirSync(dest, { recursive: true });
  robocopy(MAIN, dest, ['/E', ...EXCLUDE_DIRS]);
  junction(join(dest, 'node_modules'), join(MAIN, 'node_modules'));
}

/** The only paths a module may contribute. */
function modulePaths(mod) {
  return {
    dirs: [`src/content/lessons/${mod.slug}`, `src/components/viz/${mod.slug}`],
    files: [`src/content/cheatsheets/${mod.slug}.mdx`, `src/data/glossary/${mod.slug}.ts`],
    notebookPrefix: `${mod.num}-`,
  };
}

function copyModule(mod, fromRoot, toRoot) {
  const p = modulePaths(mod);
  for (const d of p.dirs) {
    const src = join(fromRoot, d);
    if (existsSync(src)) { mkdirSync(join(toRoot, d), { recursive: true }); robocopy(src, join(toRoot, d), ['/E']); }
  }
  for (const f of p.files) {
    const src = join(fromRoot, f);
    if (existsSync(src)) { mkdirSync(join(toRoot, f, '..'), { recursive: true }); writeFileSync(join(toRoot, f), readFileSync(src)); }
  }
  const nbDir = join(fromRoot, 'notebooks-src');
  if (existsSync(nbDir)) {
    for (const f of readdirSync(nbDir)) {
      if (f.startsWith(p.notebookPrefix) && f.endsWith('.py')) {
        mkdirSync(join(toRoot, 'notebooks-src'), { recursive: true });
        writeFileSync(join(toRoot, 'notebooks-src', f), readFileSync(join(nbDir, f)));
      }
    }
  }
}

// ───────────── verification
function checkContent(mod, root) {
  const problems = [];
  for (const l of mod.lessons) {
    const f = join(root, 'src/content/lessons', mod.slug, `${l.slug}.mdx`);
    if (!existsSync(f)) { problems.push(`Missing file src/content/lessons/${mod.slug}/${l.slug}.mdx (${l.kind}: "${l.title}")`); continue; }
    const s = readFileSync(f, 'utf8');
    // unfinished-writing markers only; labs legitimately contain '# TODO:' exercises and the word 'placeholder'
    if (/coming soon|lorem ipsum|\bTBD\b|\[(?:TODO|PLACEHOLDER)\]|\{\/\*\s*TODO/i.test(s)) problems.push(`${l.slug}.mdx contains unfinished-writing markers (coming soon / TBD / [TODO])`);
    if (l.kind === 'lesson') {
      if (s.length < 9000) problems.push(`${l.slug}.mdx is too short (${s.length} chars). Lessons need ~1,800-3,000 words of real teaching.`);
      if (!s.includes('<Scrolly')) problems.push(`${l.slug}.mdx has no <Scrolly> visual`);
      if (!s.includes('<Quiz')) problems.push(`${l.slug}.mdx has no <Quiz>`);
      if (!s.includes('<KeyNotes')) problems.push(`${l.slug}.mdx has no <KeyNotes>`);
      if (!new RegExp(`@/components/viz/${mod.slug}/`).test(s)) problems.push(`${l.slug}.mdx does not import a visual from src/components/viz/${mod.slug}/`);
    } else {
      if (s.length < 4500) problems.push(`${l.slug}.mdx (${l.kind}) is too short (${s.length} chars)`);
      if (!s.includes('<KeyNotes')) problems.push(`${l.slug}.mdx has no <KeyNotes>`);
      const nb = s.match(/^colab:\s*['"]?([^'"\n]+)/m);
      if (nb) {
        const py = join(root, 'notebooks-src', nb[1].trim().replace(/\.ipynb$/, '.py'));
        if (!existsSync(py)) problems.push(`${l.slug}.mdx references notebook ${nb[1].trim()} but notebooks-src/${nb[1].trim().replace(/\.ipynb$/, '.py')} does not exist`);
        if (!nb[1].trim().startsWith(`${mod.num}-`)) problems.push(`Notebook name ${nb[1].trim()} must start with "${mod.num}-"`);
      }
    }
  }
  if (!existsSync(join(root, 'src/content/cheatsheets', `${mod.slug}.mdx`))) problems.push(`Missing cheat sheet src/content/cheatsheets/${mod.slug}.mdx`);
  return problems;
}

let buildLock = Promise.resolve();
function withBuildLock(fn) { const p = buildLock.then(fn, fn); buildLock = p.catch(() => {}); return p; }

function runBuild(root, label) {
  return new Promise((res) => {
    const env = { ...process.env, PWD: root, INIT_CWD: root, ASTRO_CACHE_DIR: join(root, '.cache', 'astro'), VITE_CACHE_DIR: join(root, '.cache', 'vite'), ASTRO_TELEMETRY_DISABLED: '1' };
    const child = spawn('npm run build', { cwd: root, shell: true, env });
    let out = '';
    child.stdout.on('data', (d) => (out += d));
    child.stderr.on('data', (d) => (out += d));
    const t = setTimeout(() => { killTree(child.pid); }, 15 * 60 * 1000);
    child.on('close', (code) => {
      clearTimeout(t);
      writeFileSync(join(LOGS, `build-${label}.log`), out);
      // eslint-disable-next-line no-control-regex
      const clean = out.replace(/\x1b\[[0-9;]*m/g, '');
      const errIdx = clean.search(/\[ERROR\]|Error:|error TS|Could not parse|Build failed/);
      res({ ok: code === 0, tail: errIdx >= 0 ? clean.slice(errIdx, errIdx + 2500) : clean.slice(-2500) });
    });
  });
}

/** Stage = fresh copy of main + this module only; build must pass there. */
async function verify(mod, workerRoot, attempt) {
  const problems = checkContent(mod, workerRoot);
  const stage = join(WORK, `_stage-${mod.slug}`);
  removeDir(stage);
  snapshotMain(stage);
  copyModule(mod, workerRoot, stage);
  const b = await runBuild(stage, `${mod.slug}-a${attempt}`);
  if (!b.ok) problems.push(`npm run build FAILED when your module was dropped into a clean copy of the site (shared files reset to the originals). Error:\n${b.tail}`);
  return { problems, stage };
}

// ───────────── agent runs
function killTree(pid) { try { spawnSync('taskkill', ['/PID', String(pid), '/T', '/F']); } catch {} }

function runAgent(tool, mod, root, attempt) {
  const prompt = attempt === 1
    ? `Your working directory is ${root}. Read ${join(root, 'TASK.md')} and complete it fully and carefully, working only inside ${root}. Do not stop until every checklist item in TASK.md is done and npm run build passes with zero errors.`
    : `Your working directory is ${root}. Read ${join(root, 'TASK.md')} and then ${join(root, 'FIXES.md')}. FIXES.md lists problems found when your module was verified. Fix every one of them, finish anything incomplete, and do not stop until npm run build passes with zero errors.`;
  const cmd = TOOLS[tool](prompt, root);
  return new Promise((res) => {
    const env = { ...process.env, PWD: root, INIT_CWD: root, OLDPWD: root, ASTRO_CACHE_DIR: join(root, '.cache', 'astro'), VITE_CACHE_DIR: join(root, '.cache', 'vite'), ASTRO_TELEMETRY_DISABLED: '1', CI: '1' };
    const logf = join(LOGS, `${mod.slug}-a${attempt}-${tool}.log`);
    writeFileSync(logf, `$ ${cmd}\n(cwd ${root})\n\n`);
    const child = spawn(cmd, { cwd: root, shell: true, env, stdio: ['ignore', 'pipe', 'pipe'] });
    child.stdout.on('data', (d) => appendFileSync(logf, d));
    child.stderr.on('data', (d) => appendFileSync(logf, d));
    const t = setTimeout(() => { appendFileSync(logf, '\n[orchestrator] timeout, killing\n'); killTree(child.pid); }, AGENT_TIMEOUT_MIN * 60 * 1000);
    child.on('close', (code) => { clearTimeout(t); res(code); });
  });
}

function taskFile(mod) {
  const lessons = mod.lessons.map((l, i) => `  ${i + 1}. [${l.kind}] slug \`${l.slug}\`: "${l.title}" → src/content/lessons/${mod.slug}/${l.slug}.mdx`).join('\n');
  const all = readModules();
  const idx = all.findIndex((m) => m.slug === mod.slug);
  const before = all.slice(Math.max(0, idx - 3), idx).map((m) => `${m.num} ${m.title}`).join('; ');
  const after = all.slice(idx + 1, idx + 3).map((m) => `${m.num} ${m.title}`).join('; ');
  return `# Your task: write Module ${mod.num} · ${mod.title}

You are one of several authors writing **Tokens per Second**, an inference engineering course website (Astro + MDX + React).
Your module: **${mod.num} · ${mod.title}** (slug \`${mod.slug}\`). Tagline: "${mod.tagline.replace(/\\u2019/g, '’')}"

Start by reading **AUTHORING.md** (all of it). Then study the finished reference modules it lists, and \`src/data/course.ts\` for where your module sits.
The learner has already finished every earlier module (most recently: ${before || 'none'}). Coming next: ${after || 'nothing, this is the final module'}. Build on what came before, refer back to it, and do not re-teach it at length.

## Lessons you must write (all of them)
${lessons}

## Files you own (do not touch anything else)
- src/content/lessons/${mod.slug}/*.mdx
- src/components/viz/${mod.slug}/* (your visuals and helpers)
- src/content/cheatsheets/${mod.slug}.mdx
- src/data/glossary/${mod.slug}.ts
- notebooks-src/${mod.num}-*.py (Colab notebooks for GPU labs, named \`${mod.num}-<short-name>.py\`; reference them as \`${mod.num}-<short-name>.ipynb\`)

Anything you change outside these paths is thrown away, and your module is tested against the original shared files. So never depend on edits to shared files.

## Checklist (all required)
- [ ] All ${mod.lessons.length} MDX files above exist, complete, matching AUTHORING.md's anatomy and length targets.
- [ ] Each lesson has its own visual in src/components/viz/${mod.slug}/ driven by a <Scrolly>; STEPS count equals the number of <Step>s.
- [ ] Cheat sheet src/content/cheatsheets/${mod.slug}.mdx.
- [ ] Notebooks for GPU labs (runs on a free Colab T4), referenced via \`colab:\` frontmatter and <Colab nb="...">.
- [ ] Glossary terms in src/data/glossary/${mod.slug}.ts.
- [ ] \`node scripts/fix-dollars.mjs src/content/lessons/${mod.slug}\` run.
- [ ] \`npm run build\` passes with zero errors. Run it, fix errors, and run it again until clean.

Work through the lessons in order, building after each one so errors stay small. Quality bar: as good as the reference lessons. This is someone's path to a career.
`;
}

async function processModule(mod, toolOrder) {
  const root = join(WORK, mod.slug);
  status[mod.slug] = { state: 'preparing', attempts: 0 };
  saveStatus();
  if (!existsSync(join(root, 'AUTHORING.md'))) { snapshotMain(root); }
  else { // refresh shared files but keep the module's own work
    robocopy(MAIN, root, ['/E', ...EXCLUDE_DIRS, join(MAIN, 'src/content/lessons', mod.slug), join(MAIN, 'src/components/viz', mod.slug)]);
    junction(join(root, 'node_modules'), join(MAIN, 'node_modules'));
  }
  writeFileSync(join(root, 'TASK.md'), taskFile(mod));

  for (let attempt = 1; attempt <= ATTEMPTS; attempt++) {
    const tool = pickTool(toolOrder, attempt);
    status[mod.slug] = { state: 'writing', tool, attempts: attempt };
    saveStatus();
    log(`${mod.num} ${mod.slug}: attempt ${attempt} with ${tool}`);
    const code = await runAgent(tool, mod, root, attempt);
    log(`${mod.num} ${mod.slug}: ${tool} exited ${code}; verifying`);
    status[mod.slug] = { state: 'verifying', tool, attempts: attempt };
    saveStatus();
    const { problems, stage } = await withBuildLock(() => verify(mod, root, attempt));
    removeDir(stage);
    if (problems.length === 0) {
      await withBuildLock(async () => {
        copyModule(mod, root, MAIN);
        const b = await runBuild(MAIN, `main-after-${mod.slug}`);
        log(`${mod.num} ${mod.slug}: MERGED into main (main build ${b.ok ? 'ok' : 'FAILED: ' + b.tail.slice(0, 300)})`);
      });
      status[mod.slug] = { state: 'done', tool, attempts: attempt };
      saveStatus();
      return true;
    }
    log(`${mod.num} ${mod.slug}: ${problems.length} problem(s); first: ${problems[0].slice(0, 160).replace(/\n/g, ' ')}`);
    writeFileSync(join(root, 'FIXES.md'), `# Problems found when verifying Module ${mod.num} (attempt ${attempt})\n\nFix every item, then run \`npm run build\` until it passes.\n\n${problems.map((p) => `- ${p}`).join('\n')}\n`);
  }
  status[mod.slug] = { state: 'failed', attempts: ATTEMPTS };
  saveStatus();
  log(`${mod.num} ${mod.slug}: FAILED after ${ATTEMPTS} attempts`);
  return false;
}

// ───────────── main
const all = readModules();
// a module is already finished in main when every lesson file and its cheat sheet exist
const finished = (m) => m.lessons.every((l) => existsSync(join(MAIN, 'src/content/lessons', m.slug, `${l.slug}.mdx`))) && existsSync(join(MAIN, 'src/content/cheatsheets', `${m.slug}.mdx`));
const todo = all.filter((m) => (ONLY.length ? ONLY.includes(m.slug) : !finished(m)));
if (process.argv.includes('--dry')) {
  for (const m of all) console.log(m.num, m.slug, m.lessons.length, finished(m) ? 'finished' : 'TODO');
  console.log('modules', all.length, 'lessons', all.reduce((a, m) => a + m.lessons.length, 0), 'todo', todo.length);
  process.exit(0);
}
log(`syllabus: ${all.length} modules; to build: ${todo.map((m) => m.num).join(', ')}; concurrency ${CONCURRENCY}`);
for (const m of all) if (!todo.includes(m)) status[m.slug] = { state: 'done', attempts: 0 };
saveStatus();

let next = 0;
let slot = 0;
async function worker() {
  const mySlot = slot++;
  while (next < todo.length) {
    const mod = todo[next++];
    // alternate primary tool across slots; each module retries with the other tool
    const order = mySlot % 2 === 0 ? ['opencode', 'cmdc'] : ['cmdc', 'opencode'];
    try { await processModule(mod, order); }
    catch (e) { log(`${mod.slug}: crashed: ${e.stack || e}`); status[mod.slug] = { state: 'failed', error: String(e) }; saveStatus(); }
  }
}
await Promise.all(Array.from({ length: Math.min(CONCURRENCY, todo.length) }, worker));
const failed = Object.entries(status).filter(([, s]) => s.state !== 'done').map(([k]) => k);
log(`ALL DONE. failed: ${failed.length ? failed.join(', ') : 'none'}`);
process.exit(failed.length ? 1 : 0);
