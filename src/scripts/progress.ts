// Learner state lives in this browser only (no accounts). Everything is namespaced "tps:".
export interface LastLesson { key: string; href: string; code: string; title: string; at: number }
export interface QuizResult { score: number; total: number; at: number }
export interface State { done: Record<string, number>; last?: LastLesson; quiz: Record<string, QuizResult> }

const KEY = 'tps:state:v1';
const NOTES = 'tps:notes:';

function safeGet(k: string): string | null { try { return localStorage.getItem(k); } catch { return null; } }
function safeSet(k: string, v: string) { try { localStorage.setItem(k, v); } catch { /* private mode: ignore */ } }

export function load(): State {
  const raw = safeGet(KEY);
  if (!raw) return { done: {}, quiz: {} };
  try { const s = JSON.parse(raw); return { done: s.done || {}, quiz: s.quiz || {}, last: s.last }; } catch { return { done: {}, quiz: {} }; }
}
export function save(s: State) { safeSet(KEY, JSON.stringify(s)); window.dispatchEvent(new CustomEvent('tps:state')); }

export function setDone(key: string, done: boolean) { const s = load(); if (done) s.done[key] = Date.now(); else delete s.done[key]; save(s); }
export function setLast(last: Omit<LastLesson, 'at'>) { const s = load(); s.last = { ...last, at: Date.now() }; save(s); }
export function setQuiz(key: string, score: number, total: number) { const s = load(); const prev = s.quiz[key]; if (!prev || score >= prev.score) s.quiz[key] = { score, total, at: Date.now() }; save(s); }

export function getNotes(key: string) { return safeGet(NOTES + key) || ''; }
export function setNotes(key: string, text: string) { if (text.trim()) safeSet(NOTES + key, text); else { try { localStorage.removeItem(NOTES + key); } catch {} } window.dispatchEvent(new CustomEvent('tps:state')); }
export function allNotes(): Record<string, string> {
  const out: Record<string, string> = {};
  try { for (let i = 0; i < localStorage.length; i++) { const k = localStorage.key(i)!; if (k.startsWith(NOTES)) out[k.slice(NOTES.length)] = localStorage.getItem(k) || ''; } } catch {}
  return out;
}

/** Paint progress marks onto any element carrying data-lesson-key. State is a mark, not only a colour. */
export function applyMarks(root: ParentNode = document) {
  const s = load();
  const notes = allNotes();
  root.querySelectorAll<HTMLElement>('[data-lesson-key]').forEach((el) => {
    const k = el.dataset.lessonKey!;
    el.classList.toggle('is-done', !!s.done[k]);
    el.classList.toggle('is-flag', s.last?.key === k);
    el.classList.toggle('has-notes', !!notes[k]);
    const q = s.quiz[k];
    el.classList.toggle('has-quiz', !!q);
    if (q) el.dataset.quiz = `${q.score}/${q.total}`; else delete el.dataset.quiz;
  });
  root.querySelectorAll<HTMLElement>('[data-module-progress]').forEach((el) => {
    const keys = (el.dataset.moduleProgress || '').split(',').filter(Boolean);
    const n = keys.filter((k) => s.done[k]).length;
    el.textContent = `${n}/${keys.length}`;
    el.classList.toggle('is-complete', n === keys.length && n > 0);
  });
  root.querySelectorAll<HTMLAnchorElement>('[data-resume]').forEach((a) => {
    if (!s.last) { a.hidden = a.dataset.resume === 'hide'; return; }
    a.hidden = false;
    a.href = s.last.href;
    const label = a.querySelector('[data-resume-label]');
    if (label) label.textContent = `${s.last.code} ${s.last.title}`;
  });
}
