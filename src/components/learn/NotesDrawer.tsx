import { useEffect, useRef, useState } from 'react';
import { Icon } from '../viz/lib/Icon';
import { getNotes, setNotes } from '../../scripts/progress';

/** Personal notes for this lesson, saved in the browser as you type. */
export default function NotesDrawer({ lessonKey, title }: { lessonKey: string; title: string }) {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState('');
  const [saved, setSaved] = useState(true);
  const area = useRef<HTMLTextAreaElement>(null);

  useEffect(() => setText(getNotes(lessonKey)), [lessonKey]);

  useEffect(() => {
    const onAppend = (e: Event) => {
      const add = (e as CustomEvent<string>).detail;
      setText((t) => {
        const next = (t.trim() ? t.trimEnd() + '\n\n' : '') + add;
        setNotes(lessonKey, next);
        return next;
      });
      setOpen(true);
    };
    window.addEventListener('tps:append-notes', onAppend);
    return () => window.removeEventListener('tps:append-notes', onAppend);
  }, [lessonKey]);

  useEffect(() => {
    if (saved) return;
    const t = window.setTimeout(() => { setNotes(lessonKey, text); setSaved(true); }, 400);
    return () => window.clearTimeout(t);
  }, [text, saved, lessonKey]);

  useEffect(() => {
    if (open) area.current?.focus();
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open]);

  return (
    <>
      <button type="button" className={`notes-fab ${text.trim() ? 'has' : ''}`} onClick={() => setOpen(!open)} aria-expanded={open} aria-controls="notes-panel">
        <Icon name="notes" size={18} />
        <span>My notes</span>
      </button>
      <aside id="notes-panel" className={`notes-panel ${open ? 'open' : ''}`} aria-label="My notes for this lesson" hidden={!open}>
        <header>
          <div>
            <strong>My notes</strong>
            <span className="np-sub">{title}</span>
          </div>
          <button type="button" className="np-close" onClick={() => setOpen(false)} aria-label="Close notes">
            <Icon name="close" size={18} />
          </button>
        </header>
        <textarea
          ref={area}
          value={text}
          placeholder={'Write it in your own words. Explaining it back is how it sticks.\n\nSaved in this browser as you type.'}
          onChange={(e) => { setText(e.target.value); setSaved(false); }}
        />
        <footer>
          <span>{saved ? 'Saved' : 'Saving…'}</span>
          <a href={(import.meta.env.BASE_URL.replace(/\/?$/, '/')) + 'notes/'}>All my notes</a>
        </footer>
      </aside>
      <style>{`
        .notes-fab { position: fixed; left: 20px; bottom: 20px; z-index: 40; display: flex; align-items: center; gap: .45rem; font: 700 .8rem var(--font-sans); color: var(--text); background: var(--ground-3); border: 1px solid var(--rule-2); border-radius: 999px; padding: .6rem .95rem; cursor: pointer; box-shadow: 0 8px 24px rgb(0 0 0 / .45); transition: border-color .2s, transform .25s var(--ease-out); }
        .notes-fab:hover { border-color: var(--text-2); transform: translateY(-1px); }
        .notes-fab.has::after { content: ''; width: 6px; height: 6px; border-radius: 50%; background: var(--compute); }
        .notes-panel { position: fixed; left: 20px; bottom: 76px; z-index: 41; width: min(420px, calc(100vw - 40px)); height: min(520px, calc(100vh - 160px)); display: flex; flex-direction: column; background: var(--ground-2); border: 1px solid var(--rule-2); border-radius: 6px; box-shadow: 0 18px 50px rgb(0 0 0 / .55); }
        .notes-panel[hidden] { display: none; }
        .notes-panel header { display: flex; justify-content: space-between; align-items: start; padding: .8rem .9rem .6rem; border-bottom: 1px solid var(--rule); }
        .notes-panel header strong { display: block; font-size: .9rem; }
        .np-sub { font-size: .75rem; color: var(--text-2); }
        .np-close { background: none; border: 0; color: var(--text-2); cursor: pointer; padding: .2rem; }
        .np-close:hover { color: var(--text); }
        .notes-panel textarea { flex: 1; resize: none; border: 0; background: transparent; color: var(--text); font: .92rem/1.6 var(--font-sans); padding: .9rem; outline: none; }
        .notes-panel footer { display: flex; justify-content: space-between; padding: .55rem .9rem; border-top: 1px solid var(--rule); font-size: .74rem; color: var(--text-3); }
        .notes-panel footer a { color: var(--text-2); }
        @media (max-width: 900px) { .notes-fab { left: auto; right: 16px; bottom: 16px; } .notes-panel { left: auto; right: 16px; } }
        @media print { .notes-fab, .notes-panel { display: none !important; } }
      `}</style>
    </>
  );
}
