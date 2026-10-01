import { useEffect, useState } from 'react';
import { allNotes, load, setNotes } from '../../scripts/progress';
import { Icon } from '../viz/lib/Icon';

interface LessonRef { key: string; code: string; title: string; href: string; module: string }

/** Every personal note, in course order, with export to Markdown. */
export default function AllNotes({ lessons }: { lessons: LessonRef[] }) {
  const [notes, setAll] = useState<Record<string, string>>({});
  const [done, setDone] = useState(0);
  useEffect(() => {
    const refresh = () => { setAll(allNotes()); setDone(Object.keys(load().done).length); };
    refresh();
    window.addEventListener('tps:state', refresh);
    return () => window.removeEventListener('tps:state', refresh);
  }, []);

  const withNotes = lessons.filter((l) => notes[l.key]?.trim());
  const exportMd = () => {
    const md = ['# My notes: Tokens per Second', '', ...withNotes.flatMap((l) => [`## ${l.code} ${l.title}`, '', notes[l.key].trim(), ''])].join('\n');
    const blob = new Blob([md], { type: 'text/markdown' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'tokens-per-second-notes.md';
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <div className="allnotes">
      <div className="an-bar">
        <span>{withNotes.length} lessons with notes · {done} lessons completed</span>
        <div className="an-actions">
          <button type="button" className="btn ghost" onClick={() => window.print()} disabled={!withNotes.length}>Print</button>
          <button type="button" className="btn" onClick={exportMd} disabled={!withNotes.length}>
            <Icon name="download" size={15} /> Export as Markdown
          </button>
        </div>
      </div>
      {withNotes.length === 0 ? (
        <div className="an-empty">
          <p><strong>No notes yet.</strong></p>
          <p>Every lesson has a <em>My notes</em> button in the bottom-right corner. Write what you learned in your own words, or press <em>Copy into my notes</em> under a lesson's key notes. Everything you write collects here.</p>
        </div>
      ) : (
        withNotes.map((l) => (
          <article key={l.key} className="an-item">
            <header>
              <a href={l.href}><span className="code">{l.code}</span> {l.title}</a>
              <span className="mod">{l.module}</span>
            </header>
            <textarea
              defaultValue={notes[l.key]}
              aria-label={`Notes for ${l.title}`}
              onChange={(e) => setNotes(l.key, e.target.value)}
              rows={Math.min(16, Math.max(4, notes[l.key].split('\n').length + 1))}
            />
          </article>
        ))
      )}
      <style>{`
        .an-bar { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 1rem; padding: 1rem 0; border-top: 1px solid var(--rule-2); border-bottom: 1px solid var(--rule); font-size: .85rem; color: var(--text-2); }
        .an-actions { display: flex; gap: .6rem; }
        .an-actions .btn { font-size: .82rem; padding: .6em .9em; }
        .an-empty { padding: 2rem 0; color: var(--text-2); max-width: 60ch; }
        .an-item { padding: 1.4rem 0; border-bottom: 1px solid var(--rule); }
        .an-item header { display: flex; justify-content: space-between; gap: 1rem; flex-wrap: wrap; margin-bottom: .6rem; }
        .an-item header a { font-weight: 700; text-decoration: none; }
        .an-item header a:hover { text-decoration: underline; }
        .code { font-family: var(--font-mono); color: var(--text-3); font-weight: 400; margin-right: .3rem; }
        .mod { font-size: .78rem; color: var(--text-3); }
        .an-item textarea { width: 100%; font: .93rem/1.6 var(--font-sans); color: var(--text); background: var(--ground-2); border: 1px solid var(--rule); border-radius: 4px; padding: .8rem .9rem; resize: vertical; }
        .an-item textarea:focus { outline: 2px solid var(--memory); outline-offset: 1px; }
        @media print { .an-bar { display: none; } .an-item textarea { border: 0; background: none; color: #111; resize: none; } }
      `}</style>
    </div>
  );
}
