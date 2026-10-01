import type { ReactNode } from 'react';
/** Tiny inline formatter for quiz text: `code`, **bold**, *italic*. */
export function inline(s: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = re.exec(s))) {
    if (m.index > last) out.push(s.slice(last, m.index));
    const t = m[0];
    if (t.startsWith('`')) out.push(<code key={i++}>{t.slice(1, -1)}</code>);
    else if (t.startsWith('**')) out.push(<strong key={i++}>{t.slice(2, -2)}</strong>);
    else out.push(<em key={i++}>{t.slice(1, -1)}</em>);
    last = m.index + t.length;
  }
  if (last < s.length) out.push(s.slice(last));
  return out;
}

export function lessonKeyFromPath(): string {
  const p = window.location.pathname;
  const i = p.indexOf('/learn/');
  return i >= 0 ? p.slice(i + 7).replace(/\/$/, '') : p;
}
