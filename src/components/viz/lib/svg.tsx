import type { CSSProperties, ReactNode } from 'react';
import type { Kind } from './Stage';

export const COLOR: Record<Kind, string> = {
  compute: 'var(--compute)',
  memory: 'var(--memory)',
  token: 'var(--token)',
  comm: 'var(--comm)',
  idle: 'var(--idle)',
  error: 'var(--error)',
  text: 'var(--text)',
};
export const DIM: Record<Kind, string> = {
  compute: 'var(--compute-dim)',
  memory: 'var(--memory-dim)',
  token: 'var(--token-dim)',
  comm: 'var(--comm-dim)',
  idle: 'rgb(74 82 98 / 0.25)',
  error: 'var(--error-dim)',
  text: 'rgb(232 230 225 / 0.08)',
};

/** A unit of work on a track. `ghost` = a possibility not (yet) taken: outline only. */
export function Span({
  x, y, w, h, kind = 'compute', label, ghost, dim, opacity = 1, labelTone, rx = 2, style,
}: {
  x: number; y: number; w: number; h: number; kind?: Kind; label?: ReactNode; ghost?: boolean; dim?: boolean;
  opacity?: number; labelTone?: 'dark' | 'light'; rx?: number; style?: CSSProperties;
}) {
  const fill = kind === 'idle' ? 'url(#tps-hatch)' : ghost ? 'none' : dim ? DIM[kind] : COLOR[kind];
  const stroke = ghost ? 'var(--rule-2)' : kind === 'idle' ? 'var(--idle)' : COLOR[kind];
  const approx = typeof label === 'string' ? label.length * 6.3 + 8 : 18;
  const showLabel = label !== undefined && w > approx;
  const dark = labelTone ? labelTone === 'dark' : !ghost && !dim && kind !== 'idle';
  return (
    <g className="v-span" style={{ opacity, ...style }}>
      <rect x={x} y={y} width={Math.max(0, w)} height={h} rx={rx} fill={fill} stroke={stroke} strokeWidth={1} strokeDasharray={ghost ? '3 3' : undefined} />
      {showLabel && (
        <text x={x + w / 2} y={y + h / 2 + 3.8} textAnchor="middle" className="v-label" style={{ fill: dark ? '#0e1116' : ghost ? 'var(--text-3)' : 'var(--text)', fontSize: 10.5, fontWeight: 600 }}>
          {label}
        </text>
      )}
    </g>
  );
}

export function Label({
  x, y, children, anchor = 'start', tone = 2, size, mono = true, bold, kind, math, style, className = '',
}: {
  x: number; y: number; children: ReactNode; anchor?: 'start' | 'middle' | 'end'; tone?: 1 | 2 | 3; size?: number; mono?: boolean;
  bold?: boolean; kind?: Kind; math?: boolean; style?: CSSProperties; className?: string;
}) {
  const cls = ['v-label', `t${tone}`, mono ? '' : 'sans', bold ? 'bold' : '', math ? 'math' : '', className].join(' ');
  return (
    <text x={x} y={y} textAnchor={anchor} className={cls} style={{ fontSize: size, fill: kind ? COLOR[kind] : undefined, ...style }}>
      {children}
    </text>
  );
}

export function Arrow({
  x1, y1, x2, y2, kind, dashed, curve = 0, muted, width = 1.4, opacity = 1,
}: { x1: number; y1: number; x2: number; y2: number; kind?: Kind; dashed?: boolean; curve?: number; muted?: boolean; width?: number; opacity?: number }) {
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  // perpendicular offset for a gentle curve
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.hypot(dx, dy) || 1;
  const cx = mx - (dy / len) * curve;
  const cy = my + (dx / len) * curve;
  const d = curve ? `M${x1},${y1} Q${cx},${cy} ${x2},${y2}` : `M${x1},${y1} L${x2},${y2}`;
  const stroke = muted ? 'var(--text-3)' : kind ? COLOR[kind] : 'var(--text-2)';
  return (
    <path d={d} className="v-arrow v-fade" stroke={stroke} strokeWidth={width} strokeDasharray={dashed ? '4 4' : undefined} markerEnd="url(#tps-arrow)" style={{ opacity }} />
  );
}

/** A horizontal time ruler with ticks. */
export function Ruler({ x, y, w, ticks, label }: { x: number; y: number; w: number; ticks: { at: number; label?: string }[]; label?: string }) {
  return (
    <g>
      <line x1={x} x2={x + w} y1={y} y2={y} className="v-rule" />
      {ticks.map((t, i) => (
        <g key={i}>
          <line x1={x + t.at * w} x2={x + t.at * w} y1={y} y2={y + (t.label ? 5 : 3)} className="v-rule" />
          {t.label && <Label x={x + t.at * w} y={y + 16} anchor="middle" tone={3} size={9.5}>{t.label}</Label>}
        </g>
      ))}
      {label && <Label x={x + w} y={y - 6} anchor="end" tone={3} size={9.5}>{label}</Label>}
    </g>
  );
}

/** Track: a labelled lane. Children draw spans inside it. */
export function Track({ x, y, w, h, label, children }: { x: number; y: number; w: number; h: number; label?: string; children?: ReactNode }) {
  return (
    <g>
      <line x1={x} x2={x + w} y1={y + h + 3} y2={y + h + 3} className="v-rule faint" />
      {label && <text x={x - 10} y={y + h / 2 + 4} textAnchor="end" className="v-track-label">{label}</text>}
      {children}
    </g>
  );
}

/** Dimension line: |←— label —→| */
export function Dim({ x1, x2, y, label, vertical, kind }: { x1: number; x2: number; y: number; label: ReactNode; vertical?: boolean; kind?: Kind }) {
  const c = kind ? COLOR[kind] : 'var(--text-3)';
  if (vertical) {
    // x1..x2 are y-range, y is the x position
    return (
      <g>
        <line x1={y} x2={y} y1={x1} y2={x2} stroke={c} strokeWidth={1} />
        <line x1={y - 4} x2={y + 4} y1={x1} y2={x1} stroke={c} strokeWidth={1} />
        <line x1={y - 4} x2={y + 4} y1={x2} y2={x2} stroke={c} strokeWidth={1} />
        <Label x={y - 8} y={(x1 + x2) / 2 + 4} anchor="end" tone={2} size={10} style={{ fill: kind ? c : undefined }}>{label}</Label>
      </g>
    );
  }
  return (
    <g>
      <line x1={x1} x2={x2} y1={y} y2={y} stroke={c} strokeWidth={1} />
      <line x1={x1} x2={x1} y1={y - 4} y2={y + 4} stroke={c} strokeWidth={1} />
      <line x1={x2} x2={x2} y1={y - 4} y2={y + 4} stroke={c} strokeWidth={1} />
      <rect x={(x1 + x2) / 2 - 34} y={y - 8} width={68} height={16} fill="var(--ground-2)" />
      <Label x={(x1 + x2) / 2} y={y + 3.5} anchor="middle" tone={2} size={10} style={{ fill: kind ? c : undefined }}>{label}</Label>
    </g>
  );
}

/** A token chip: rounded box with text. */
export function Chip({ x, y, text, kind = 'text', w, h = 22, ghost, dim, opacity = 1, mono = true, size = 11 }: {
  x: number; y: number; text: ReactNode; kind?: Kind; w?: number; h?: number; ghost?: boolean; dim?: boolean; opacity?: number; mono?: boolean; size?: number;
}) {
  const width = w ?? Math.max(24, String(text).length * (size * 0.62) + 14);
  const filled = !ghost && !dim;
  return (
    <g className="v-fade" style={{ opacity }}>
      <rect x={x} y={y} width={width} height={h} rx={3}
        fill={ghost ? 'none' : dim ? DIM[kind] : kind === 'text' ? 'var(--ground-3)' : COLOR[kind]}
        stroke={ghost ? 'var(--rule-2)' : kind === 'text' ? 'var(--rule-2)' : COLOR[kind]}
        strokeDasharray={ghost ? '3 3' : undefined} />
      <text x={x + width / 2} y={y + h / 2 + size * 0.36} textAnchor="middle" className={`v-label ${mono ? '' : 'sans'}`}
        style={{ fontSize: size, fill: filled && kind !== 'text' ? '#0e1116' : ghost ? 'var(--text-3)' : 'var(--text)', fontWeight: 600 }}>
        {text}
      </text>
    </g>
  );
}

/** Group that glides to (x, y) with a CSS transform transition. */
export function Move({ x, y, children, opacity = 1, scale = 1, dur }: { x: number; y: number; children: ReactNode; opacity?: number; scale?: number; dur?: number }) {
  return (
    <g className="v-move" style={{ transform: `translate(${x}px, ${y}px) scale(${scale})`, opacity, transitionDuration: dur ? `${dur}ms` : undefined }}>
      {children}
    </g>
  );
}

export function Fade({ show, children, dur }: { show: boolean; children: ReactNode; dur?: number }) {
  return (
    <g className="v-fade" style={{ opacity: show ? 1 : 0, pointerEvents: show ? undefined : 'none', transitionDuration: dur ? `${dur}ms` : undefined }} aria-hidden={!show}>
      {children}
    </g>
  );
}

/** Grid of cells (a matrix / tensor slice). fill(r,c) returns a kind, 'dim:<kind>', or null for empty. */
export function Matrix({
  x, y, rows, cols, cell = 14, gap = 2, fill, text, label, labelBelow, stroke,
}: {
  x: number; y: number; rows: number; cols: number; cell?: number; gap?: number;
  fill: (r: number, c: number) => Kind | `dim:${Kind}` | 'empty' | null;
  text?: (r: number, c: number) => string | null; label?: ReactNode; labelBelow?: ReactNode; stroke?: string;
}) {
  const cells: ReactNode[] = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const f = fill(r, c);
      let fillC = 'transparent';
      let strokeC = stroke ?? 'var(--rule-2)';
      if (f && f !== 'empty') {
        if (f.startsWith('dim:')) { const k = f.slice(4) as Kind; fillC = DIM[k]; strokeC = COLOR[k]; }
        else { fillC = COLOR[f as Kind]; strokeC = COLOR[f as Kind]; }
      }
      const cx = x + c * (cell + gap);
      const cy = y + r * (cell + gap);
      const t = text?.(r, c);
      cells.push(
        <g key={`${r}-${c}`}>
          <rect x={cx} y={cy} width={cell} height={cell} rx={1.5} className="v-cell" fill={fillC} stroke={strokeC} strokeWidth={0.8} />
          {t && <text x={cx + cell / 2} y={cy + cell / 2 + 3.3} textAnchor="middle" className="v-label" style={{ fontSize: Math.min(10, cell * 0.55), fill: f && f !== 'empty' && !f.startsWith('dim:') ? '#0e1116' : 'var(--text)' }}>{t}</text>}
        </g>,
      );
    }
  }
  const W = cols * (cell + gap) - gap;
  const H = rows * (cell + gap) - gap;
  return (
    <g>
      {label && <Label x={x + W / 2} y={y - 8} anchor="middle" tone={2} size={10.5}>{label}</Label>}
      {cells}
      {labelBelow && <Label x={x + W / 2} y={y + H + 15} anchor="middle" tone={3} size={10}>{labelBelow}</Label>}
    </g>
  );
}

/** Horizontal bar with a label and value, for memory / quantity comparisons. */
export function Bar({ x, y, w, h = 14, frac, kind = 'memory', label, value, ghost }: {
  x: number; y: number; w: number; h?: number; frac: number; kind?: Kind; label?: string; value?: string; ghost?: boolean;
}) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={2} fill="none" stroke="var(--rule)" />
      <rect x={x} y={y} width={Math.max(0, Math.min(1, frac)) * w} height={h} rx={2} fill={ghost ? 'none' : COLOR[kind]} stroke={COLOR[kind]} strokeDasharray={ghost ? '3 3' : undefined} style={{ transition: 'width .6s var(--ease-out)' }} />
      {label && <Label x={x - 8} y={y + h / 2 + 4} anchor="end" tone={2} size={10.5} mono={false}>{label}</Label>}
      {value && <Label x={x + w + 8} y={y + h / 2 + 4} tone={1} size={10.5}>{value}</Label>}
    </g>
  );
}

export const fmt = {
  int: (n: number) => Math.round(n).toLocaleString('en-US'),
  f1: (n: number) => n.toFixed(1),
  f2: (n: number) => n.toFixed(2),
  bytes: (b: number) => {
    const u = ['B', 'KB', 'MB', 'GB', 'TB'];
    let i = 0;
    while (b >= 1000 && i < u.length - 1) { b /= 1000; i++; }
    return `${b >= 100 ? b.toFixed(0) : b >= 10 ? b.toFixed(1) : b.toFixed(2)} ${u[i]}`;
  },
  gb: (b: number) => `${(b / 1e9).toFixed(b / 1e9 >= 100 ? 0 : 1)} GB`,
  si: (n: number) => {
    const u = ['', 'K', 'M', 'B', 'T'];
    let i = 0;
    while (Math.abs(n) >= 1000 && i < u.length - 1) { n /= 1000; i++; }
    return `${n >= 100 ? n.toFixed(0) : n >= 10 ? n.toFixed(1) : n.toFixed(2)}${u[i]}`;
  },
  ms: (s: number) => (s < 1 ? `${(s * 1000).toFixed(0)} ms` : `${s.toFixed(2)} s`),
};
