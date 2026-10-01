// Drawing helpers shared by this module's visuals.
// A bracket is the "|←— label —→|" dimension mark used all over the timeline visuals,
// and msec() keeps time formatting identical across every stage in the module.

import type { ReactNode } from 'react';

/** Milliseconds below a second, seconds above, always 1 or 2 significant decimals. */
export const msec = (v: number) =>
  v < 1000 ? `${v < 10 ? v.toFixed(1) : Math.round(v).toString()} ms` : `${(v / 1000).toFixed(2)} s`;

/**
 * A measurement bracket. When the span is too narrow to hold its own label the label
 * moves outside the bracket (side 'left' or 'right') instead of overlapping it.
 */
export function Bracket({
  x1, x2, y, label, color = 'var(--text-3)', side = 'right', size = 10, weight,
}: {
  x1: number;
  x2: number;
  y: number;
  label: ReactNode;
  color?: string;
  side?: 'left' | 'right';
  size?: number;
  weight?: number;
}) {
  const l = Math.min(x1, x2);
  const r = Math.max(x1, x2);
  const narrow = r - l < size * (String(label).length * 0.62) + 14;
  const anchor = narrow ? (side === 'left' ? 'left' : 'right') : 'middle';
  const lx = narrow ? (side === 'left' ? l - 7 : r + 7) : (l + r) / 2;
  return (
    <g className="v-fade">
      <line x1={l} x2={r} y1={y} y2={y} stroke={color} strokeWidth={1} />
      <line x1={l} x2={l} y1={y - 4} y2={y + 4} stroke={color} strokeWidth={1} />
      <line x1={r} x2={r} y1={y - 4} y2={y + 4} stroke={color} strokeWidth={1} />
      <text
        x={lx}
        y={y + 3.5}
        textAnchor={anchor}
        className="v-label"
        style={{ fill: color, fontSize: size, fontWeight: weight }}
      >
        {label}
      </text>
    </g>
  );
}

/** Percentile markers: the value that p% of samples beat. */
export function PctMark({ x, y, h, label, color = 'var(--text-3)', strong }: { x: number; y: number; h: number; label: string; color?: string; strong?: boolean }) {
  return (
    <g className="v-fade">
      <line x1={x} x2={x} y1={y - 4} y2={y + h + 4} stroke={color} strokeWidth={strong ? 1.4 : 1} strokeDasharray={strong ? undefined : '3 3'} />
      <text x={x} y={y - 8} textAnchor="middle" className="v-label" style={{ fill: color, fontSize: 9.5, fontWeight: 600 }}>
        {label}
      </text>
    </g>
  );
}