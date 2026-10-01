import type { ReactNode } from 'react';
import { Label, Fade } from '../lib/svg';
import type { Kind } from '../lib/Stage';

/**
 * Shared geometry for the module's engine block diagrams.
 *
 * A three-column engine picture: the API world on the left, the engine core in the
 * middle (the only thing that owns the GPU), and the model on the right. Every step
 * in this module reveals one box in it, so the box metrics live here rather than
 * being re-declared in five components.
 */

export const VIEW_W = 640;
export const VIEW_H = 372;

/** Lane x-offsets. Three columns with generous gutters. */
export const COL = { api: 24, core: 214, model: 470 } as const;
export const COL_W = 150;
export const MODEL_X = 470;
export const MODEL_W = 146;

export const BOX_H = 34;
export const BOX_GAP = 8;

/** Rows for the three stacks, in draw order. y is the top of each box. */
export function row(i: number, y = 84, h = BOX_H, gap = BOX_GAP): number {
  return y + i * (h + gap);
}

/** A labelled block. `kind` colours it by what it *is*, never decoratively. */
export function Box({
  x, y, w, h = BOX_H, title, sub, kind = 'text', ghost, dim, opacity = 1,
}: {
  x: number; y: number; w?: number; h?: number; title: string; sub?: ReactNode;
  kind?: Kind; ghost?: boolean; dim?: boolean; opacity?: number;
}) {
  const stroke = ghost ? 'var(--rule-2)' : kind === 'text' ? 'var(--rule-2)' : `var(--${kind})`;
  const fill = ghost ? 'none' : dim ? `var(--${kind}-dim)` : kind === 'text' ? 'var(--ground-3)' : `var(--${kind})`;
  return (
    <g className="v-fade" style={{ opacity }}>
      <rect x={x} y={y} width={w} height={h} rx={4} fill={fill} stroke={stroke} strokeWidth={ghost ? 1 : 1.3} strokeDasharray={ghost ? '3 3' : undefined} />
      <text x={x + w! / 2} y={sub ? y + h / 2 - 1 : y + h / 2 + 4} textAnchor="middle" className="v-label"
        style={{ fontSize: 11, fontWeight: 600, fill: !ghost && !dim && kind !== 'text' ? '#0e1116' : 'var(--text)' }}>
        {title}
      </text>
      {sub && (
        <text x={x + w! / 2} y={y + h / 2 + 11} textAnchor="middle" className="v-label"
          style={{ fontSize: 9, fill: 'var(--text-3)' }}>
          {sub}
        </text>
      )}
    </g>
  );
}

/** A column header with a caption under it. */
export function ColHead({ x, w, title, note, kind }: { x: number; w: number; title: string; note: string; kind?: Kind }) {
  return (
    <g>
      <Label x={x + w / 2} y={62} anchor="middle" tone={kind ? 1 : 2} size={11.5} bold kind={kind}>{title}</Label>
      <Label x={x + w / 2} y={75} anchor="middle" tone={3} size={9.5}>{note}</Label>
    </g>
  );
}

/** Vertical bracket marking a span of the diagram as "owns the GPU" / "no GPU". */
export function Bracket({ x, y1, y2, label, kind }: { x: number; y1: number; y2: number; label: string; kind: Kind }) {
  const c = `var(--${kind})`;
  return (
    <g className="v-fade">
      <line x1={x} x2={x} y1={y1} y2={y2} stroke={c} strokeWidth={1.2} />
      <line x1={x - 4} x2={x + 4} y1={y1} y2={y1} stroke={c} strokeWidth={1.2} />
      <line x1={x - 4} x2={x + 4} y1={y2} y2={y2} stroke={c} strokeWidth={1.2} />
      <Label x={x + (kind === 'comm' ? 8 : -8)} y={(y1 + y2) / 2 + 3.5} anchor={kind === 'comm' ? 'start' : 'end'} size={9.5} kind={kind}>{label}</Label>
    </g>
  );
}

export { Fade };