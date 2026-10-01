import type { ReactNode, RefObject } from 'react';
import { Icon } from './Icon';
import './viz.css';

export type Kind = 'compute' | 'memory' | 'token' | 'comm' | 'idle' | 'error' | 'text';

export const KIND_LABEL: Record<Kind, string> = {
  compute: 'compute',
  memory: 'memory traffic',
  token: 'tokens',
  comm: 'communication',
  idle: 'idle',
  error: 'error / waste',
  text: 'data',
};

interface StageProps {
  rootRef: RefObject<HTMLDivElement | null>;
  title: string;
  steps: string[];
  captions?: string[];
  step: number;
  go: (i: number) => void;
  playing: boolean;
  setPlaying: (p: boolean) => void;
  inScrolly: boolean;
  legend?: (Kind | [Kind, string])[];
  controls?: ReactNode;
  readout?: ReactNode;
  children: ReactNode;
  hideTransport?: boolean;
}

/** The frame every visual lives in: title, legend, canvas, live readout, transport (step ruler). */
export function Stage(p: StageProps) {
  const n = p.steps.length;
  const multi = n > 1 && !p.hideTransport;
  return (
    <figure className="stage" ref={p.rootRef}>
      <figcaption className="stage-head">
        <span className="stage-title">{p.title}</span>
        {p.legend && (
          <span className="stage-legend">
            {p.legend.map((l) => {
              const [k, label] = Array.isArray(l) ? l : [l, KIND_LABEL[l]];
              return (
                <span key={label} className={`lg k-${k}`}>
                  <i aria-hidden="true" />
                  {label}
                </span>
              );
            })}
          </span>
        )}
      </figcaption>
      <div className="stage-canvas">{p.children}</div>
      {p.readout && <div className="stage-readout">{p.readout}</div>}
      {p.controls && <div className="stage-controls">{p.controls}</div>}
      {multi && (
        <div className="transport">
          <div className="t-buttons">
            <button type="button" className="t-btn" onClick={() => { p.setPlaying(false); p.go(0); }} aria-label="Back to the start" disabled={p.step === 0}>
              <Icon name="first" />
            </button>
            <button type="button" className="t-btn" onClick={() => { p.setPlaying(false); p.go(p.step - 1); }} aria-label="Previous step" disabled={p.step === 0}>
              <Icon name="prev" />
            </button>
            <button
              type="button"
              className="t-btn t-play"
              onClick={() => { if (!p.playing && p.step >= n - 1) p.go(0); p.setPlaying(!p.playing); }}
              aria-label={p.playing ? 'Pause' : 'Play all steps'}
            >
              <Icon name={p.playing ? 'pause' : 'play'} />
            </button>
            <button type="button" className="t-btn" onClick={() => { p.setPlaying(false); p.go(p.step + 1); }} aria-label="Next step" disabled={p.step >= n - 1}>
              <Icon name="next" />
            </button>
          </div>
          <ol className="t-ruler" aria-label="Steps">
            {p.steps.map((s, i) => (
              <li key={i} className={i < p.step ? 'past' : i === p.step ? 'now' : ''}>
                <button type="button" onClick={() => { p.setPlaying(false); p.go(i); }} aria-label={`Step ${i + 1}: ${s}`} aria-current={i === p.step ? 'step' : undefined}>
                  <span className="seg" />
                </button>
              </li>
            ))}
          </ol>
          <span className="t-label" aria-live="polite">
            <span className="t-count">{p.step + 1}/{n}</span> {p.steps[p.step]}
          </span>
        </div>
      )}
      {!p.inScrolly && p.captions && p.captions[p.step] && <p className="stage-caption">{p.captions[p.step]}</p>}
    </figure>
  );
}

/* ───────────── small controls used inside stages ───────────── */

export function Slider(props: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  kind?: Kind;
  log?: boolean;
}) {
  const { label, value, min, max, step = 1, onChange, format = (v) => String(v), kind, log } = props;
  // optional log scale: slider works on log2(value)
  const toSlider = (v: number) => (log ? Math.log2(v) : v);
  const fromSlider = (v: number) => (log ? Math.round(Math.pow(2, v)) : v);
  const sMin = toSlider(min);
  const sMax = toSlider(max);
  const pct = ((toSlider(value) - sMin) / (sMax - sMin)) * 100;
  return (
    <label className={`slider ${kind ? `k-${kind}` : 'k-text'}`}>
      <span className="s-top">
        <span className="s-label">{label}</span>
        <span className="s-val">{format(value)}</span>
      </span>
      <input
        type="range"
        min={sMin}
        max={sMax}
        step={log ? 0.01 : step}
        value={toSlider(value)}
        style={{ ['--pct' as string]: `${pct}%` }}
        onChange={(e) => onChange(fromSlider(Number(e.target.value)))}
      />
    </label>
  );
}

export function Segmented<T extends string | number>(props: {
  label?: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="segmented" role="radiogroup" aria-label={props.label}>
      {props.label && <span className="s-label">{props.label}</span>}
      <div className="seg-row">
        {props.options.map((o) => (
          <button
            type="button"
            key={String(o.value)}
            role="radio"
            aria-checked={o.value === props.value}
            className={o.value === props.value ? 'on' : ''}
            onClick={() => props.onChange(o.value)}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export function Readout({ items }: { items: { label: string; value: string; kind?: Kind; note?: string }[] }) {
  return (
    <dl className="readout">
      {items.map((it) => (
        <div key={it.label} className={it.kind ? `k-${it.kind}` : 'k-text'}>
          <dt>{it.label}</dt>
          <dd>
            {it.value}
            {it.note && <small>{it.note}</small>}
          </dd>
        </div>
      ))}
    </dl>
  );
}
