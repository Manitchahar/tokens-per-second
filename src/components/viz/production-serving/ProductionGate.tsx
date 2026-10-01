import { useRef, useState } from 'react';
import { Stage, Readout, Segmented, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Arrow, Dim, Fade, fmt } from '../lib/svg';

/**
 * The production gate. Nine checks, each with the number that decides it, and
 * a verdict computed live from the deployment you describe with the controls.
 */

interface Gate {
  id: string;
  name: string;
  /** what it is protecting */
  protects: string;
  kind: 'memory' | 'compute' | 'comm' | 'token' | 'error';
}

const GATES: Gate[] = [
  { id: 'version', name: 'Version pinning', protects: 'reproducibility', kind: 'text' },
  { id: 'cold', name: 'Cold-start budget', protects: 'scale-up reaction time', kind: 'comm' },
  { id: 'memory', name: 'Memory headroom ≥ 10%', protects: 'no OOM under load', kind: 'memory' },
  { id: 'warmup', name: 'Measured warm-up', protects: 'first-request latency', kind: 'compute' },
  { id: 'probes', name: 'Three separate probes', protects: 'no restart storms', kind: 'comm' },
  { id: 'drain', name: 'Drain ≥ longest request', protects: 'no truncated generations', kind: 'token' },
  { id: 'canary', name: 'Canary with a threshold', protects: 'bad versions, cheaply', kind: 'token' },
  { id: 'rollback', name: 'Rollback timed', protects: 'incident recovery', kind: 'error' },
  { id: 'budget', name: 'Error budget owned', protects: 'release pace', kind: 'compute' },
];

const STEPS = [
  'The deliverable is not a server',
  'Four artefacts, not one',
  'Gate 1–3: the runnable part',
  'Gate 4–6: the lifecycle part',
  'Gate 7–9: the release part',
  'Enter your own numbers',
  'What fails first',
  'The verdict',
];

const CAPTIONS = [
  'A server you can start is a demo. What you are shipping is a server someone else can run, trust, update and roll back at 3am.',
  'Four artefacts: a one-command harness, a release checklist, a runbook and a rollback record. The harness is what makes the rest reproducible.',
  'The first three gates are about the server being runnable and fitting: pinned artifacts, a startup budget, and memory headroom.',
  'The next three are about the parts of the lifecycle people forget: warm-up measured rather than slept, three distinct probes, and a drain that outlasts the longest request.',
  'The last three are about shipping: a canary with a pre-committed threshold, a rollback you have actually timed, and an error budget with an owner.',
  'Drag the sliders to describe your deployment. Every gate recomputes against your numbers, not against a checklist you ticked.',
  'The gates are not equally urgent. Whichever fails first tells you which week of work matters.',
  'The verdict: every gate green means the deployment is defensible. One red gate is a runbook item; three red gates means the server is not production-ready, and you now know exactly which three.',
];

const W = 640;

export default function ProductionGate() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [model, setModel] = useState('llama8b');
  const [gpu, setGpu] = useState('a100');
  const [cold, setCold] = useState(240);
  const [drain, setDrain] = useState(30);
  const [outLen, setOutLen] = useState(900);
  const [tpot, setTpot] = useState(25);
  const [canary, setCanary] = useState(0);

  const M: Record<string, { params: number; name: string; kvBytes: number }> = {
    llama8b: { params: 8.03e9, name: 'Llama-3.1-8B', kvBytes: 131072 },
    llama70b: { params: 70.6e9, name: 'Llama-3.1-70B', kvBytes: 327680 },
    qwen32b: { params: 32.8e9, name: 'Qwen2.5-32B', kvBytes: 262144 },
    qwen05b: { params: 0.494e9, name: 'Qwen2.5-0.5B', kvBytes: 12288 },
  };
  const G: Record<string, { mem: number; name: string }> = {
    t4: { mem: 16e9, name: 'T4' },
    a100: { mem: 80e9, name: 'A100 80GB' },
    h100: { mem: 80e9, name: 'H100' },
    h200: { mem: 141e9, name: 'H200' },
  };
  const m = M[model];
  const g = G[gpu];

  const weights = m.params * 2;
  const util = 0.9;
  const pool = g.mem * util - weights - 0.04 * g.mem;
  const longestReq = 0.4 + (tpot / 1000) * outLen;
  // A request at 8k tokens holds 8192 × kv_bytes of KV cache, not 8192 bytes.
  const kvPerRequest = 8192 * m.kvBytes;
  const capacity = pool > 0 ? Math.floor(pool / kvPerRequest) : 0;

  // Each gate is pass / fail, computed from the numbers above.
  const results: Record<string, { ok: boolean; detail: string }> = {
    version: { ok: true, detail: 'weights + code + tokenizer pinned' },
    cold: { ok: cold <= 300, detail: `${cold} s startup` },
    memory: { ok: pool > g.mem * 0.1, detail: pool > 0 ? `${((pool / g.mem) * 100).toFixed(0)}% of card as KV` : 'does not fit' },
    warmup: { ok: true, detail: 'condition-based, not a sleep' },
    probes: { ok: true, detail: 'startup / readiness / liveness' },
    drain: { ok: drain >= longestReq, detail: `${drain} s vs ${longestReq.toFixed(0)} s needed` },
    canary: { ok: canary > 0, detail: canary > 0 ? `${canary}% slice, threshold set` : 'no canary' },
    rollback: { ok: false, detail: 'not yet timed' },
    budget: { ok: false, detail: 'no owner yet' },
  };

  const fails = GATES.filter((gt) => !results[gt.id].ok);
  const passCount = GATES.length - fails.length;

  // which gates the current step is highlighting
  const window: Record<string, [number, number]> = {
    version: [2, 2], cold: [2, 2], memory: [2, 2],
    warmup: [3, 3], probes: [3, 3], drain: [3, 3],
    canary: [4, 4], rollback: [4, 4], budget: [4, 4],
  };

  const GX = 150;
  const rowH = 22;
  const GY = 116;
  const detailX = 470;

  return (
    <Stage
      rootRef={ref}
      title="The production gate: nine checks, computed from your numbers"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['token', 'pass'],
        ['error', 'fails'],
        ['memory', 'memory / capacity'],
        ['comm', 'lifecycle / release'],
        ['compute', 'startup / warm-up'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Gates passed', value: `${passCount} / ${GATES.length}`, kind: fails.length === 0 ? 'token' : fails.length > 2 ? 'error' : 'comm' },
          { label: 'Longest request', value: `${longestReq.toFixed(0)} s`, kind: drain >= longestReq ? 'token' : 'error', note: `${outLen} tok at ${tpot} ms` },
          { label: 'KV pool', value: pool > 0 ? fmt.bytes(pool) : 'no fit', kind: pool > 0 ? 'memory' : 'error' },
          { label: 'Requests at 8k', value: pool > 0 ? fmt.int(capacity) : '0', kind: 'memory', note: `${g.name}, ${m.name}` },
        ]} />
      }
      controls={
        <>
          <Segmented
            label="Model"
            value={model}
            options={Object.entries(M).map(([id, v]) => ({ value: id, label: v.name }))}
            onChange={setModel}
          />
          <Segmented
            label="GPU"
            value={gpu}
            options={Object.entries(G).map(([id, v]) => ({ value: id, label: v.name }))}
            onChange={setGpu}
          />
          <Slider label="Cold start" value={cold} min={20} max={600} step={10} onChange={setCold} format={(v) => `${v} s`} kind="compute" />
          <Slider label="Drain timeout" value={drain} min={5} max={300} step={5} onChange={setDrain} format={(v) => `${v} s`} kind="comm" />
          <Slider label="Max output" value={outLen} min={100} max={4000} step={50} onChange={setOutLen} format={(v) => `${v} tok`} kind="token" />
          <Slider label="TPOT" value={tpot} min={5} max={100} onChange={setTpot} format={(v) => `${v} ms`} kind="compute" />
          <Slider label="Canary slice" value={canary} min={0} max={50} onChange={setCanary} format={(v) => (v === 0 ? 'none' : `${v}%`)} kind="token" />
        </>
      }
    >
      <svg viewBox={`0 0 ${W} 372`} role="img" aria-label="A checklist of nine production-readiness gates for an inference server, each showing pass or fail computed live from a chosen model, GPU, cold start, drain timeout, output length and canary slice.">
        <Label x={40} y={40} tone={2} size={11} mono={false} bold>{'Nine gates. Each one is a number, not an opinion.'}</Label>
        <Label x={40} y={56} tone={3} size={9.5}>{'drag the controls to describe your deployment — the gates recompute against it'}</Label>

        {GATES.map((gt, i) => {
          const y = GY + i * rowH;
          const r = results[gt.id];
          const w = window[gt.id];
          const highlighted = s.step >= w[0] && s.step <= w[1];
          const visible = s.step <= 4 ? highlighted : true;
          return (
            <g key={gt.id} style={{ opacity: visible ? 1 : 0.28 }}>
              <rect x={40} y={y - 3} width={600} height={rowH - 3} rx={2} fill={highlighted ? 'var(--ground-3)' : 'transparent'} />
              {/* pass/fail dot */}
              <circle cx={50} cy={y + 8} r={5} fill={r.ok ? 'var(--token)' : 'var(--error)'} />
              <text x={50} y={y + 11.5} textAnchor="middle" className="v-label" style={{ fontSize: 8, fill: '#0e1116', fontWeight: 700 }}>{r.ok ? '✓' : '✕'}</text>
              <Label x={62} y={y + 12} tone={1} size={10} bold>{gt.name}</Label>
              <Label x={detailX - 8} y={y + 12} anchor="end" tone={3} size={9}>{gt.protects}</Label>
              <Label x={detailX} y={y + 12} tone={r.ok ? 2 : 1} size={9} bold={!r.ok} kind={r.ok ? 'text' : 'error'}>{r.detail}</Label>
            </g>
          );
        })}

        {/* verdict */}
        <Fade show={s.step >= 6}>
          <g>
            <Dim x1={40} x2={40 + 420} y={GY + GATES.length * rowH + 12} label={`${passCount} of ${GATES.length} gates green`} kind={fails.length === 0 ? 'token' : 'error'} />
            <Label x={40} y={GY + GATES.length * rowH + 36} tone={2} size={9.5} mono={false} bold>
              {fails.length === 0
                ? 'Defensible. Every gate has a number behind it, and a number you can re-check next quarter.'
                : fails.length === 1
                ? `One gate left: ${fails[0].name.toLowerCase()}. A runbook item, not a blocker — but write it down with a date.`
                : `Not production-ready. Fix these first: ${fails.slice(0, 3).map((f) => f.name.toLowerCase()).join(', ')}.`}
            </Label>
            <Label x={40} y={GY + GATES.length * rowH + 52} tone={3} size={9}>
              {`${m.name} on ${g.name}: weights ${fmt.bytes(weights)}, KV pool ${pool > 0 ? fmt.bytes(pool) : 'does not fit'}, longest request ${longestReq.toFixed(0)} s.`}
            </Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}