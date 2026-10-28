import { useRef, useState } from 'react';
import { Stage, Segmented, Readout, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Chip, Arrow, Fade, Bar, fmt } from '../lib/svg';

// ── the arithmetic, local to this lesson ──────────────────────────────────────
// Shapes are published configuration values; hardware figures are approximate
// public spec-sheet numbers and vary by SKU and clock.

interface EdgeModel {
  id: string;
  name: string;
  params: number;
  layers: number;
  kvHeads: number;
  headDim: number;
}

const MODELS: EdgeModel[] = [
  { id: '1b', name: '1B', params: 1.54e9, layers: 28, kvHeads: 2, headDim: 128 },
  { id: '3b', name: '3B', params: 3.21e9, layers: 28, kvHeads: 8, headDim: 128 },
  { id: '8b', name: '8B', params: 8.03e9, layers: 32, kvHeads: 8, headDim: 128 },
];

/** 2 (K and V) × layers × KV heads × head dim × 2 bytes per number. */
const kvBytesPerToken = (m: EdgeModel) => 2 * m.layers * m.kvHeads * m.headDim * 2;

/** A 4-bit file is not 4.0 bits per weight: block scales and minimums push it to
 *  about 4.5, which is 0.5625 bytes per parameter. */
const B4 = 4.5 / 8;
const B16 = 2;

const RAM = 8e9; // an 8 GB phone, approximate
const OS = 3.0e9; // the operating system's resident share, approximate
const APP = 1.5e9; // this app plus its runtime, approximate
const ACHIEVED = 0.6; // fraction of peak bandwidth a real decode sustains
const THROTTLE = 0.55; // sustained clock as a fraction of the burst clock (an assumption)

const STEPS = [
  'A phone is a whole computer with a budget',
  'The stack: a file, a runtime, a wrapper',
  'Decode is a pipe from memory to math',
  'Swap 16-bit weights for 4-bit',
  'Sustained load heats the chip',
  'Hard requests go to the cloud',
];
const CAPTIONS = [
  'A phone has 8 GB of memory in total and the model does not get all of it. The operating system and every other app are already resident, so what is left for weights and KV cache is a fraction of the sticker number.',
  'Three layers, not one. A model file in a runtime-specific format, a runtime that turns it into instructions for this chip, and an app wrapper that handles the input and output. Conversion and numeric validation sit between training and shipping.',
  'Decode reads every weight once per token, so the step time is bytes divided by bandwidth. The compute units are almost idle; the pipe is the constraint. This is the bandwidth-bound rule from Module 02, unchanged, on a much smaller machine.',
  'Four bits instead of sixteen cuts the bytes per weight by roughly 3.5×, and tokens per second rise by the same factor. Compression on the edge is not an optimisation you get to skip; it is what makes the model fit and run at a usable speed.',
  'A phone cannot dissipate sustained power the way a server can. Clocks drop after tens of seconds, achieved bandwidth drops with them, and tokens per second fall by roughly the same fraction. Benchmarks must run long enough to see the flat part.',
  'The phone keeps what it can answer and escalates the rest. Network quality, battery, temperature, confidence and request size are all routing signals, and the arrow to the cloud is a communication cost you pay on purpose.',
];

export default function EdgeInference() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;

  const [modelId, setModelId] = useState<string>('3b');
  const [ctx, setCtx] = useState(2048);
  const [bwGbs, setBwGbs] = useState(60);

  const model = MODELS.find((m) => m.id === modelId) ?? MODELS[1];
  const fourBit = st >= 3; // the swap happens at step 4
  const throttled = st >= 4;

  const w16 = model.params * B16;
  const weights = model.params * (fourBit ? B4 : B16);
  const kv = ctx * kvBytesPerToken(model);
  const used = OS + APP + weights + kv;
  const free = RAM - used;
  const over = free < 0;

  const bw = bwGbs * 1e9;
  const effBw = bw * ACHIEVED * (throttled ? THROTTLE : 1);
  const tps = effBw / weights;
  const msPerToken = 1000 / tps;

  // memory bar: scale to the requirement, so an over-budget bar visibly spills
  const total = Math.max(RAM, used);
  const BX = 24;
  const BW = 144;
  const px = (b: number) => (b / total) * BW;
  const osW = px(OS);
  const appW = px(APP);
  const wW = px(weights);
  const kvW = px(kv);
  const ramX = BX + px(RAM);

  const kvKb = kvBytesPerToken(model) / 1024;

  return (
    <Stage
      rootRef={ref}
      title="On-device inference: budget, pipe, heat, routing"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['memory', 'weights and KV cache'],
        ['compute', 'the compute units'],
        ['token', 'tokens per second'],
        ['comm', 'the network'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Weights', value: fmt.bytes(weights), kind: 'memory', note: fourBit ? '4-bit, 4.5 bits per weight' : '16-bit' },
          { label: 'KV cache', value: fmt.bytes(kv), kind: 'memory', note: `${fmt.int(ctx)} tokens at ${kvKb.toFixed(0)} KB each` },
          { label: 'Decode', value: `${tps.toFixed(0)} tok/s`, kind: 'token', note: `${msPerToken.toFixed(0)} ms per token${throttled ? ', throttled' : ''}` },
          { label: over ? 'Over budget' : 'Spare RAM', value: over ? fmt.bytes(-free) : fmt.bytes(free), kind: over ? 'error' : 'text', note: `${(RAM / 1e9).toFixed(0)} GB in total` },
        ]} />
      }
      controls={
        <>
          <Segmented label="Model" value={modelId} options={MODELS.map((m) => ({ value: m.id, label: m.name }))} onChange={setModelId} />
          <Slider label="Context" kind="memory" min={256} max={8192} step={256} value={ctx} onChange={setCtx} format={(v) => `${fmt.int(v)} tokens`} />
          <Slider label="Memory bandwidth" kind="compute" min={40} max={400} step={10} value={bwGbs} onChange={setBwGbs} format={(v) => `${v} GB/s`} />
        </>
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="A phone outline with a memory budget bar split between operating system, app, model weights and KV cache; a pipe from memory to compute showing tokens per second; a comparison of 16-bit and 4-bit weights; a thermal curve showing sustained throttling; and a hard request routed over the network to a cloud GPU.">
        {/* ── the device ── */}
        <rect x={8} y={26} width={170} height={346} rx={20} fill="var(--ground-2)" stroke="var(--rule-2)" strokeWidth={1.2} />
        <rect x={72} y={32} width={42} height={4} rx={2} fill="var(--rule-2)" />
        <rect x={16} y={42} width={154} height={316} rx={8} fill="none" stroke="var(--rule)" />

        {/* ── memory budget ── */}
        <Label x={24} y={58} tone={2} size={9.5} bold>{`${(RAM / 1e9).toFixed(0)} GB of RAM, in total`}</Label>
        <rect x={BX} y={66} width={BW} height={24} rx={2} fill="none" stroke="var(--rule)" />
        {/* free space, hatched */}
        {!over && (
          <rect x={BX + px(used)} y={66} width={Math.max(0, px(RAM) - px(used))} height={24} fill="url(#tps-hatch)" stroke="var(--idle)" strokeWidth={0.8} />
        )}
        <rect x={BX} y={66} width={osW} height={24} fill="var(--ground-3)" stroke="var(--rule-2)" strokeWidth={0.8} />
        <rect x={BX + osW} y={66} width={appW} height={24} fill="var(--text-dim)" stroke="var(--rule-2)" strokeWidth={0.8} />
        <rect x={BX + osW + appW} y={66} width={wW} height={24} fill="var(--memory)" stroke="var(--memory)" strokeWidth={0.8} style={{ transition: 'x .5s var(--ease-out), width .5s var(--ease-out)' }} />
        <rect x={BX + osW + appW + wW} y={66} width={kvW} height={24} fill="var(--memory-dim)" stroke="var(--memory)" strokeWidth={0.8} style={{ transition: 'x .5s var(--ease-out), width .5s var(--ease-out)' }} />
        {over && (
          <>
            <rect x={ramX} y={66} width={BX + BW - ramX} height={24} fill="var(--error)" opacity={0.32} />
            <line x1={ramX} x2={ramX} y1={62} y2={94} stroke="var(--error)" strokeWidth={1.4} />
            <Label x={ramX} y={106} anchor="middle" kind="error" size={7.5}>limit</Label>
          </>
        )}
        <Label x={24} y={over ? 120 : 106} tone={3} size={7.5}>OS 3.0</Label>
        <Label x={78} y={over ? 120 : 106} tone={3} size={7.5}>app 1.5</Label>
        <Label x={24} y={over ? 131 : 117} kind="memory" size={7.5}>{`weights ${(weights / 1e9).toFixed(1)}`}</Label>
        <Label x={84} y={over ? 131 : 117} kind="memory" size={7.5}>{`KV ${(kv / 1e9).toFixed(2)}`}</Label>
        {!over && <Label x={130} y={117} tone={3} size={7.5}>{`free ${(free / 1e9).toFixed(1)}`}</Label>}
        {over && (
          <Label x={24} y={144} kind="error" size={8}>{`over budget by ${((used - RAM) / 1e9).toFixed(1)} GB`}</Label>
        )}

        {/* ── the stack ── */}
        <Fade show={st >= 1}>
          <g>
            <Label x={24} y={over ? 162 : 146} tone={2} size={9}>the device stack</Label>
            <Chip x={24} y={over ? 168 : 152} w={146} h={26} text="model file .gguf" kind="text" size={8.5} />
            <Arrow x1={97} y1={over ? 196 : 180} x2={97} y2={over ? 204 : 190} muted />
            <Chip x={24} y={over ? 206 : 192} w={146} h={26} text="runtime: llama.cpp" kind="compute" size={8.5} />
            <Arrow x1={97} y1={over ? 234 : 220} x2={97} y2={over ? 242 : 230} muted />
            <Chip x={24} y={over ? 244 : 232} w={146} h={26} text="app wrapper" kind="text" size={8.5} />
          </g>
        </Fade>

        {/* ── thermal sparkline ── */}
        <Fade show={st >= 4}>
          <g>
            <Label x={24} y={278} tone={2} size={9}>sustained load</Label>
            <line x1={28} x2={162} y1={292} y2={292} stroke="var(--rule-2)" strokeDasharray="3 3" />
            <Label x={30} y={288} tone={3} size={7}>burst</Label>
            <path d="M28,292 L58,292 L96,316 L162,316" fill="none" stroke="var(--compute)" strokeWidth={1.8} />
            <Label x={122} y={310} kind="error" size={8}>−45%</Label>
            <Label x={28} y={344} tone={3} size={7}>0 s</Label>
            <Label x={162} y={344} anchor="end" tone={3} size={7}>120 s</Label>
          </g>
        </Fade>

        {/* ── A: the bandwidth pipe ── */}
        <Fade show={st >= 2}>
          <g>
            <Label x={200} y={44} tone={2} size={11} bold>Decode: memory to math, once per token</Label>
            <rect x={200} y={54} width={90} height={44} rx={3} fill="var(--memory-dim)" stroke="var(--memory)" />
            <Label x={245} y={72} anchor="middle" tone={2} size={8.5}>weights</Label>
            <Label x={245} y={88} anchor="middle" tone={1} size={10} bold>{fmt.bytes(weights)}</Label>
            <rect x={296} y={64} width={132} height={24} rx={2} fill="var(--memory-dim)" stroke="var(--memory)" />
            <Arrow x1={302} y1={76} x2={422} y2={76} kind="memory" width={1 + bwGbs / 60} />
            <Label x={362} y={58} anchor="middle" tone={3} size={8}>{`${bwGbs} GB/s peak`}</Label>
            <Label x={362} y={100} anchor="middle" tone={3} size={8}>{`${(effBw / 1e9).toFixed(0)} GB/s achieved`}</Label>
            <rect x={434} y={54} width={76} height={44} rx={3} fill="var(--compute)" stroke="var(--compute)" />
            <Label x={472} y={71} anchor="middle" size={8} style={{ fill: '#0e1116' }}>CPU /</Label>
            <Label x={472} y={85} anchor="middle" size={8} style={{ fill: '#0e1116' }}>GPU / NPU</Label>
            <Label x={518} y={78} kind="token" size={20} bold>{tps.toFixed(0)}</Label>
            <Label x={518} y={92} tone={2} size={8.5}>tok/s</Label>
            <Label x={518} y={104} tone={3} size={7.5}>{`${msPerToken.toFixed(0)} ms/token`}</Label>
          </g>
        </Fade>

        {/* ── B: 16-bit against 4-bit ── */}
        <Fade show={st >= 3}>
          <g>
            <Label x={200} y={130} tone={2} size={11} bold>Bits are bytes</Label>
            <Label x={200} y={152} tone={3} size={8.5}>16-bit</Label>
            <Bar x={246} y={144} w={150} h={13} frac={1} kind="memory" ghost={fourBit} />
            <Label x={404} y={154} tone={fourBit ? 3 : 1} size={8.5}>{fmt.bytes(w16)}</Label>
            <Label x={462} y={154} kind="token" size={8.5}>{`${(effBw / w16).toFixed(0)} tok/s`}</Label>
            <Label x={200} y={174} tone={3} size={8.5}>4-bit</Label>
            <Bar x={246} y={166} w={150} h={13} frac={weights / w16} kind="memory" ghost={!fourBit} />
            <Label x={404} y={176} tone={fourBit ? 1 : 3} size={8.5}>{fmt.bytes(weights)}</Label>
            <Label x={462} y={176} kind="token" size={8.5}>{`${tps.toFixed(0)} tok/s`}</Label>
            <Label x={200} y={192} tone={3} size={7.5}>4.5 bits per weight once the block scales are counted</Label>
          </g>
        </Fade>

        {/* ── C: edge and cloud ── */}
        <Fade show={st >= 5}>
          <g>
            <Label x={200} y={214} tone={2} size={11} bold>Hard request? Escalate to the cloud</Label>
            <rect x={200} y={226} width={104} height={46} rx={3} fill="var(--ground-3)" stroke="var(--rule-2)" />
            <Label x={252} y={244} anchor="middle" tone={2} size={8.5}>on-device</Label>
            <Label x={252} y={260} anchor="middle" tone={3} size={8}>{`${model.name} · ${fourBit ? '4-bit' : '16-bit'}`}</Label>
            <Arrow x1={310} y1={249} x2={392} y2={249} kind="comm" width={2} />
            <Label x={351} y={241} anchor="middle" kind="comm" size={7.5}>escalate</Label>
            <rect x={396} y={226} width={104} height={46} rx={3} fill="var(--compute-dim)" stroke="var(--compute)" />
            <Label x={448} y={244} anchor="middle" tone={2} size={8.5}>cloud GPU</Label>
            <Label x={448} y={260} anchor="middle" tone={3} size={8}>70B · BF16</Label>
            <Label x={200} y={292} tone={3} size={8}>route on:</Label>
            <Chip x={200} y={300} w={62} h={18} text="network" kind="comm" size={8} />
            <Chip x={268} y={300} w={62} h={18} text="battery" kind="text" size={8} />
            <Chip x={336} y={300} w={66} h={18} text="thermal" kind="error" size={8} />
            <Chip x={200} y={324} w={76} h={18} text="confidence" kind="text" size={8} />
            <Chip x={284} y={324} w={84} h={18} text="request size" kind="text" size={8} />
            <Label x={200} y={358} tone={3} size={8}>the phone keeps what it can answer locally</Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}
