import { useRef } from 'react';
import { Stage, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Arrow, Fade, Label } from '../lib/svg';
import { Box, Bracket, ColHead, COL, COL_W, MODEL_X, MODEL_W, VIEW_H, VIEW_W, row } from './parts';

/**
 * The shape of a serving engine, revealed one box at a time.
 * Used by "Reading a real engine's codebase": the learner's first job is to find
 * these six boxes in a real repository and know which one owns the GPU.
 */

const STEPS = [
  'One request, no queue',
  'The API server owns the port',
  'The engine core owns the GPU',
  'The model runner owns the weights',
  'The KV manager owns memory',
  'The scheduler owns everything else',
  'Read it in this order',
];
const CAPTIONS = [
  'What a tutorial shows: one string in, one string out, generate() in a loop. Everything in this module fills the gap between that and what a server does.',
  'The HTTP world: validate the request, apply the chat template, tokenize, hold the stream open. It never touches a GPU.',
  'The engine core is everything that must run between GPU steps. If it is in Python, you are paying for it in inter-token latency.',
  'The model runner holds the weights and the KV tensors and runs one forward pass. In vLLM this is the thinnest and least interesting file.',
  'The KV cache manager is the memory system: blocks, block tables, prefix sharing, eviction. This is where PagedAttention lives.',
  'The scheduler decides who is in the batch every iteration. It is short, it is pure Python, and it is the component that decides who waits.',
  'Start at the scheduler and read outwards. Knowing which box a question belongs to is most of reading a large codebase.',
];

const API = [
  { title: 'HTTP handler', sub: 'validate, template' },
  { title: 'Tokenizer', sub: 'text → ids' },
  { title: 'Stream writer', sub: 'ids → SSE' },
];
const CORE = [
  { title: 'Scheduler', sub: 'who is in this step' },
  { title: 'KV manager', sub: 'blocks & tables' },
  { title: 'Model runner', sub: 'weights & forward' },
  { title: 'Sampler', sub: 'ids → token' },
];
const MODEL = [
  { title: 'Weights', sub: '16 GB, read only' },
  { title: 'KV cache', sub: 'grows per token' },
];

export default function EngineMap() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const k = s.step;

  // Which boxes are lit at each step.
  const apiOn = k >= 1;
  const coreOn = k >= 2;
  const runnerOn = k >= 3;
  const kvOn = k >= 4;
  const schedOn = k >= 5;
  const order = k >= 6;

  const apiY = [row(0), row(1), row(2)];
  const coreY = [row(0), row(1), row(2), row(3)];
  const modelY = [row(0), row(1)];
  const coreTop = coreY[0] - 6;
  const coreBot = coreY[3] + 40 + 6;
  const modelTop = modelY[0] - 6;
  const modelBot = modelY[1] + 40 + 6;

  return (
    <Stage
      rootRef={ref}
      title="The six boxes of a serving engine"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', 'comm', 'token', 'text']}
      {...s}
      readout={
        <Readout items={[
          { label: 'HTTP work', value: 'Python', kind: 'comm', note: 'off the GPU path' },
          { label: 'Engine core', value: k >= 2 ? 'Python + GPU' : '—', kind: 'compute', note: k >= 2 ? 'runs every iteration' : 'not yet revealed' },
          { label: 'GPU-resident', value: k >= 3 ? 'weights + KV' : '—', kind: 'memory', note: 'the only bytes that matter' },
          { label: 'Lines you must read', value: k >= 6 ? '~2,000' : '—', note: 'not 400,000' },
        ]} />
      }
    >
      <svg viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} role="img" aria-label="A block diagram of an LLM serving engine: an API server column on the left, an engine core column in the middle holding a scheduler, KV manager, model runner and sampler, and a model column on the right holding weights and the KV cache. Boxes light up one group at a time as the steps advance.">
        {/* column headers */}
        <ColHead x={COL.api} w={COL_W} title="API server" note="no GPU, all Python" />
        <ColHead x={COL.core} w={COL_W} title="Engine core" note="owns the GPU" kind={coreOn ? 'compute' : undefined} />
        <ColHead x={MODEL_X} w={MODEL_W} title="Model state" note="lives in HBM" kind={k >= 3 ? 'memory' : undefined} />

        {/* the GPU bracket, only once the core is revealed */}
        <Fade show={coreOn}>
          <rect x={COL.core - 10} y={coreTop - 20} width={COL_W + 20} height={coreBot - coreTop + 22} rx={6} fill="none" stroke="var(--rule)" strokeDasharray="4 4" />
        </Fade>

        {/* API column */}
        {API.map((b, i) => (
          <Box key={b.title} x={COL.api} y={apiY[i]} w={COL_W} title={b.title} sub={b.sub} kind="comm" opacity={apiOn ? 1 : 0.16} dim={!apiOn} />
        ))}

        {/* API → core: the handoff that starts the clock */}
        <Fade show={apiOn && coreOn}>
          <Arrow x1={COL.api + COL_W} y1={apiY[2] + 17} x2={COL.core - 14} y2={coreY[3] + 17} kind="token" curve={-14} />
          <Label x={(COL.api + COL_W + COL.core) / 2} y={apiY[2] + 6} anchor="middle" tone={3} size={9}>token ids</Label>
        </Fade>
        <Fade show={k === 0}>
          <Label x={COL.api + COL_W / 2} y={VIEW_H - 96} anchor="middle" tone={3} size={10.5} mono={false}>the tutorial’s whole world</Label>
          <Label x={COL.api + COL_W / 2} y={VIEW_H - 80} anchor="middle" tone={3} size={10.5} mono={false}>one prompt, one loop, no queue</Label>
        </Fade>

        {/* Core column */}
        <Box x={COL.core} y={coreY[0]} w={COL_W} title="Scheduler" sub="who is in this step" kind="token" opacity={coreOn ? (schedOn ? 1 : 0.55) : 0.16} dim={!coreOn} />
        <Box x={COL.core} y={coreY[1]} w={COL_W} title="KV manager" sub="blocks & tables" kind="memory" opacity={coreOn ? (kvOn ? 1 : 0.55) : 0.16} dim={!coreOn} />
        <Box x={COL.core} y={coreY[2]} w={COL_W} title="Model runner" sub="weights & forward" kind="compute" opacity={coreOn ? (runnerOn ? 1 : 0.55) : 0.16} dim={!coreOn} />
        <Box x={COL.core} y={coreY[3]} w={COL_W} title="Sampler" sub="ids → token" kind="token" opacity={coreOn ? (schedOn ? 1 : 0.55) : 0.16} dim={!coreOn} />

        {/* Core → model */}
        <Fade show={runnerOn}>
          <Arrow x1={COL.core + COL_W} y1={coreY[2] + 17} x2={MODEL_X - 14} y2={modelY[0] + 17} kind="memory" />
          <Arrow x1={MODEL_X + MODEL_W} y1={modelY[1] + 17} x2={COL.core + COL_W + 14} y2={coreY[1] + 17} kind="memory" dashed />
        </Fade>

        {/* Model column */}
        <Box x={MODEL_X} y={modelY[0]} w={MODEL_W} title="Weights" sub="16 GB, read only" kind="memory" opacity={runnerOn ? 1 : 0.16} dim={!runnerOn} />
        <Box x={MODEL_X} y={modelY[1]} w={MODEL_W} title="KV cache" sub="grows per token" kind="memory" opacity={kvOn ? 1 : 0.16} dim={!kvOn} />

        {/* Who owns the GPU */}
        <Fade show={coreOn}>
          <Bracket x={COL.core - 16} y1={coreTop} y2={coreBot} label="runs between every GPU step" kind="compute" />
        </Fade>
        <Fade show={runnerOn}>
          <Bracket x={MODEL_X + MODEL_W + 8} y1={modelTop} y2={modelBot} label="bytes, read every step" kind="memory" />
        </Fade>

        {/* Reading order */}
        <Fade show={order}>
          <Label x={COL.core + COL_W / 2} y={VIEW_H - 30} anchor="middle" size={10.5} kind="token" bold>read in this order →</Label>
          {['scheduler', 'KV manager', 'runner', 'API'].map((t, i) => (
            <g key={t}>
              <circle cx={74 + i * 122} cy={VIEW_H - 30} r={9} fill="none" stroke="var(--token)" strokeWidth={1.2} />
              <text x={74 + i * 122} y={VIEW_H - 26.5} textAnchor="middle" className="v-label" style={{ fontSize: 10, fill: 'var(--token)' }}>{i + 1}</text>
              <Label x={74 + i * 122} y={VIEW_H - 12} anchor="middle" tone={2} size={9.5} mono={false}>{t}</Label>
            </g>
          ))}
        </Fade>
      </svg>
    </Stage>
  );
}