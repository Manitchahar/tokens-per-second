import { useEffect, useRef, useState } from 'react';
import { Stage, Slider, Readout, type Kind } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Arrow, Chip, Fade, Label } from '../lib/svg';

const STEPS = [
  'Meaning becomes distance',
  'A near neighbour hits',
  'A miss pays and stores',
  'Widen the circle',
  'Route the easy ones small',
  'Route the hard ones large',
];
const CAPTIONS = [
  'Every query becomes a vector, and stored question–answer pairs live in the same space. Each stored pair gets a circle whose radius is set by the similarity threshold.',
  'A new question lands inside the circle of a stored question. Its cosine similarity clears the threshold, so the stored answer is returned: no prefill, no decode, no GPU work.',
  'A question outside every circle misses. It goes to the LLM for a full call, and the pair is stored with its vector so a repeat can hit later.',
  'Lower the threshold and the circles grow. More repeats hit — and one query lands in the wrong circle, returning a confident answer to a different question. That is a false hit, and it costs W.',
  'The same embedding can route as well as cache. An easy query scores below the complexity line and goes to a small model, which is far cheaper per call.',
  'A hard query crosses the line and goes to the large model, or to the same model with reasoning enabled. The router trades model cost against answer quality, one threshold at a time.',
];

// Stored question–answer pairs, drawn as dots in a two-dimensional sketch of embedding space.
const ENTRIES = [
  { x: 86, y: 234, label: 'flight time' },
  { x: 205, y: 108, label: 'reset password' },
  { x: 326, y: 222, label: 'refund policy' },
];
// A pair stored during step 2, after the miss.
const STORED = { x: 300, y: 194 };
// Background traffic: queries that were embedded but are not stored answers.
const TRAFFIC = [
  { x: 50, y: 120 },
  { x: 150, y: 285 },
  { x: 265, y: 62 },
  { x: 370, y: 110 },
  { x: 390, y: 270 },
  { x: 150, y: 175 },
  { x: 40, y: 265 },
  { x: 330, y: 70 },
];

// Positions of the three queries used by the story.
const Q1 = { x: 102, y: 226, sim: 0.96, label: 'how long is the flight?' };
const Q2 = { x: 300, y: 194, sim: 0.71, label: 'can I get my money back?' };
const Q3 = { x: 240, y: 150, sim: 0.74, label: 'cancel my plan' };
const Q4 = { x: 230, y: 270, sim: 0.0, label: 'hard · multi-step' };
// Distance from Q3 to the "reset password" entry at (205, 108).
const Q3_DIST = Math.hypot(Q3.x - 205, Q3.y - 108);

const NOTES: Record<number, string[]> = {
  0: ['Stored pairs sit in the same', 'embedding space as new queries.', 'Each circle is the threshold.'],
  1: ['The nearest stored question', 'clears the threshold, so its', 'answer is returned unchanged.'],
  2: ['No stored question is close', 'enough. The LLM runs, then the', 'pair is stored for next time.'],
  3: ['A lower threshold lets one query', 'into the wrong circle: a', 'confident answer, wrong question.'],
  4: ['Short factual queries take the', 'cheap path. The small model', 'answers at a fraction of the cost.'],
  5: ['Multi-step queries need the', 'large model, or the same model', 'with reasoning switched on.'],
};

export default function SemanticCache() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [thr, setThr] = useState(0.9);

  // Step 3 is the "widen the threshold" step; every other step starts narrow.
  useEffect(() => setThr(s.step === 3 ? 0.68 : 0.9), [s.step]);

  const radius = (1 - thr) * 190;
  const falseHit = radius >= Q3_DIST;
  const cacheScene = s.step <= 3;

  const decision: { text: string; kind: Kind } =
    s.step === 0
      ? { text: '3 · compare with cosine', kind: 'text' }
      : s.step === 1
        ? thr <= Q1.sim
          ? { text: `HIT · ${Q1.sim.toFixed(2)} ≥ ${thr.toFixed(2)}`, kind: 'token' }
          : { text: `MISS · ${Q1.sim.toFixed(2)} < ${thr.toFixed(2)}`, kind: 'compute' }
        : s.step === 2
          ? thr <= Q2.sim
            ? { text: `HIT · ${Q2.sim.toFixed(2)} ≥ ${thr.toFixed(2)}`, kind: 'token' }
            : { text: `MISS · ${Q2.sim.toFixed(2)} < ${thr.toFixed(2)}`, kind: 'compute' }
          : falseHit
            ? { text: '3 hits · 1 false hit', kind: 'error' }
            : { text: '2 hits · 0 false hits', kind: 'token' };

  const readout: { label: string; value: string; kind?: Kind; note?: string }[] = cacheScene
    ? [
        { label: 'Threshold', value: thr.toFixed(2), kind: 'comm', note: 'minimum cosine similarity' },
        { label: 'Stored pairs', value: s.step >= 2 ? '4' : '3', kind: 'memory', note: 'tenant-filtered index' },
        {
          label: 'Answer source',
          value: s.step === 0 ? '—' : s.step === 2 ? 'LLM call' : s.step === 3 ? (falseHit ? 'cache · wrong' : 'cache') : 'cache',
          kind: s.step === 3 && falseHit ? 'error' : s.step === 2 ? 'compute' : 'token',
        },
        {
          label: 'GPU work',
          value: s.step === 2 ? '1 full call' : s.step === 1 ? '0 calls' : s.step === 3 ? '2 calls avoided' : '—',
          kind: s.step === 2 ? 'compute' : 'idle',
        },
      ]
    : [
        { label: 'Complexity score', value: s.step === 4 ? '0.22' : '0.78', kind: 'comm', note: 'one number per query' },
        { label: 'Route', value: s.step === 4 ? 'small · 3B' : 'large · 70B', kind: s.step === 4 ? 'token' : 'compute' },
        { label: 'Relative cost', value: s.step === 4 ? '1×' : '≈23×', kind: 'compute', note: 'parameters, illustrative' },
        { label: 'What is saved', value: s.step === 4 ? '22/23 of a call' : 'quality, not cost', kind: s.step === 4 ? 'token' : 'text' },
      ];

  return (
    <Stage
      rootRef={ref}
      title="Semantic cache and semantic router: meaning as a lookup key"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['memory', ['token', 'true hit / small model'], ['compute', 'LLM call / large model'], ['error', 'false hit'], ['comm', 'router score']]}
      {...s}
      readout={<Readout items={readout} />}
      controls={
        <Slider
          label="Similarity threshold (cache steps; lower means wider circles)"
          value={thr}
          min={0.68}
          max={0.99}
          step={0.01}
          onChange={setThr}
          format={(v) => v.toFixed(2)}
          kind="comm"
        />
      }
    >
      <svg
        viewBox="0 0 640 340"
        role="img"
        aria-label="Left: a two-dimensional sketch of embedding space with stored question–answer pairs, each surrounded by a dashed circle whose radius is the similarity threshold. A near query lands inside a circle and is a green cache hit; a far query misses in amber and is sent to the LLM; lowering the threshold grows the circles and creates a red false hit. Right: the answer pipeline, then a semantic router sending an easy query to a small model and a hard query to a large model."
      >
        <Label x={8} y={18} tone={2} size={11} bold>Embedding space: similar meaning, small distance</Label>
        <Label x={412} y={18} anchor="end" tone={3} size={9.5}>
          {`threshold ${thr.toFixed(2)} · radius ${radius.toFixed(0)} px`}
        </Label>

        <rect x={8} y={30} width={404} height={266} rx={4} fill="var(--ground)" stroke="var(--rule)" />
        <rect x={420} y={30} width={212} height={266} rx={4} fill="var(--ground-2)" stroke="var(--rule)" />

        {/* faint axes */}
        <line x1={24} x2={396} y1={284} y2={284} className="v-rule faint" />
        <line x1={24} x2={24} y1={48} y2={284} className="v-rule faint" />
        <Label x={396} y={296} anchor="end" tone={3} size={8.5}>meaning axis 1</Label>
        <Label x={18} y={48} anchor="end" tone={3} size={8.5}>2</Label>

        {/* background queries */}
        {TRAFFIC.map((d, i) => (
          <circle
            key={i}
            cx={d.x}
            cy={d.y}
            r={2.2}
            fill="var(--text-3)"
            opacity={s.step >= 4 ? 0.2 : 0.55}
          />
        ))}

        {/* stored entries and their threshold circles */}
        {ENTRIES.map((e, i) => (
          <g key={i} opacity={s.step >= 4 ? 0.35 : 1}>
            <circle
              cx={e.x}
              cy={e.y}
              r={radius}
              fill="var(--memory-dim)"
              fillOpacity={0.22}
              stroke="var(--memory)"
              strokeWidth={1}
              strokeDasharray="3 3"
            />
            <circle cx={e.x} cy={e.y} r={4.5} fill="var(--memory)" stroke="var(--ground)" strokeWidth={1.5} />
            <Label x={e.x} y={e.y - 26} anchor="middle" tone={2} size={9.5} kind="memory">{e.label}</Label>
          </g>
        ))}

        {/* the pair stored after the miss */}
        <Fade show={s.step === 3}>
          <circle
            cx={STORED.x}
            cy={STORED.y}
            r={radius}
            fill="var(--memory-dim)"
            fillOpacity={0.18}
            stroke="var(--memory)"
            strokeWidth={1}
            strokeDasharray="3 3"
          />
          <circle cx={STORED.x} cy={STORED.y} r={4.5} fill="var(--memory)" stroke="var(--ground)" strokeWidth={1.5} />
          <Label x={STORED.x} y={STORED.y - 22} anchor="middle" tone={2} size={9} kind="memory">refund (stored)</Label>
        </Fade>

        {/* step 1: a true hit */}
        <Fade show={s.step >= 1 && s.step <= 3}>
          <line x1={Q1.x} y1={Q1.y} x2={ENTRIES[0].x} y2={ENTRIES[0].y} stroke="var(--token)" strokeWidth={1.2} opacity={0.55} />
          <circle cx={Q1.x} cy={Q1.y} r={5.5} fill="var(--token)" stroke="var(--ground)" strokeWidth={1.5} />
          <Label x={Q1.x + 10} y={Q1.y + 3.5} tone={1} size={9.5} kind="token">{Q1.label}</Label>
          <Label x={Q1.x + 10} y={Q1.y + 15} tone={3} size={8.5}>{`cos ${Q1.sim.toFixed(2)}`}</Label>
        </Fade>

        {/* step 2: a miss that goes to the LLM */}
        <Fade show={s.step === 2}>
          <circle cx={Q2.x} cy={Q2.y} r={5.5} fill="var(--compute)" stroke="var(--ground)" strokeWidth={1.5} />
          <Label x={Q2.x - 8} y={Q2.y + 3.5} anchor="end" tone={1} size={9.5} kind="compute">{Q2.label}</Label>
          <Label x={Q2.x - 8} y={Q2.y + 16} anchor="end" tone={3} size={8.5}>{`cos ${Q2.sim.toFixed(2)}`}</Label>
          <Arrow x1={Q2.x + 8} y1={Q2.y} x2={416} y2={Q2.y} kind="compute" />
          <Label x={356} y={Q2.y - 8} anchor="middle" tone={3} size={9}>one full call</Label>
        </Fade>

        {/* step 3: a false hit on a widened circle */}
        <Fade show={s.step === 3}>
          <line x1={Q3.x} y1={Q3.y} x2={205} y2={108} stroke="var(--error)" strokeWidth={1.4} strokeDasharray="4 3" />
          <circle cx={Q3.x} cy={Q3.y} r={5.5} fill="var(--error)" stroke="var(--ground)" strokeWidth={1.5} />
          <Label x={Q3.x + 10} y={Q3.y - 4} tone={1} size={9.5}>{Q3.label}</Label>
          <Label x={Q3.x + 10} y={Q3.y + 8} size={8.5} kind="error">false hit · cos 0.74</Label>
        </Fade>

        {/* router steps: the same vectors, marked easy and hard */}
        <Fade show={s.step >= 4}>
          <circle cx={Q1.x} cy={Q1.y} r={5.5} fill="var(--token)" stroke="var(--ground)" strokeWidth={1.5} />
          <Label x={Q1.x + 10} y={Q1.y + 3.5} tone={1} size={9.5} kind="token">easy · one fact</Label>
          <circle cx={Q4.x} cy={Q4.y} r={5.5} fill="var(--compute)" stroke="var(--ground)" strokeWidth={1.5} />
          <Label x={Q4.x + 10} y={Q4.y + 3.5} tone={1} size={9.5} kind="compute">{Q4.label}</Label>
        </Fade>

        {/* ── right panel: the cache pipeline (steps 0–3) ── */}
        <Fade show={cacheScene}>
          <Label x={436} y={50} tone={2} size={10.5} bold>Answer pipeline</Label>
          <Chip x={436} y={58} w={184} h={24} text="1 · embed query" kind="memory" />
          <Arrow x1={528} y1={82} x2={528} y2={90} kind="memory" />
          <Chip x={436} y={92} w={184} h={24} text="2 · vector search" kind="memory" />
          <Arrow x1={528} y1={116} x2={528} y2={124} kind="memory" />
          <Chip x={436} y={126} w={184} h={24} text={decision.text} kind={decision.kind} />

          <Fade show={s.step === 0}>
            <Chip x={436} y={168} w={184} h={24} text="cache empty" kind="idle" />
          </Fade>
          <Fade show={s.step === 1}>
            <Chip x={436} y={168} w={184} h={24} text="return stored answer" kind="token" />
            <Chip x={436} y={200} w={184} h={24} text="no prefill · no decode" kind="idle" />
          </Fade>
          <Fade show={s.step === 2}>
            <Chip x={436} y={168} w={184} h={24} text="call the LLM · cost C" kind="compute" />
            <Chip x={436} y={200} w={184} h={24} text="store query + answer" kind="memory" />
          </Fade>
          <Fade show={s.step === 3}>
            <Chip x={436} y={168} w={184} h={24} text={falseHit ? 'wrong answer · cost W' : 'no wrong answer here'} kind={falseHit ? 'error' : 'token'} />
            <Chip x={436} y={200} w={184} h={24} text={falseHit ? 'saving can go negative' : 'saving stays positive'} kind={falseHit ? 'error' : 'token'} />
          </Fade>

          {NOTES[s.step].map((line, i) => (
            <Label key={i} x={436} y={248 + i * 15} tone={3} size={9.5} mono={false}>{line}</Label>
          ))}
        </Fade>

        {/* ── right panel: the router (steps 4–5) ── */}
        <Fade show={s.step >= 4}>
          <Label x={436} y={50} tone={2} size={10.5} bold>Semantic router</Label>
          <Chip x={436} y={58} w={184} h={24} text="1 · embed query" kind="memory" />
          <Arrow x1={528} y1={82} x2={528} y2={90} kind="memory" />
          <Chip x={436} y={92} w={184} h={24} text="2 · complexity score" kind="comm" />

          <Chip x={430} y={138} w={88} h={42} text="small 3B" kind={s.step === 4 ? 'token' : 'text'} />
          <Chip x={526} y={138} w={88} h={42} text="large 70B" kind={s.step === 5 ? 'compute' : 'text'} />
          <Label x={474} y={194} anchor="middle" tone={3} size={9}>1× cost</Label>
          <Label x={570} y={194} anchor="middle" tone={3} size={9}>≈23× cost</Label>

          <Fade show={s.step === 4}>
            <Arrow x1={524} y1={116} x2={480} y2={136} kind="token" />
          </Fade>
          <Fade show={s.step === 5}>
            <Arrow x1={524} y1={116} x2={566} y2={136} kind="compute" />
          </Fade>

          <line x1={430} x2={614} y1={212} y2={212} className="v-rule" />
          <line
            x1={s.step === 4 ? 430 + 0.22 * 184 : 430 + 0.78 * 184}
            x2={s.step === 4 ? 430 + 0.22 * 184 : 430 + 0.78 * 184}
            y1={204}
            y2={220}
            stroke={s.step === 4 ? 'var(--token)' : 'var(--compute)'}
            strokeWidth={2}
          />
          <Label x={430} y={232} tone={3} size={9}>easy 0</Label>
          <Label x={614} y={232} anchor="end" tone={3} size={9}>hard 1</Label>

          {NOTES[s.step].map((line, i) => (
            <Label key={i} x={436} y={252 + i * 15} tone={3} size={9.5} mono={false}>{line}</Label>
          ))}
        </Fade>
      </svg>
    </Stage>
  );
}
