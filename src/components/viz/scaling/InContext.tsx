import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Fade, Span, Arrow, Chip } from '../lib/svg';
import { GPT3_SIZES, count, sci, logScale } from './scale';

// In-context learning: the same frozen model, a prompt with K solved examples in it,
// and an accuracy that rises with K far more steeply for big models than for small ones.
// The curves are ILLUSTRATIVE: their shape follows Brown et al. 2020 (Figures 1.2 and 1.3:
// flat for small models, a widening few-shot gap for large ones), not the paper's data.

const STEPS = [
  'The old way: one finetuned model per task',
  'Zero-shot: just describe the task',
  'One-shot: show one solved example',
  'Few-shot: show K examples',
  'The weights never change',
  'Bigger models learn more from the prompt',
  'The context window caps K',
];
const CAPTIONS = [
  'Before GPT-3, a new task meant collecting thousands of labelled examples, running gradient descent, and keeping a separate copy of the weights for that task.',
  'Zero-shot: the prompt holds only a description of the task and the question. The model has to guess what is wanted from the words alone.',
  'One-shot: add a single solved example. The model now sees the format of the answer, not just a description of it.',
  'Few-shot: add K solved examples, as many as fit. Accuracy climbs with K, and for the biggest model it climbs a long way.',
  'Nothing in the weights moved. No gradient, no update. The examples are only input text; delete them and the model is exactly what it was.',
  'At 125M parameters, examples barely help. At 175B the gap between zero-shot and few-shot is wide. In-context learning is something scale buys.',
  'Every example costs tokens. A 2,048-token window holds about a hundred short examples, and every one of them is prefill work on every query.',
];

// the toy task: remove the symbols to recover the word (in the spirit of the paper's
// symbol-insertion task in Figure 1.2; the examples are ours)
const DESC = 'Remove the symbols to get the word.';
const EXAMPLES = [
  ['r.i!v/e.r', 'river'],
  ['c/l.o!u.d', 'cloud'],
  ['s!t.o/n.e', 'stone'],
  ['g.a/r!d.e/n', 'garden'],
  ['b!r/i.d.g/e', 'bridge'],
  ['w/i.n!t/e.r', 'winter'],
];
const QUERY = ['p.l!a/n.e/t', 'planet'];
const TOK_DESC = 10;
const TOK_EX = 16; // symbols split into ~1 token per character
const TOK_Q = 10;
const CTX = 2048;
const K_MAX = Math.floor((CTX - TOK_DESC - TOK_Q) / TOK_EX); // 126

const LOG_SMALL = Math.log10(125e6);
const LOG_BIG = Math.log10(175e9);
/** Illustrative accuracy for a model of N parameters given K examples in context. */
function acc(N: number, K: number) {
  const s = Math.max(0, Math.min(1, (Math.log10(N) - LOG_SMALL) / (LOG_BIG - LOG_SMALL)));
  const zero = 0.02 + 0.1 * s * s;
  const gain = 0.03 + 0.52 * Math.pow(s, 2.5);
  const k = Math.min(K, K_MAX);
  const f = Math.min(1, Math.log(1 + k) / Math.log(101));
  return zero + gain * f;
}

// chart geometry
const X0 = 356, X1 = 604, Y0 = 266, Y1 = 64;
const px = logScale(1e8, 1e12, X0, X1);
const py = (a: number) => Y0 - (a / 0.8) * (Y0 - Y1);

const PRESET: Record<number, { K: number; size?: number }> = {
  0: { K: 0, size: 7 }, 1: { K: 0, size: 7 }, 2: { K: 1, size: 7 }, 3: { K: 16, size: 7 },
  4: { K: 16 }, 5: { K: 32, size: 7 }, 6: { K: 140, size: 7 },
};

function path(K: number) {
  return GPT3_SIZES.map((m, i) => `${i ? 'L' : 'M'}${px(m.N).toFixed(1)},${py(acc(m.N, K)).toFixed(1)}`).join(' ');
}

export default function InContext() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref, { interval: 3400 });
  const st = s.step;
  const [K, setK] = useState(0);
  const [size, setSize] = useState(7);

  useEffect(() => {
    const p = PRESET[st];
    if (p) { setK(p.K); if (p.size !== undefined) setSize(p.size); }
  }, [st]);

  const model = GPT3_SIZES[size];
  const kEff = Math.min(K, K_MAX);
  const tokens = TOK_DESC + K * TOK_EX + TOK_Q;
  const overflow = tokens > CTX;
  const usedTokens = Math.min(tokens, CTX);
  const a = acc(model.N, K);
  const prefill = 2 * model.N * usedTokens;

  const yourPath = useMemo(() => path(Math.max(2, K)), [K]);
  const zeroPath = useMemo(() => path(0), []);
  const onePath = useMemo(() => path(1), []);
  const mx = useTween(px(model.N), 500);
  const my = useTween(py(a), 500);

  const scene = st === 0 ? 'finetune' : 'prompt';
  const shown = EXAMPLES.slice(0, Math.min(K, EXAMPLES.length));
  const more = K - shown.length;

  // prompt layout
  const PX = 16, PY = 58, LH = 19;
  const qY = PY + 26 + (shown.length + (more > 0 ? 1 : 0)) * LH + 8;

  const gapSmall = acc(125e6, 32) - acc(125e6, 0);
  const gapBig = acc(175e9, 32) - acc(175e9, 0);

  return (
    <Stage
      rootRef={ref}
      title="Learning from the prompt: same weights, K examples"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['token', 'examples in the prompt'], ['memory', 'weights'], ['compute', 'training / prefill FLOPs'], ['error', 'over the window']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Examples in the prompt', value: `K = ${K}`, kind: 'token', note: K === 0 ? 'zero-shot' : K === 1 ? 'one-shot' : 'few-shot' },
          { label: 'Prompt length', value: `${tokens.toLocaleString('en-US')} tokens`, kind: overflow ? 'error' : 'text', note: overflow ? `over the ${CTX.toLocaleString('en-US')}-token window` : `of ${CTX.toLocaleString('en-US')} in the window` },
          { label: 'Accuracy (illustrative)', value: `${(a * 100).toFixed(0)}%`, kind: 'token', note: `GPT-3 ${model.name === '175B' ? '175B' : count(model.N)}, weights frozen` },
          { label: 'Prefill per query', value: `${sci(prefill, 1)} FLOPs`, kind: 'compute', note: '2N × prompt tokens' },
        ]} />
      }
      controls={
        <>
          <Slider label="Solved examples in the prompt (K)" kind="token" value={K} min={0} max={140} onChange={setK} format={(v) => `${v}`} />
          <Segmented label="GPT-3 model size" value={size} options={GPT3_SIZES.map((m, i) => ({ value: i, label: m.name === 'Small' ? '125M' : m.name === 'Medium' ? '350M' : m.name === 'Large' ? '760M' : m.name === 'XL' ? '1.3B' : m.name }))} onChange={setSize} />
        </>
      }
    >
      <svg viewBox="0 0 640 340" role="img" aria-label="Left: a text prompt containing a task description, K solved examples of removing symbols from a word, and a final question, with a bar showing how much of the 2,048-token window it uses. Right: illustrative accuracy against GPT-3 model size for zero-shot, one-shot and few-shot prompts; the few-shot curve rises far above the others only for the largest models.">
        {/* ─────────────── scene 0: one finetuned copy per task */}
        <Fade show={scene === 'finetune'}>
          <g>
            <Label x={16} y={26} tone={2} size={11} bold>Before: a new task means a new training run</Label>
            {['sentiment of a review', 'English → French', 'answer a trivia question'].map((task, i) => {
              const y = 58 + i * 74;
              return (
                <g key={task}>
                  <Label x={16} y={y - 6} tone={3} size={9.5} mono={false}>{`task ${i + 1}: ${task}`}</Label>
                  {Array.from({ length: 6 }, (_, j) => (
                    <rect key={j} x={16 + j * 22} y={y} width={18} height={26} rx={2} fill="var(--token-dim)" stroke="var(--token)" strokeWidth={0.8} />
                  ))}
                  <Label x={16} y={y + 42} tone={3} size={9} mono={false}>thousands of labelled examples</Label>
                  <Arrow x1={156} y1={y + 13} x2={214} y2={y + 13} kind="compute" />
                  <Label x={185} y={y + 2} anchor="middle" size={9} kind="compute">gradient</Label>
                  <Label x={185} y={y + 30} anchor="middle" size={9} kind="compute">descent</Label>
                  <Span x={222} y={y} w={150} h={26} kind="memory" label={`weights copy #${i + 1}`} />
                  <Label x={382} y={y + 17} tone={2} size={9.5} mono={false}>a model that does only this task</Label>
                </g>
              );
            })}
            <Label x={16} y={298} size={10} tone={1} mono={false}>Three tasks: three datasets, three training runs, three copies of the weights to store and serve.</Label>
            <Label x={16} y={318} size={9.5} tone={3} mono={false}>GPT-3 asked a different question: can one frozen model pick up the task from the prompt alone?</Label>
          </g>
        </Fade>

        {/* ─────────────── scenes 1–6: the prompt and the curves */}
        <Fade show={scene === 'prompt'}>
          <g>
            <Label x={PX} y={26} tone={2} size={11} bold>The prompt: input text, nothing else</Label>
            <rect x={PX} y={38} width={300} height={232} rx={4} fill="var(--ground-3)" stroke={overflow ? 'var(--error)' : 'var(--rule-2)'} />
            <Label x={PX + 10} y={PY} size={10} tone={1} mono={false}>{DESC}</Label>
            {shown.map(([q, ans], i) => (
              <g key={q} className="v-fade">
                <Label x={PX + 10} y={PY + 26 + i * LH} size={10.5} tone={2}>{q}</Label>
                <Label x={PX + 112} y={PY + 26 + i * LH} size={10.5} tone={3}>→</Label>
                <Label x={PX + 130} y={PY + 26 + i * LH} size={10.5} kind="token">{ans}</Label>
              </g>
            ))}
            {more > 0 && (
              <Label x={PX + 10} y={PY + 26 + shown.length * LH} size={9.5} tone={3} mono={false}>
                {`… ${more} more solved example${more === 1 ? '' : 's'} …`}
              </Label>
            )}
            <line x1={PX + 10} x2={PX + 290} y1={qY - 14} y2={qY - 14} className="v-rule faint" />
            <Label x={PX + 10} y={qY} size={10.5} tone={1}>{QUERY[0]}</Label>
            <Label x={PX + 112} y={qY} size={10.5} tone={3}>→</Label>
            <Chip x={PX + 128} y={qY - 15} text={a >= 0.5 ? QUERY[1] : 'p.lanet?'} kind={a >= 0.5 ? 'token' : 'error'} dim size={10} />
            <Label x={PX + 214} y={qY} size={9} tone={3} mono={false}>{`${(a * 100).toFixed(0)}% right`}</Label>

            {/* the frozen weights */}
            <Fade show={st >= 4}>
              <g>
                <Span x={PX} y={280} w={300} h={22} kind="memory" dim />
                <Label x={PX + 150} y={295} anchor="middle" size={10} kind="memory" bold>weights frozen · 0 gradient updates</Label>
              </g>
            </Fade>

            {/* context window */}
            <g>
              <Label x={PX} y={322} size={9} tone={3} mono={false}>context window</Label>
              <rect x={PX + 84} y={313} width={216} height={10} rx={2} fill="none" stroke={overflow ? 'var(--error)' : 'var(--rule-2)'} />
              <rect x={PX + 84} y={313} width={216 * (usedTokens / CTX)} height={10} rx={2} fill={overflow ? 'var(--error)' : 'var(--token)'} style={{ transition: 'width .4s var(--ease-out), fill .4s' }} />
              <Label x={PX + 84} y={337} size={8.5} tone={3}>{`${usedTokens.toLocaleString('en-US')} / ${CTX.toLocaleString('en-US')}${overflow ? '' : ' tokens'}`}</Label>
              {overflow && <Label x={PX + 300} y={337} anchor="end" size={8.5} kind="error">{`only ${K_MAX} examples fit`}</Label>}
            </g>

            {/* ── the chart */}
            <Label x={X0 - 20} y={26} tone={2} size={11} bold>Accuracy vs model size</Label>
            <line x1={X0} x2={X1 + 20} y1={Y0} y2={Y0} className="v-rule" />
            <line x1={X0} x2={X0} y1={Y0} y2={Y1 - 8} className="v-rule" />
            {[0, 0.2, 0.4, 0.6, 0.8].map((v) => (
              <g key={v}>
                <line x1={X0} x2={X1 + 20} y1={py(v)} y2={py(v)} className="v-rule faint" />
                <Label x={X0 - 6} y={py(v) + 3} anchor="end" tone={3} size={8.5}>{`${v * 100}%`}</Label>
              </g>
            ))}
            {[1e8, 1e9, 1e10, 1e11, 1e12].map((v) => (
              <Label key={v} x={px(v)} y={Y0 + 14} anchor="middle" tone={3} size={8.5}>{count(v)}</Label>
            ))}
            <Label x={(X0 + X1) / 2} y={Y0 + 28} anchor="middle" tone={3} size={9} mono={false}>parameters (log scale)</Label>

            <Fade show={st >= 1}>
              <path d={zeroPath} fill="none" stroke="var(--text-3)" strokeWidth={1.4} strokeDasharray="4 3" />
              <Label x={px(175e9) + 6} y={py(acc(175e9, 0)) + 4} size={9} tone={3}>zero</Label>
            </Fade>
            <Fade show={st >= 2}>
              <path d={onePath} fill="none" stroke="var(--text-2)" strokeWidth={1.4} />
              <Label x={px(175e9) + 6} y={py(acc(175e9, 1)) + 4} size={9} tone={2}>one</Label>
            </Fade>
            <Fade show={st >= 3 && K >= 2}>
              <path d={yourPath} fill="none" stroke="var(--token)" strokeWidth={1.8} style={{ transition: 'd .4s var(--ease-out)' }} />
              <Label x={px(175e9) + 6} y={py(acc(175e9, Math.max(2, K))) + 4} size={9} kind="token">{`K=${Math.min(K, K_MAX)}`}</Label>
            </Fade>
            <Fade show={st >= 1}>
              {GPT3_SIZES.map((m) => (
                <circle key={m.name} cx={px(m.N)} cy={py(acc(m.N, 0))} r={2} fill="var(--text-3)" />
              ))}
            </Fade>

            {/* the widening gap */}
            <Fade show={st === 5}>
              <g>
                <line x1={px(125e6) - 8} x2={px(125e6) - 8} y1={py(acc(125e6, 0))} y2={py(acc(125e6, 32))} stroke="var(--token)" strokeWidth={1.4} />
                <Label x={px(125e6) + 4} y={py(acc(125e6, 32)) - 8} size={9} kind="token">{`+${(gapSmall * 100).toFixed(0)} pts`}</Label>
                <line x1={px(175e9) - 8} x2={px(175e9) - 8} y1={py(acc(175e9, 0))} y2={py(acc(175e9, 32))} stroke="var(--token)" strokeWidth={1.4} />
                <Label x={px(175e9) + 6} y={py(acc(175e9, 32)) + 22} size={9} kind="token">{`+${(gapBig * 100).toFixed(0)} pts`}</Label>
              </g>
            </Fade>

            {/* the selected model */}
            <Fade show={st >= 1}>
              <g className="v-move" style={{ transform: `translate(${mx}px, ${my}px)` }}>
                <circle r={5.5} fill="var(--token)" stroke="var(--ground-2)" strokeWidth={2.5} />
              </g>
            </Fade>

            <Label x={X1 + 20} y={Y0 + 44} anchor="end" tone={3} size={8.5} mono={false}>illustrative curves; shape after Brown et al. 2020, Figs 1.2–1.3</Label>
            <Fade show={st === 6}>
              <Label x={X1 + 20} y={Y0 + 60} anchor="end" size={9} kind="compute" mono={false}>{`every query re-reads the examples: ${sci(prefill, 1)} FLOPs of prefill`}</Label>
            </Fade>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}
