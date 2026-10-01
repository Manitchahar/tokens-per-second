import { useMemo, useRef, useState } from 'react';
import { Stage, Segmented, Readout, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Chip, Arrow, Fade, Dim, Bar, fmt } from '../lib/svg';
import {
  ROLLOUT_MODELS, rolloutById, rolloutCost, rolloutKvPerToken, raggedness, rolloutHeadroom, gpu, B,
} from './workloads';

const STEPS = [
  'RL needs thousands of whole generations',
  'The KV cache is the entire problem',
  'Sizing a rollout group to a cluster',
  'Nobody is waiting, so batch until it hurts',
  'The batch shrinks as rollouts finish',
  'Why the tail of a rollout is slow',
  'Generation length: the dial people get wrong',
  'What a rollout budget looks like',
];
const CAPTIONS = [
  'Reinforcement learning on a language model means: take a frozen policy, generate many complete answers per prompt, score them, and update. One optimisation step needs an entire group of finished generations — thousands of sequences, each thousands of tokens.',
  'Each rollout is a full conversation, so each one holds a full KV cache. Multiply sequence length by group size and you get the number that decides your cluster size. This is the same 2LHkvD formula as everywhere else in the course, applied to a much larger multiplier.',
  'Fit the group across your GPUs: weights stay resident once, KV cache does not. The number that must fit is the total KV of the whole group, and it dwarfs the weights by orders of magnitude.',
  'Unlike serving, no user is waiting on a single rollout. Throughput is the only objective, so you run a batch far larger than anything you would serve to humans — and decode, being memory-bound, barely slows down as the batch grows.',
  'Rollouts finish at different lengths, so the batch is ragged: it starts at the full group and drains continuously. This is exactly backwards from serving, where finished requests are replaced and the batch stays full.',
  'As the batch shrinks, the weight read is amortised over fewer tokens and arithmetic intensity rises toward the compute roofline. The last few rollouts run at a fraction of the throughput the first ones did.',
  'Cap the generation length hard, or a fraction of rollouts never terminate and the step hangs. But keep the cap well inside the context limit — a rollout that hits the cap has no room left for the reward computation.',
  'Budget it honestly: GPUs needed, wall-clock per optimisation step, and tokens per second across the cluster. Those three numbers are what a research team plans a run around.',
];

export default function BatchRlRollouts() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;

  const [mid, setMid] = useState<string>('qwen2.5-7b');
  const [group, setGroup] = useState(512);
  const [seq, setSeq] = useState(4096);
  const [gpus, setGpus] = useState(8);

  const base = rolloutById(mid);
  const cfg = useMemo(() => ({ ...base, groupSize: group, seqLen: seq, gpus }), [base, group, seq, gpus]);
  const cost = useMemo(() => rolloutCost(cfg), [cfg]);
  const rag = useMemo(() => raggedness(cfg, 0.8), [cfg]);
  const head = useMemo(() => rolloutHeadroom(cfg, 0.75), [cfg]);
  const g = gpu(base.gpuId);

  const kvT = rolloutKvPerToken(cfg);
  const card = 80e9;
  const usedFrac = cost.totalBytes / (card * gpus);

  // Drain curve: batch size over the generation window.
  const drain = useMemo(() => {
    const pts: { at: number; batch: number }[] = [];
    for (let i = 0; i <= 20; i++) {
      const at = i / 20;
      // Completions spread uniformly over the window, ending at the last one.
      const fracDone = Math.min(1, at / 0.8);
      const remaining = Math.max(0, 1 - fracDone);
      pts.push({ at, batch: Math.max(1, Math.round(group * remaining)) });
    }
    return pts;
  }, [group]);

  return (
    <Stage
      rootRef={ref}
      title="Batch inference: RL rollouts to completion"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['memory', 'KV cache'],
        ['compute', 'weights, read every step'],
        ['token', 'generated tokens'],
        ['error', 'does not fit / does not finish'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'KV per token', value: fmt.bytes(kvT), kind: 'memory', note: `${base.layers} layers, ${base.kvHeads} KV heads` },
          { label: 'KV for the group', value: fmt.bytes(cost.totalKvBytes), kind: cost.fits ? 'memory' : 'error', note: `${fmt.int(group)} rollouts × ${fmt.int(seq)} tokens` },
          { label: 'Rollouts per GPU', value: fmt.int(cost.fitPerGpu), kind: cost.fitPerGpu > 0 ? 'token' : 'error', note: `needs ${cost.gpusNeeded} of ${gpus}` },
          { label: 'Cluster throughput', value: `${fmt.int(cost.tpsCluster)} tok/s`, kind: 'compute', note: `${cost.wallS.toFixed(0)}s per group` },
        ]} />
      }
      controls={
        <>
          <Segmented label="Policy model" value={mid} options={ROLLOUT_MODELS.map((x) => ({ value: x.id, label: x.modelId.replace('qwen2.5-', 'Qwen-').replace('llama3.1-', 'Llama-') }))} onChange={setMid} />
          <Slider label="Rollouts per group" kind="token" min={64} max={4096} step={64} value={group} onChange={setGroup} format={(v) => fmt.int(v)} />
          <Slider label="Tokens per rollout" kind="memory" min={512} max={16384} step={512} value={seq} onChange={setSeq} format={(v) => fmt.int(v)} />
          <Slider label="GPUs" kind="compute" min={1} max={64} step={1} value={gpus} onChange={setGpus} format={(v) => `${v} GPUs`} />
        </>
      }
    >
      <svg viewBox="0 0 640 340" role="img" aria-label="The KV cache size of an RL rollout group against GPU memory, the draining batch over the generation window, and the generation length budget.">
        {/* ── panel A: the group ── */}
        <Label x={12} y={20} tone={2} size={11} bold>One optimisation step, one group of rollouts</Label>
        <Fade show={st >= 0}>
          <g>
            <Label x={12} y={44} tone={3} size={9.5} mono={false}>prompt</Label>
            <Chip x={60} y={34} text={`${base.modelId}`} kind="text" h={18} w={86} size={9} />
            <Arrow x1={150} y1={43} x2={180} y2={43} kind="compute" />
            <Label x={184} y={40} tone={3} size={9} mono={false}>{`× ${fmt.int(group)} rollouts`}</Label>
            <Arrow x1={260} y1={43} x2={290} y2={43} kind="token" />
            <Label x={294} y={40} tone={3} size={9} mono={false}>{`${fmt.int(seq)} tokens each`}</Label>
            <Label x={294} y={54} tone={2} size={9}>{`= ${B(group * seq)} tokens total`}</Label>
          </g>
        </Fade>

        {/* ── panel B: KV is the problem ── */}
        <Fade show={st >= 1}>
          <g>
            <Label x={12} y={80} tone={2} size={11} bold>The KV cache is the whole sizing problem</Label>
            <Label x={12} y={100} tone={3} size={9.5} mono={false}>weights ({B(base.params)} × 2 B, resident once)</Label>
            <Bar x={230} y={92} w={270} h={13} frac={(base.params * 2) / (card * gpus)} kind="compute" />
            <Label x={506} y={103} tone={1} size={9.5}>{fmt.bytes(base.params * 2)}</Label>
            <Label x={12} y={120} tone={3} size={9.5} mono={false}>KV cache for the whole group</Label>
            <Bar x={230} y={112} w={270} h={13} frac={cost.totalKvBytes / (card * gpus)} kind="memory" />
            <Label x={506} y={123} tone={1} size={9.5}>{fmt.bytes(cost.totalKvBytes)}</Label>
            <Dim x1={230} x2={500} y={140} label={`${gpus} × ${g.name} = ${fmt.gb(card * gpus)}`} />
            <Label x={12} y={158} size={9.5} kind={usedFrac > 1 ? 'error' : 'token'}>
              {usedFrac > 1
                ? `Over budget by ${((usedFrac - 1) * 100).toFixed(0)}% — needs ${cost.gpusNeeded} GPUs, you have ${gpus}.`
                : `${(usedFrac * 100).toFixed(0)}% of cluster memory; ${cost.fitPerGpu} rollouts fit per GPU.`}
            </Label>
          </g>
        </Fade>

        {/* ── panel C: batch big, nobody waiting ── */}
        <Fade show={st >= 3}>
          <g>
            <Label x={12} y={184} tone={2} size={11} bold>No user is waiting, so batch until it hurts</Label>
            <Label x={12} y={204} tone={3} size={9.5} mono={false}>batch</Label>
            {[64, 256, 1024, 4096].map((b, i) => {
              const over = b > cost.fitTotal;
              return (
                <g key={b}>
                  <Label x={60} y={212 + i * 22} tone={3} size={9}>{fmt.int(b)}</Label>
                  <Bar x={110} y={204 + i * 22} w={Math.min(300, b / 4096 * 300)} h={12} frac={b / 4096} kind={over ? 'error' : 'token'} ghost={over} />
                  <Label x={420} y={214 + i * 22} tone={3} size={9}>{over ? 'won’t fit' : ''}</Label>
                </g>
              );
            })}
            <Label x={440} y={226} tone={3} size={9.5} mono={false}>decode is memory-bound:</Label>
            <Label x={440} y={242} tone={3} size={9.5} mono={false}>a bigger batch costs</Label>
            <Label x={440} y={258} tone={3} size={9.5} mono={false}>almost nothing.</Label>
          </g>
        </Fade>

        {/* ── panel D: the ragged drain ── */}
        <Fade show={st >= 4}>
          <g>
            <Label x={12} y={296} tone={2} size={9.5}>Batch over the generation window</Label>
            {drain.map((p, i) => {
              const x = 130 + p.at * 330;
              const h = (p.batch / group) * 60;
              return <rect key={i} x={x} y={320 - h} width={13} height={h} fill="var(--token)" opacity={0.8} className="v-cell" />;
            })}
            <line x1={130} x2={460} y1={320} y2={320} className="v-rule" />
            <Label x={130} y={334} tone={3} size={8.5}>start {fmt.int(group)}</Label>
            <Label x={460} y={334} anchor="end" tone={3} size={8.5}>drains to 1</Label>
          </g>
        </Fade>

        {/* ── panel E: the tail and the cap ── */}
        <Fade show={st >= 5}>
          <g>
            <Label x={480} y={296} tone={3} size={8.5} mono={false}>{`avg batch ${fmt.int(rag.avgBatch)} of ${fmt.int(group)}`}</Label>
            <Label x={480} y={310} tone={2} size={8.5}>{`intensity ${rag.intensity.toFixed(1)} FLOP/B`}</Label>
            <Label x={480} y={324} size={8.5} kind={rag.ridgeGap >= 1 ? 'token' : 'memory'}>{`ridge at ${rag.ridgeIntensity.toFixed(1)}`}</Label>
          </g>
        </Fade>

        <Fade show={st >= 6}>
          <g>
            <Chip x={12} y={160} text="no EOS?" kind="error" h={18} w={56} size={9} />
            <Label x={76} y={173} tone={3} size={9.5}>{`cap at ${head.stopsAt.toFixed(0)}% of context = ${fmt.int(head.cap)} tokens; ${fmt.int(head.slack)} left for the reward model`}</Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}