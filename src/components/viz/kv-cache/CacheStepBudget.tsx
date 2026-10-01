import { useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTweens } from '../lib/tween';
import { Label, Span, Fade, fmt } from '../lib/svg';
import { GPUS, MODELS, gpuById, modelById } from '../lib/perfModel';
import { kvPerToken, weightBytes, naiveForwards, cachedForwards, attnFlops, linearFlops } from './kvMath';

const MFU = 0.55, MBU = 0.8, OVERHEAD = 0.0015;
const BATCH = 32;
const BX = 24, BW = 600;

const gb = (b: number) => (b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : `${Math.round(b / 1e6)} MB`);
const tf = (n: number) => (n >= 1e15 ? `${(n / 1e15).toFixed(2)} PFLOP` : n >= 1e12 ? `${(n / 1e12).toFixed(2)} TFLOP` : `${(n / 1e9).toFixed(1)} GFLOP`);

export default function CacheStepBudget() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(1, ref);
  const [modelId, setModelId] = useState('llama8b');
  const [gpuId, setGpuId] = useState('h100');
  const [p, setP] = useState(512);
  const [n, setN] = useState(128);

  const model = modelById(modelId);
  const gpu = gpuById(gpuId);
  const T = p + n;

  const w = weightBytes(model, 2);
  const per = kvPerToken(model, 2);
  const kvRead = T * per;

  const tW = w / (gpu.bw * MBU);
  const tK = kvRead / (gpu.bw * MBU);
  const tMem = tW + tK;
  const flops = 2 * model.params + 4 * model.hidden * T * model.layers;
  const tC = flops / (gpu.flops * MFU);
  const tOver = Math.max(tMem, tC) + OVERHEAD;

  const naive = naiveForwards(p, n);
  const cached = cachedForwards(p, n);
  const naiveS = (naive * 2 * model.params) / (gpu.flops * MFU);
  const cachedS = Math.max(linearFlops(model, p) + attnFlops(model, p), w / (gpu.bw * MBU)) + n * tOver;

  const ws = 0.04 * gpu.mem;
  const avail = Math.max(0, (gpu.mem - w - ws) * 0.9);
  const seqs = Math.floor(avail / (T * per));

  const seg = {
    w: BW * (tW / tOver),
    k: BW * (tK / tOver),
    c: BW * (Math.max(0, tC - tMem) / tOver),
    o: BW * (OVERHEAD / tOver),
  };
  const tw = useTweens(seg, 600);
  const x1 = BX + tw.w;
  const x2 = x1 + tw.k;
  const x3 = x2 + tw.c;

  return (
    <Stage
      rootRef={ref}
      title="Cached or not: what a decode step costs, what it buys, what it holds"
      steps={['Budget']}
      captions={[`One request, one model, one GPU. The cache is simultaneously the reason the request is fast, the reason the step slows down, and the reason the GPU eventually runs out of room.`]}
      legend={['memory', 'compute', 'token', 'error', 'idle']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Decode step', value: fmt.ms(tOver), kind: tC > tMem ? 'compute' : 'memory', note: tC > tMem ? 'compute-bound' : `cache is ${((tK / tMem) * 100).toFixed(0)}% of the memory read` },
          { label: 'Cache per token', value: `${fmt.int(per / 1024)} KB`, kind: 'memory', note: `${model.layers} layers, ${model.kvHeads} KV heads` },
          { label: 'Whole request', value: `${fmt.ms(naiveS)} → ${fmt.ms(cachedS)}`, kind: 'token', note: `${(naiveS / cachedS).toFixed(1)}× faster with a cache, ${(naive / cached).toFixed(0)}× less arithmetic` },
          { label: `Conversations that fit`, value: fmt.int(seqs), kind: seqs > 0 ? 'token' : 'error', note: `${fmt.int(T)} tokens each on ${gpu.id.toUpperCase()}` },
        ]} />
      }
      controls={
        <>
          <Segmented label="Model" value={modelId} options={MODELS.filter((m) => ['qwen05', 'llama8b', 'qwen32b', 'llama70b'].includes(m.id)).map((m) => ({ value: m.id, label: m.name.replace('Llama-3.1-', '').replace('Qwen2.5-', '') }))} onChange={setModelId} />
          <Segmented label="GPU" value={gpuId} options={GPUS.filter((g) => ['t4', 'a100', 'h100', 'h200'].includes(g.id)).map((g) => ({ value: g.id, label: g.id.toUpperCase() }))} onChange={setGpuId} />
          <Slider label="Prompt length" value={p} min={64} max={8192} log onChange={setP} format={(v) => `${fmt.int(v)} tokens`} kind="memory" />
          <Slider label="Tokens generated" value={n} min={8} max={1024} log onChange={setN} format={(v) => `${fmt.int(v)}`} kind="token" />
        </>
      }
    >
      <svg viewBox="0 0 640 372" role="img" aria-label="A decode step broken into weight reads, cache reads, compute and overhead; what the cache saves in arithmetic and wall clock; and how much memory the cache holds at 32 concurrent conversations.">
        <Label x={BX} y={18} tone={2} size={10.5} mono={false} bold>{`${model.name} on one ${gpu.name} · ${fmt.int(p)}-token prompt, ${fmt.int(n)}-token answer`}</Label>

        <Label x={BX} y={40} tone={3} size={10}>One decode step, decomposed</Label>
        <rect x={BX} y={48} width={BW} height={20} rx={2} fill="none" stroke="var(--rule-2)" />
        <Span x={BX} y={48} w={tw.w} h={20} kind="memory" label={`read weights ${(w / 1e9).toFixed(1)} GB`} />
        <Span x={x1} y={48} w={tw.k} h={20} kind="memory" dim label={tw.k > 90 ? `cache ${gb(kvRead)}` : undefined} labelTone="light" />
        {tw.c > 1 && <Span x={x2} y={48} w={tw.c} h={20} kind="compute" label={tw.c > 90 ? 'compute' : undefined} />}
        <Span x={x3} y={48} w={Math.max(0, BX + BW - x3)} h={20} kind="idle" label="overhead" labelTone="light" />
        <Label x={BX} y={84} tone={2} size={10.5} mono={false}>
          {`${fmt.ms(tOver)} per token — ${(1000 / tOver).toFixed(0)} tokens/s for one user. Math is ${tf(flops)}, or ${((tC / tOver) * 100).toFixed(0)}% of the step.`}
        </Label>
        <Label x={BX} y={100} size={10} tone={3} mono={false}>
          {tC > tMem
            ? 'Compute-bound: at this context the arithmetic has caught up with the memory read.'
            : `Memory-bound: the cache read is ${((tK / tMem) * 100).toFixed(0)}% of the bytes and rising with every token generated.`}
        </Label>

        <line x1={BX} x2={624} y1={114} y2={114} className="v-rule faint" />
        <Label x={BX} y={130} tone={3} size={10}>The whole request, side by side</Label>
        <Label x={330} y={130} anchor="middle" size={9} kind="error" bold mono={false}>recomputing</Label>
        <Label x={470} y={130} anchor="middle" size={9} kind="token" bold mono={false}>with a cache</Label>
        <Label x={556} y={130} anchor="middle" size={9} tone={3} mono={false}>gain</Label>
        {[
          ['token-forwards', fmt.int(naive), fmt.int(cached), 'error'],
          ['arithmetic', tf(naive * 2 * model.params), tf(cached * 2 * model.params), 'error'],
          ['wall clock, this GPU', fmt.ms(naiveS), fmt.ms(cachedS), 'token'],
          ['memory held', '0', gb(T * per), 'memory'],
        ].map(([k, a, b], i) => (
          <g key={k}>
            <Label x={BX} y={152 + i * 17} tone={2} size={10} mono={false}>{k}</Label>
            <Label x={330} y={152 + i * 17} size={10} kind="error" bold>{a}</Label>
            <Label x={470} y={152 + i * 17} size={10} kind={b === 'token' ? 'token' : 'memory'} bold>{b}</Label>
            <Label x={556} y={152 + i * 17} size={9} tone={3} mono={false}>{(i === 0 || i === 2) ? `${(parseFloat(a.replace(/,/g, '')) / parseFloat(b.replace(/,/g, ''))).toFixed(0)}×` : ''}</Label>
          </g>
        ))}

        <line x1={BX} x2={624} y1={226} y2={226} className="v-rule faint" />
        <Label x={BX} y={242} tone={3} size={10}>{`Memory at ${BATCH} concurrent conversations of ${fmt.int(T)} tokens`}</Label>
        <rect x={BX} y={250} width={BW} height={18} rx={2} fill="none" stroke="var(--rule-2)" />
        <Span x={BX} y={250} w={BW * Math.min(1, w / gpu.mem)} h={18} kind="memory" label={`weights ${(w / 1e9).toFixed(1)} GB`} />
        <Span x={BX + BW * Math.min(1, w / gpu.mem)} y={250} w={BW * Math.min(1, (BATCH * T * per) / gpu.mem)} h={18} kind="memory" dim label={`cache ${gb(BATCH * T * per)}`} labelTone="light" />
        <Fade show={w + BATCH * T * per > gpu.mem}>
          <rect x={BX} y={247} width={BW} height={24} rx={3} fill="url(#tps-hatch-red)" stroke="var(--error)" />
          <text x={320} y={263} textAnchor="middle" className="v-label bold" style={{ fill: 'var(--error)', fontSize: 11 }}>
            {`${BATCH} conversations need ${((w + BATCH * T * per) / 1e9).toFixed(0)} GB of ${(gpu.mem / 1e9).toFixed(0)}`}
          </text>
        </Fade>
        <Label x={BX} y={286} tone={2} size={10.5} mono={false}>
          {`So this card holds ${fmt.int(seqs)} conversation${seqs === 1 ? '' : 's'} at ${fmt.int(T)} tokens — ${fmt.si(Math.floor(avail / per))} tokens of context in all.`}
        </Label>
        <Label x={BX} y={302} tone={3} size={10} mono={false}>
          {`Weights are the fixed bar; the cache scales with your traffic. That is why 70B needs two cards and 8B needs one.`}
        </Label>
        <Label x={BX} y={322} tone={3} size={10} mono={false}>
          {`Ask of any serving plan, in order: does it fit, is it fast, what does it cost per million tokens?`}
        </Label>
        <Label x={BX} y={342} size={10} tone={2} mono={false} kind="error">
          {`The lab notebook measures all of this on a real T4, and asks you to check the formula against the memory manager.`}
        </Label>
      </svg>
    </Stage>
  );
}
