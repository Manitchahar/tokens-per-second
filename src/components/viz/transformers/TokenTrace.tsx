import { useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Arrow, Fade, fmt } from '../lib/svg';
import { arch, modelIds, perLayerParams, headParams, embedParams, kvBytesPerToken, type ModelId } from './modelShapes';
import { GPUS } from '../lib/perfModel';

// One prompt, tokenised the way a Llama-3-family BPE would. The IDs differ slightly
// per model; the counts are what matter, and they land within a token of each other.
const PIECES = ['Expl', 'ain', ' the', ' K', 'V', ' cache', ' in', ' two', ' sent', 'ences', '.'];
const CHARS = 'Explain the KV cache in two sentences.'.length;

const STEPS = [
  'The prompt becomes tokens',
  'Layer 0: the embedding lookup',
  'Layer by layer',
  'The KV cache grows',
  'The output head',
  'The bill for one step',
  'Will it fit?',
];
const CAPTIONS = [
  '38 characters become 11 tokens — 3.5 characters each, close to the English average. Every downstream number in this lab is counted in these eleven units.',
  'One row of the embedding table: 4,096 numbers, 8 KB read. This is the only stage where the model touches the 525 M-parameter table, and it touches exactly one row.',
  'Every layer reads all of its weights — 436.2 MB for Llama-3.1-8B — and none of it is remembered. Drag the layer slider: the running total climbs by the same amount every time.',
  'The only structure that grows is the KV cache: one entry per token, per layer. Linear in context, invisible in the parameter count, and the reason long context costs memory.',
  'A final norm and one 4,096 × 128,256 matrix multiply. 525 M more parameters, fully read — 1.05 GB, about 6.8% of the step.',
  'Add it up: weights, output head, KV cache. That total divided by the bandwidth you measured in Module 02 is the floor on how fast one token can arrive.',
  'Weights plus KV cache against the memory of a real GPU. Pick a card and see how many users it holds at this context length.',
];
const GPU_IDS = ['t4', 'a100', 'h100', 'h200'];

export default function TokenTrace() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [mid, setMid] = useState<ModelId>('llama8b');
  const [gid, setGid] = useState('h100');
  const [phase, setPhase] = useState<'decode' | 'prefill'>('decode');
  const [layer, setLayer] = useState(1);
  const [ctx, setCtx] = useState(4096);

  const M = arch(mid);
  const G = GPUS.find((g) => g.id === gid)!;
  const L = M.layers;
  const ly = Math.min(L, Math.max(1, layer));
  const perLayer = perLayerParams(M);
  const kvPer = kvBytesPerToken(M);
  const wTotal = M.params * 2;
  const promptTokens = PIECES.length;
  const seq = phase === 'decode' ? ctx + 1 : promptTokens;
  const kvNow = seq * kvPer;
  const headW = headParams(M) * 2;

  // bytes actually read by one forward pass of this sequence
  const blockW = perLayer * 2;
  const bytesNow = blockW * ly + headW * (ly >= L ? 1 : 0) + kvNow;
  const totalStep = L * blockW + headW + kvNow;
  const used = wTotal + kvNow;
  const fits = used <= G.mem;
  const perUser = kvPer / G.bw;
  const cross = wTotal / kvPer; // context at which the KV cache equals all the weights

  const stackOn = st >= 2;
  const kvOn = st >= 3;
  const billOn = st >= 5;

  // ── the stack ──
  const CAPW = 42;
  const HEADW = 50;
  const x0 = 24;
  const barArea = 616 - x0 - CAPW - 10 - HEADW - 10;
  const bw = barArea / L;
  const barY = 112;
  const barH = 40;

  // token chips, laid out left to right
  const CHIPS = (() => {
    let x = 24;
    return PIECES.map((pc, i) => {
      const w = pc.length * 8.4 + 14;
      const at = { pc, i, x, w };
      x += w + 5;
      return at;
    });
  })();

  // ── the KV growth chart (log-log) ──
  const CHX0 = 116;
  const CHX1 = 616;
  const CHY0 = 218;
  const CHY1 = 286;
  const LX0 = 0;
  const LX1 = Math.log10(131072);
  const LY0 = 0; // 1 byte
  const LY1 = Math.log10(Math.max(wTotal * 2, 1e11));
  const px = (t: number) => CHX0 + ((Math.log10(Math.max(1, t)) - LX0) / (LX1 - LX0)) * (CHX1 - CHX0);
  const py = (b: number) => CHY1 - ((Math.log10(Math.max(1, b)) - LY0) / (LY1 - LY0)) * (CHY1 - CHY0);

  return (
    <Stage
      rootRef={ref}
      title="Trace one token end to end"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['memory', 'compute', 'token', ['error', 'does not fit']]}
      {...s}
      readout={
        <Readout
          items={[
            { label: 'weights re-read', value: `${fmt.gb(L * blockW + headW)}`, kind: 'memory' as const, note: `every step, ${phase} and prefill alike` },
            { label: `KV cache at ${fmt.int(seq)} tokens`, value: kvNow < 1e9 ? `${fmt.int(kvNow / 1024)} KiB` : fmt.gb(kvNow), kind: 'memory' as const, note: `${fmt.int(kvPer / 1024)} KiB per token` },
            { label: 'used on ' + G.name, value: fmt.gb(used), kind: fits ? 'memory' : 'error', note: fits ? `${(G.mem / used).toFixed(1)}× headroom` : 'will not fit' },
          ]}
        />
      }
      controls={
        <>
          <Segmented
            label="Model"
            value={mid}
            options={modelIds.map((id) => ({ value: id, label: arch(id).name.replace('Llama-3.1-', 'L3.1-').replace('Llama-3.2-', 'L3.2-').replace('Qwen2.5-', 'Q2.5-') }))}
            onChange={(v) => { setMid(v); setLayer(1); }}
          />
          <Segmented
            label="GPU"
            value={gid}
            options={GPU_IDS.map((id) => ({ value: id, label: GPUS.find((g) => g.id === id)!.name.split(' ')[0] }))}
            onChange={setGid}
          />
          <Segmented
            label="Phase"
            value={phase}
            options={[{ value: 'decode' as const, label: 'decode' }, { value: 'prefill' as const, label: 'prefill' }]}
            onChange={(v) => { setPhase(v); }}
          />
          <Slider label="Which layer are you inside" value={ly} min={1} max={L} step={1} onChange={setLayer} format={(v) => `layer ${v} of ${L}`} kind="compute" />
          <Slider label="Generated tokens so far" value={ctx} min={0} max={131072} log onChange={setCtx} format={(v) => `${fmt.int(v)}`} kind="memory" />
        </>
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="An interactive trace of one token: the prompt tokenised, a stack of layers with a moving highlight, a log-log chart of KV cache growth against context length, and the resulting memory bill.">
        {/* ── the prompt ── */}
        <Label x={24} y={20} tone={2} size={9.5}>the prompt, tokenised</Label>
        <Label x={616} y={20} anchor="end" tone={3} size={9}>{`vocabulary ${fmt.int(M.vocab)}`}</Label>
        {CHIPS.map((c) => (
          <g key={c.i}>
            <rect x={c.x} y={28} width={c.w} height={22} rx={3} fill="var(--token)" stroke="var(--token)" strokeWidth={0.9} className="v-cell" />
            <text x={c.x + c.w / 2} y={43} textAnchor="middle" className="v-label" style={{ fontSize: 9.5, fontWeight: 600, fill: '#0e1116' }}>{c.pc}</text>
            <text x={c.x + c.w / 2} y={60} textAnchor="middle" className="v-label t3" style={{ fontSize: 8 }}>{c.i}</text>
          </g>
        ))}
        <Label x={24} y={76} tone={3} size={9}>{`${CHARS} characters → ${promptTokens} tokens (${(CHARS / promptTokens).toFixed(1)} chars each). Each piece has an integer ID; the model sees only those.`}</Label>
        <line x1={24} y1={90} x2={616} y2={90} className="v-rule faint" />

        {/* ── the stack ── */}
        <Label x={24} y={106} tone={1} size={10} bold>the stack, and where you are</Label>
        <Fade show={st >= 1}>
          <g>
            <rect x={x0} y={barY} width={CAPW} height={barH} rx={2} fill="var(--token-dim)" stroke="var(--token)" strokeWidth={0.9} className="v-cell" />
            <text x={x0 + CAPW / 2} y={barY + 26} textAnchor="middle" className="v-label" style={{ fontSize: 9, fontWeight: 700, fill: 'var(--text)' }}>embed</text>
            <Arrow x1={x0 + CAPW} y1={barY + barH / 2} x2={x0 + CAPW + 8} y2={barY + barH / 2} kind="text" width={1} />
          </g>
        </Fade>
        <Fade show={stackOn}>
          <g>
            {Array.from({ length: L }, (_, i) => (
              <rect key={i} x={x0 + CAPW + 10 + i * bw} y={barY + (i === ly - 1 ? 0 : 8)} width={Math.max(1.5, bw - 2)} height={barH - (i === ly - 1 ? 0 : 16)} rx={1.5}
                fill={i === ly - 1 ? 'var(--token)' : 'var(--memory-dim)'} stroke={i === ly - 1 ? 'var(--token)' : 'var(--memory)'} strokeWidth={0.8} className="v-cell" />
            ))}
            <Arrow x1={x0 + CAPW + 10 + L * bw + 2} y1={barY + barH / 2} x2={x0 + CAPW + 10 + L * bw + 8} y2={barY + barH / 2} kind="text" width={1} />
            <rect x={x0 + CAPW + 10 + L * bw + 10} y={barY} width={HEADW} height={barH} rx={2}
              fill="var(--compute-dim)" stroke="var(--compute)" strokeWidth={0.9} className="v-cell" />
            <text x={x0 + CAPW + 10 + L * bw + 10 + HEADW / 2} y={barY + 26} textAnchor="middle" className="v-label" style={{ fontSize: 9, fontWeight: 700, fill: 'var(--text)' }}>head</text>
          </g>
        </Fade>
        <Label x={24} y={170} tone={3} size={9}>{`${L} layers × ${fmt.si(perLayer)} parameters = ${fmt.bytes(blockW)} each. Read in full, every step, and never remembered.`}</Label>
        <Fade show={stackOn}>
          <Label x={24} y={188} tone={1} size={9.5}>{`through layer ${ly}: ${fmt.gb(bytesNow)} moved so far this step`}</Label>
        </Fade>

        {/* ── the KV cache chart ── */}
        <Label x={24} y={208} tone={1} size={10} bold>{kvOn ? 'the KV cache, and the only thing that grows' : ''}</Label>
        <Fade show={kvOn}>
          <g>
            <line x1={CHX0} x2={CHX1} y1={CHY1} y2={CHY1} className="v-rule" />
            <line x1={CHX0} x2={CHX0} y1={CHY0 - 6} y2={CHY1} className="v-rule" />
            {[1, 1024, 131072].map((t) => (
              <g key={t}>
                <line x1={px(t)} x2={px(t)} y1={CHY1} y2={CHY1 + 4} className="v-rule" />
                <text x={px(t)} y={CHY1 + 14} textAnchor="middle" className="v-label t3" style={{ fontSize: 8 }}>{fmt.si(t)}</text>
              </g>
            ))}
            <text x={(CHX0 + CHX1) / 2} y={CHY1 + 28} textAnchor="middle" className="v-label t3" style={{ fontSize: 8.5 }}>context length, tokens</text>
            <text x={CHX0 - 8} y={CHY0 + 2} textAnchor="end" className="v-label t3" style={{ fontSize: 8.5 }}>{fmt.gb(Math.pow(10, LY1))}</text>
            <text x={CHX0 - 8} y={CHY1} textAnchor="end" className="v-label t3" style={{ fontSize: 8.5 }}>1 B</text>
            <text x={CHX0 - 8} y={(CHY0 + CHY1) / 2 + 4} textAnchor="end" className="v-label t3" style={{ fontSize: 8.5 }}>bytes</text>

            {/* weights: a horizontal line */}
            <line x1={CHX0} x2={CHX1} y1={py(wTotal)} y2={py(wTotal)} stroke="var(--compute)" strokeWidth={1.4} strokeDasharray="6 4" />
            <text x={CHX1} y={py(wTotal) - 6} textAnchor="end" className="v-label" style={{ fontSize: 8.5, fill: 'var(--compute)' }}>{`all the weights, ${fmt.gb(wTotal)}`}</text>

            {/* the KV curve */}
            <path d={`M${px(1)},${py(kvPer)} L${px(131072)},${py(kvPer * 131072)}`} fill="none" stroke="var(--memory)" strokeWidth={1.8} />
            <line x1={px(seq)} x2={px(seq)} y1={CHY0 - 6} y2={CHY1} stroke="var(--token)" strokeWidth={1.2} strokeDasharray="3 3" />
            <circle cx={px(seq)} cy={py(kvNow)} r={3.5} fill="var(--token)" stroke="var(--ground-2)" strokeWidth={1.5} />
            <text x={Math.min(CHX1 - 60, px(seq) + 6)} y={Math.max(CHY0 + 8, py(kvNow) - 8)} className="v-label" style={{ fontSize: 8.5, fill: 'var(--token)', fontWeight: 700 }}>
              {kvNow < 1e9 ? `${fmt.int(kvNow / 1024)} KiB` : fmt.gb(kvNow)}
            </text>
            {cross < 131072 && (
              <>
                <circle cx={px(cross)} cy={py(wTotal)} r={3} fill="none" stroke="var(--error)" strokeWidth={1.2} />
                <text x={px(cross) - 6} y={py(wTotal) + 14} textAnchor="end" className="v-label" style={{ fontSize: 8.5, fill: 'var(--error)' }}>{`KV = weights at ${fmt.si(cross)} tokens`}</text>
              </>
            )}
          </g>
        </Fade>

        {/* ── the bill ── */}
        <Fade show={billOn}>
          <g>
            <Label x={24} y={338} tone={1} size={9.5}>{`one ${phase} step: ${fmt.gb(totalStep)} read — ${fmt.ms(totalStep / (G.bw * 0.8))}, ${(G.bw * 0.8 / totalStep).toFixed(0)} tok/s floor`}</Label>
            <Label x={24} y={354} tone={2} size={9.5}>{`on ${G.name}: ${fmt.gb(used)} of ${fmt.gb(G.mem)} used — ${fits ? `${(G.mem / used).toFixed(1)}× headroom, ~${Math.floor(G.mem / used)} sequences here` : 'does not fit; quantize or shorten'}`}</Label>
          </g>
        </Fade>
        <Fade show={st === 6}>
          <Label x={24} y={370} tone={3} size={9}>{'Switch models, then switch GPUs: the KV line moves, the weight line does not.'}</Label>
        </Fade>
      </svg>
    </Stage>
  );
}
