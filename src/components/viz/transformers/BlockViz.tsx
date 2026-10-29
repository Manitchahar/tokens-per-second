import { useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Arrow, Fade, fmt } from '../lib/svg';
import { arch, mlpParams, attnParams, normParams } from './modelShapes';

const M = arch('llama8b');
const L = M.layers;
const INTER = M.inter;

// A vector whose RMS is exactly 1, so RMSNorm's arithmetic is transparent.
const UNIT = [0.673, -1.57, 1.046, 0.224, -0.598, 1.644, 0.449, -0.822];

const STEPS = [
  'The residual stream',
  'Pre-norm: read a normalised copy',
  'RMSNorm, exactly',
  'Attention writes its result back',
  'Then the MLP writes back',
  'The MLP: where the weights are',
  'Why the stream keeps growing',
];
const CAPTIONS = [
  'Give every token one vector of 4,096 numbers and keep adding to it. For Llama-3.1-8B that is 1 + 2 × 32 = 65 separate contributions summed into a single running vector.',
  'Each sublayer reads a normalised copy of the stream, computes something, and adds it back. Normalising on the way *in* is called pre-norm, and it is why 32-layer models train at all.',
  'RMSNorm divides a vector by its root-mean-square, so the output always has RMS 1, then multiplies by a learned gain. Drag the slider: the shape of the output never changes, only the scale it removes.',
  'The attention sublayer from 03.5 and 03.6 produces 4,096 numbers, and those numbers are added straight into the stream. Nothing is subtracted, nothing is overwritten.',
  'The same again for the MLP. Two sublayers, two writes, one vector — and after 32 layers the stream has been written to 65 times.',
  'The MLP is three matrices: expand 4,096 → 14,336, multiply two copies element by element, project back to 4,096. That is 176.16 M parameters — 80.8% of a layer.',
  'Each sublayer adds to the stream and nothing ever rescales the sum, so its magnitude grows with depth. RMSNorm divides it back to 1 before every sublayer, which is exactly why that growth is survivable.',
];

const FLOWX = 48;
const FLOWW = 240;
const CXF = FLOWX + FLOWW / 2; // centre line of the flow

export default function BlockViz() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [rms, setRms] = useState(6);
  const [depth, setDepth] = useState(8);

  const mlpP = mlpParams(M);
  const attnP = attnParams(M);
  const blockP = mlpP + attnP + normParams(M);
  // illustrative: the stream's magnitude accumulates roughly linearly with depth
  const streamRms = (d: number) => 1.4 + 0.42 * d;

  const note = st >= 2;

  return (
    <Stage
      rootRef={ref}
      title="One transformer block: norm, sublayer, add, twice"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory']}
      {...s}
      readout={
        <Readout
          items={[
            { label: 'stream RMS in', value: `${rms.toFixed(1)}`, kind: 'text' as const, note: note ? 'removed by the norm' : 'before RMSNorm' },
            { label: 'stream RMS out', value: note ? '1.000' : '—', kind: 'compute' as const, note: 'always, by construction' },
            { label: 'writes per token', value: `${1 + 2 * depth}`, kind: 'memory' as const, note: `1 embedding + 2 × ${depth} blocks` },
            { label: 'MLP share of a block', value: `${((mlpP / blockP) * 100).toFixed(1)}%`, kind: 'compute' as const, note: `${fmt.si(mlpP)} of ${fmt.si(blockP)} params` },
          ]}
        />
      }
      controls={
        st >= 2 ? (
          <>
            <Slider label="RMS of the stream when it reaches the norm" value={rms} min={0.5} max={30} step={0.5} onChange={setRms} format={(v) => v.toFixed(1)} kind="memory" />
            {st >= 6 && <Slider label="How many blocks deep" value={depth} min={1} max={32} step={1} onChange={setDepth} format={(v) => `${v}`} kind="compute" />}
          </>
        ) : undefined
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="A transformer block drawn as a flow: the residual stream, an RMSNorm, attention, an add back to the stream, then an RMSNorm, the MLP and a second add.">
        {/* ── the flow ── */}
        <g>
          <rect x={FLOWX} y={30} width={FLOWW} height={22} rx={3} fill="var(--memory-dim)" stroke="var(--memory)" strokeWidth={1} className="v-cell" />
          <text x={CXF} y={45} textAnchor="middle" className="v-label" style={{ fontSize: 9.5, fill: 'var(--text)' }}>residual stream · 4,096 numbers · 8 KB</text>

          <Arrow x1={CXF} y1={52} x2={CXF} y2={68} kind="text" width={1.2} />
          <rect x={FLOWX} y={70} width={FLOWW} height={22} rx={3} fill="var(--compute)" stroke="var(--compute)" strokeWidth={1} className="v-cell" />
          <text x={CXF} y={85} textAnchor="middle" className="v-label" style={{ fontSize: 10, fontWeight: 700, fill: '#0e1116' }}>RMSNorm</text>
          <Arrow x1={CXF} y1={92} x2={CXF} y2={106} kind="compute" width={1.2} />

          <rect x={FLOWX} y={108} width={FLOWW} height={26} rx={3} fill="var(--compute-dim)" stroke="var(--compute)" strokeWidth={1} className="v-cell" />
          <text x={CXF} y={125} textAnchor="middle" className="v-label" style={{ fontSize: 10.5, fontWeight: 600 }}>Attention · 03.5 + 03.6</text>
          <Arrow x1={CXF} y1={134} x2={CXF} y2={148} kind="compute" width={1.2} />

          <circle cx={CXF} cy={156} r={11} fill="var(--memory)" stroke="var(--memory)" strokeWidth={1} />
          <text x={CXF} y={160} textAnchor="middle" className="v-label" style={{ fontSize: 13, fontWeight: 700, fill: '#0e1116' }}>+</text>
          <Arrow x1={CXF} y1={167} x2={CXF} y2={179} kind="memory" width={1.2} />

          <rect x={FLOWX} y={181} width={FLOWW} height={22} rx={3} fill="var(--memory-dim)" stroke="var(--memory)" strokeWidth={1} className="v-cell" />
          <text x={CXF} y={196} textAnchor="middle" className="v-label" style={{ fontSize: 9.5, fill: 'var(--text)' }}>same vector, one contribution larger</text>

          <Arrow x1={CXF} y1={203} x2={CXF} y2={217} kind="text" width={1.2} />
          <rect x={FLOWX} y={219} width={FLOWW} height={22} rx={3} fill="var(--compute)" stroke="var(--compute)" strokeWidth={1} className="v-cell" />
          <text x={CXF} y={234} textAnchor="middle" className="v-label" style={{ fontSize: 10, fontWeight: 700, fill: '#0e1116' }}>RMSNorm</text>
          <Arrow x1={CXF} y1={241} x2={CXF} y2={255} kind="compute" width={1.2} />

          <rect x={FLOWX} y={257} width={FLOWW} height={26} rx={3} fill="var(--compute-dim)" stroke="var(--compute)" strokeWidth={1} className="v-cell" />
          <text x={CXF} y={274} textAnchor="middle" className="v-label" style={{ fontSize: 10.5, fontWeight: 600 }}>MLP · the other half</text>
          <Arrow x1={CXF} y1={283} x2={CXF} y2={297} kind="compute" width={1.2} />

          <circle cx={CXF} cy={305} r={11} fill="var(--memory)" stroke="var(--memory)" strokeWidth={1} />
          <text x={CXF} y={309} textAnchor="middle" className="v-label" style={{ fontSize: 13, fontWeight: 700, fill: '#0e1116' }}>+</text>
          <Arrow x1={CXF} y1={316} x2={CXF} y2={328} kind="memory" width={1.2} />

          <rect x={FLOWX} y={330} width={FLOWW} height={22} rx={3} fill="var(--memory-dim)" stroke="var(--memory)" strokeWidth={1} className="v-cell" />
          <text x={CXF} y={345} textAnchor="middle" className="v-label" style={{ fontSize: 9.5, fill: 'var(--text)' }}>two contributions larger</text>
          <Label x={24} y={368} tone={3} size={8.5}>{`× ${L} layers → ${1 + 2 * L} contributions summed into one vector`}</Label>
        </g>

        <line x1={312} y1={22} x2={312} y2={368} className="v-rule faint" />

        {/* ── right panel ── */}
        {st === 0 && (
          <g>
            <Label x={330} y={30} tone={1} size={10.5} bold>the residual stream</Label>
            <Label x={330} y={54} tone={2} size={9.5}>one vector per token, 4,096 numbers,</Label>
            <Label x={330} y={70} tone={2} size={9.5}>and every sublayer adds to it.</Label>
            <Label x={330} y={96} tone={2} size={9.5}>{`Llama-3.1-8B: 1 + 2 × ${L} = ${1 + 2 * L} writes`}</Label>
            <Label x={330} y={112} tone={2} size={9.5}>into that one vector.</Label>
            <Label x={330} y={138} tone={2} size={9.5}>Adds never cancel. Whatever layer 3</Label>
            <Label x={330} y={154} tone={2} size={9.5}>learned is still in there when layer</Label>
            <Label x={330} y={170} tone={2} size={9.5}>31 runs.</Label>
            <Label x={330} y={204} tone={3} size={9}>Reads and writes: 8 KB in, 8 KB out,</Label>
            <Label x={330} y={220} tone={3} size={9}>twice per block. 512 KB per token for</Label>
            <Label x={330} y={236} tone={3} size={9}>the whole 32-layer model.</Label>
          </g>
        )}

        {st === 1 && (
          <g>
            <Label x={330} y={30} tone={1} size={10.5} bold>pre-norm, not post-norm</Label>
            <Label x={330} y={54} tone={2} size={9.5}>Post-norm (the original 2017</Label>
            <Label x={330} y={70} tone={2} size={9.5}>Transformer) normalised the stream</Label>
            <Label x={330} y={86} tone={2} size={9.5}>after adding, so every layer sat on</Label>
            <Label x={330} y={102} tone={2} size={9.5}>the compounding sum. Deep models</Label>
            <Label x={330} y={118} tone={2} size={9.5}>had to be trained very carefully.</Label>
            <Label x={330} y={150} tone={2} size={9.5}>Pre-norm normalises the copy that</Label>
            <Label x={330} y={166} tone={2} size={9.5}>goes *into* the sublayer and leaves</Label>
            <Label x={330} y={182} tone={2} size={9.5}>the stream itself untouched. The</Label>
            <Label x={330} y={198} tone={2} size={9.5}>path straight from input to output</Label>
            <Label x={330} y={214} tone={2} size={9.5}>is an unbroken identity — gradients</Label>
            <Label x={330} y={230} tone={2} size={9.5}>flow through it untouched.</Label>
            <Label x={330} y={262} tone={1} size={9.5}>Every model in this course is pre-norm.</Label>
            <Label x={330} y={284} tone={3} size={9}>and there is one final RMSNorm before the</Label>
            <Label x={330} y={300} tone={3} size={9}>output head — 65 norms in total.</Label>
          </g>
        )}

        {st === 2 && (
          <g>
            <Label x={330} y={30} tone={1} size={10.5} bold>RMSNorm, exactly</Label>
            <Label x={330} y={50} tone={3} size={9}>the stream as it arrives, RMS = r:</Label>
            {UNIT.map((v, i) => (
              <g key={i}>
                <rect x={330 + i * 34} y={58} width={30} height={18} rx={2}
                  fill={v >= 0 ? 'var(--memory-dim)' : 'none'} stroke="var(--memory)" strokeWidth={0.9}
                  strokeDasharray={v >= 0 ? undefined : '3 3'} className="v-cell" />
                <text x={345 + i * 34} y={71} textAnchor="middle" className="v-label" style={{ fontSize: 8.5, fill: 'var(--text)' }}>{(v * rms).toFixed(1)}</text>
              </g>
            ))}
            <Label x={330} y={94} tone={2} size={9}>{`mean(x²) = ${(rms * rms).toFixed(1)}   ·   √ = ${rms.toFixed(1)}   ·   divide by that`}</Label>
            <Label x={330} y={114} tone={3} size={9}>after RMSNorm, RMS = 1.000, and the shape is</Label>
            <Label x={330} y={128} tone={3} size={9}>the same whatever r was — try the slider:</Label>
            {UNIT.map((v, i) => (
              <g key={i}>
                <rect x={330 + i * 34} y={136} width={30} height={18} rx={2}
                  fill={v >= 0 ? 'var(--compute)' : 'none'} stroke="var(--compute)" strokeWidth={0.9}
                  strokeDasharray={v >= 0 ? undefined : '3 3'} className="v-cell" />
                <text x={345 + i * 34} y={149} textAnchor="middle" className="v-label" style={{ fontSize: 8.5, fontWeight: 600, fill: v >= 0 ? '#0e1116' : 'var(--text-3)' }}>{v.toFixed(2)}</text>
              </g>
            ))}
            <Label x={330} y={176} tone={2} size={9}>{'then × a learned gain g (4,096 numbers per norm)'}</Label>
            <Label x={330} y={198} tone={3} size={9}>{'no mean subtraction — that is LayerNorm —'}</Label>
            <Label x={330} y={214} tone={3} size={9}>{'no bias; ε = 1e-5 guards a zero vector.'}</Label>
            <Label x={330} y={242} tone={3} size={9}>{'reading 4,096 numbers, writing 4,096. The whole'}</Label>
            <Label x={330} y={258} tone={3} size={9}>{'norm is 16 KB of traffic and 2 × 4,096 FLOPs.'}</Label>
            <Label x={330} y={286} tone={1} size={9}>{`${fmt.si(33 * M.hidden)} learned gain parameters in total`}</Label>
            <Label x={330} y={308} tone={3} size={9}>{'out of 8.03 B — 0.0017%. The cheapest part of the'}</Label>
            <Label x={330} y={324} tone={3} size={9}>{'model, and the part that keeps the rest of it'}</Label>
            <Label x={330} y={340} tone={3} size={9}>{'numerically sane.'}</Label>
          </g>
        )}

        {st >= 3 && st <= 4 && (
          <g>
            <Label x={330} y={30} tone={1} size={10.5} bold>the + is a plain elementwise add</Label>
            <Label x={330} y={54} tone={2} size={9.5}>{'x = x + Attn(Norm(x))'}</Label>
            <Label x={330} y={84} tone={2} size={9.5}>No scaling, no gate, no mixing — just</Label>
            <Label x={330} y={100} tone={2} size={9.5}>4,096 additions. That is why a residual</Label>
            <Label x={330} y={116} tone={2} size={9.5}>connection is so cheap and so effective:</Label>
            <Label x={330} y={132} tone={2} size={9.5}>the sublayer can only add information,</Label>
            <Label x={330} y={148} tone={2} size={9.5}>never remove it.</Label>
            <Label x={330} y={182} tone={2} size={9.5}>{st === 3 ? 'Attention returns 4,096 numbers:' : 'The MLP also returns 4,096 numbers:'}</Label>
            <Label x={330} y={198} tone={2} size={9.5}>{st === 3 ? 'one per head, concatenated, then' : 'expand to 14,336, mix, project back.'}</Label>
            <Label x={330} y={214} tone={2} size={9.5}>{st === 3 ? 'projected back down by W_O.' : 'Same shape in, same shape out.'}</Label>
            <Label x={330} y={248} tone={1} size={9.5}>{'The shape is preserved: a block is 4,096 → 4,096.'}</Label>
            <Label x={330} y={272} tone={3} size={9}>Two norms, two sublayers, two adds, and the</Label>
            <Label x={330} y={288} tone={3} size={9}>stream is 8 KB bigger in information and</Label>
            <Label x={330} y={304} tone={3} size={9}>exactly the same size in bytes.</Label>
          </g>
        )}

        {st === 5 && (
          <g>
            <Label x={330} y={30} tone={1} size={10.5} bold>the MLP: three matrices</Label>
            {[
              { w: 30, n: 'x', k: 'text' as const },
              { w: 96, n: `gate  ${fmt.int(M.hidden)}→${fmt.int(INTER)}`, k: 'compute' as const },
              { w: 96, n: `up  ${fmt.int(M.hidden)}→${fmt.int(INTER)}`, k: 'compute' as const },
              { w: 62, n: 'SiLU(·)×', k: 'token' as const },
              { w: 96, n: `down  ${fmt.int(INTER)}→${fmt.int(M.hidden)}`, k: 'compute' as const },
              { w: 30, n: 'y', k: 'text' as const },
            ].map((b, i, arr) => {
              let x = 330;
              for (let j = 0; j < i; j++) x += arr[j].w + 16;
              return (
                <g key={b.n}>
                  {i > 0 && <Arrow x1={x - 16} y1={62} x2={x - 2} y2={62} kind="text" width={1.1} />}
                  <rect x={x} y={50} width={b.w} height={24} rx={2}
                    fill={b.k === 'text' ? 'var(--ground-3)' : `var(--${b.k === 'compute' ? 'compute' : 'token'}-dim)`}
                    stroke={b.k === 'text' ? 'var(--rule-2)' : `var(--${b.k === 'compute' ? 'compute' : 'token'})`}
                    strokeWidth={1} className="v-cell" />
                  <text x={x + b.w / 2} y={66} textAnchor="middle" className="v-label" style={{ fontSize: 8.5, fill: 'var(--text)' }}>{b.n}</text>
                </g>
              );
            })}
            <Label x={330} y={98} tone={2} size={9}>{'y = down( SiLU(gate(x)) ⊙ up(x) )   — SwiGLU'}</Label>
            <Label x={330} y={122} tone={2} size={9}>{`intermediate ${fmt.int(INTER)} = ${(INTER / M.hidden).toFixed(1)}× hidden`}</Label>
            <Label x={330} y={150} tone={2} size={9}>{'per block, Llama-3.1-8B:'}</Label>
            {[
              { n: 'MLP', v: mlpP, k: 'compute' as const },
              { n: 'Attention', v: attnP, k: 'compute' as const },
              { n: 'Norms', v: normParams(M), k: 'text' as const },
            ].map((r, i) => (
              <g key={r.n}>
                <text x={330} y={170 + i * 16} className="v-label t2" style={{ fontSize: 9 }}>{r.n}</text>
                <rect x={380} y={162 + i * 16} width={(r.v / blockP) * 160} height={9} rx={1.5}
                  fill={r.k === 'text' ? 'var(--rule)' : 'var(--compute)'}>
                </rect>
                <text x={620} y={170 + i * 16} textAnchor="end" className="v-label t1" style={{ fontSize: 9 }}>
                  {`${fmt.si(r.v)} · ${((r.v / blockP) * 100).toFixed(1)}%`}
                </text>
              </g>
            ))}
            <Label x={330} y={232} tone={2} size={9}>{'The original Transformer used two matrices at 4×.'}</Label>
            <Label x={330} y={248} tone={2} size={9}>{'SwiGLU needs three, so the break-even width'}</Label>
            <Label x={330} y={264} tone={2} size={9}>{'is about 8/3. Llama chose 3.5× — deliberately fatter.'}</Label>
            <Label x={330} y={296} tone={1} size={9}>{`${fmt.si(mlpP)} weights × 2 bytes = ${fmt.gb(mlpP * 2)} read per token per layer`}</Label>
            <Label x={330} y={316} tone={3} size={9}>That is 70.2% of the model's weights and the</Label>
            <Label x={330} y={332} tone={3} size={9}>bulk of the bytes every decode step must move.</Label>
          </g>
        )}

        {st === 6 && (
          <g>
            <Label x={330} y={30} tone={1} size={10.5} bold>the stream grows, the norm tames it</Label>
            {/* growth curve */}
            <line x1={346} y1={60} x2={346} y2={180} className="v-rule" />
            <line x1={346} y1={180} x2={612} y2={180} className="v-rule" />
            {[8, 16, 24, 32].map((d) => (
              <g key={d}>
                <line x1={346 + ((d - 1) / 31) * 262} x2={346 + ((d - 1) / 31) * 262} y1={180} y2={185} className="v-rule" />
                <text x={346 + ((d - 1) / 31) * 262} y={196} textAnchor="middle" className="v-label t3" style={{ fontSize: 8.5 }}>{d}</text>
              </g>
            ))}
            <text x={346} y={208} className="v-label t3" style={{ fontSize: 8.5 }}>blocks deep</text>
            {Array.from({ length: 32 }, (_, i) => {
              const d = i + 1;
              const v = streamRms(d);
              const h = (v / 16) * 118;
              const hot = d <= depth;
              return (
                <rect key={d} x={346 + ((d - 1) / 31) * 262} y={180 - h} width={6} height={h} rx={1}
                  fill={hot ? 'var(--memory)' : 'var(--rule)'} className="v-cell" />
              );
            })}
            <line x1={346 + ((depth - 1) / 31) * 262} x2={346 + ((depth - 1) / 31) * 262} y1={52} y2={180} stroke="var(--token)" strokeWidth={1.4} strokeDasharray="3 3" />
            <text x={346 + ((depth - 1) / 31) * 262 + 6} y={62} className="v-label" style={{ fontSize: 9, fill: 'var(--token)', fontWeight: 700 }}>
              {`RMS ≈ ${streamRms(depth).toFixed(1)} at block ${depth}`}
            </text>
            <Label x={330} y={228} tone={3} size={9}>{'every sublayer adds and nothing rescales the sum,'}</Label>
            <Label x={330} y={244} tone={3} size={9}>{'so the magnitude grows with depth. Values here'}</Label>
            <Label x={330} y={260} tone={3} size={9}>{'are illustrative; the shape is what matters.'}</Label>
            <Label x={330} y={288} tone={2} size={9}>{'RMSNorm divides it back to 1 before every sublayer,'}</Label>
            <Label x={330} y={304} tone={2} size={9}>{'which is exactly why that growth is survivable at'}</Label>
            <Label x={330} y={320} tone={2} size={9}>{'32 layers — and why the norm is load-bearing,'}</Label>
            <Label x={330} y={336} tone={2} size={9}>{'not decoration.'}</Label>
          </g>
        )}
      </svg>
    </Stage>
  );
}
