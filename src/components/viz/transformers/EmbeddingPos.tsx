import { useRef, useState } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Arrow, Fade, Matrix, fmt } from '../lib/svg';
import { arch } from './modelShapes';

// Llama-3.1-8B's published dimensions.
const M = arch('llama8b');
const D = M.hidden;
const D_HEAD = M.headDim;
const PAIRS = D_HEAD / 2;

// RoPE base θ, as published: Llama-2 10,000 · Llama-3.1 500,000 · Qwen2.5 1,000,000.
const BASES = [
  { v: 10000, name: '10k', who: 'Llama-2' },
  { v: 500000, name: '500k', who: 'Llama-3.1' },
  { v: 1000000, name: '1M', who: 'Qwen2.5' },
];
/** Angular step per token for pair i of d_head/2 pairs: θᵢ = base^(-2i/d_head). */
const freq = (base: number, i: number) => Math.pow(base, (-2 * i) / D_HEAD);
/** Distance in tokens for one full turn of pair i. */
const wavelength = (base: number, i: number) => (2 * Math.PI) / freq(base, i);

// The sentence from 03.3, as IDs into a toy vocabulary.
const SENT = [
  { id: 15339, piece: 'the' },
  { id: 1917, piece: 'transform' },
  { id: 4734, piece: 'er' },
  { id: 11203, piece: 'reads' },
  { id: 128009, piece: 'tokens' },
];

const DELTAS = [1, 2, 4, 8, 16];

const STEPS = [
  'An ID means nothing by itself',
  'An embedding table turns an ID into a vector',
  'From here on, the token is that vector',
  'But the vector does not know its position',
  'Three ways to add position',
  'RoPE: rotate the query and key',
  'Only the difference survives',
];
const CAPTIONS = [
  'Five tokens, five integers. Nothing in 15339 says "the" — the model has no idea what these numbers mean.',
  'Each ID is a row number in a table of 128,256 rows of 4,096 numbers. Looking up 15339 reads one row. That row is the meaning.',
  'Everything downstream sees only vectors. The ID is gone. This is the "hidden state": 4,096 numbers that now carry everything the model knows about the token.',
  'The embedding of "the" at position 0 and at position 500 is the same row of the same table. Attention cannot tell them apart, so the model cannot know word order.',
  'Add a position signal somehow. Learned absolute positions (GPT-2) need a table and cap the context. Sinusoidal positions need no table. RoPE needs no table and extends further.',
  'RoPE does not add a position vector. It rotates the query and key of each attention head by an angle that grows with position. Different pairs rotate at different speeds: 64 pairs, from one full turn per token down to almost none.',
  'Because rotations compose, the dot product of a rotated query and rotated key depends only on the distance m − n. Move both tokens forward and the score does not change. That is the whole trick.',
];

const CX = 104;
const CY = 252;
const R = 50;
const pt = (a: number, r = R) => [CX + r * Math.cos(a), CY - r * Math.sin(a)] as const;

export default function EmbeddingPos() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [posQ, setPosQ] = useState(3);
  const [posK, setPosK] = useState(1);
  const [baseId, setBaseId] = useState(1);
  const base = BASES[baseId].v;

  const delta = posQ - posK;
  // the demo pair: index 8 of 64, a mid-speed frequency
  const theta = freq(base, 8);
  const aq = posQ * theta;
  const ak = posK * theta;
  const score = Math.cos(delta * theta);
  const bars = DELTAS.map((d) => ({ d, v: Math.cos(d * theta) }));

  const arcFrom = aq;
  const arcTo = aq + Math.max(-Math.PI, Math.min(Math.PI, ak - aq));
  const [ax0, ay0] = pt(arcFrom, 36);
  const [ax1, ay1] = pt(arcTo, 36);
  const arcLarge = Math.abs(arcTo - arcFrom) > Math.PI ? 1 : 0;
  const arcSweep = arcTo > arcFrom ? 0 : 1;

  const [gx] = pt(0, R);
  const [qx0, qy0] = pt(aq);
  const [kx0, ky0] = pt(ak);

  const tblOn = st >= 1;
  const vecOn = st >= 2;
  const ropeOn = st >= 5;

  // one extracted row, 8 of 4,096 numbers
  const ROW = [0.11, -0.62, 0.94, -1.24, 0.37, -0.51];

  return (
    <Stage
      rootRef={ref}
      title="From an integer to a vector, and from a vector to a position"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', ['comm', 'position']]}
      {...s}
      readout={
        <Readout
          items={[
            { label: 'embedding row', value: vecOn ? `${D.toLocaleString('en-US')} numbers` : 'waiting', kind: 'memory' as const, note: `${D * 2} bytes read per token` },
            { label: 'distance Δ = m − n', value: `${delta}`, kind: 'comm' as const, note: 'the only position that survives' },
            { label: 'q · k after RoPE', value: ropeOn ? score.toFixed(3) : '—', kind: 'compute' as const, note: ropeOn ? `cos(${delta} × ${theta.toFixed(3)})` : 'not computed' },
          ]}
        />
      }
      controls={
        st >= 5 ? (
          <>
            <Slider label="Query position m" value={posQ} min={0} max={16} step={1} onChange={setPosQ} format={(v) => `${v}`} kind="compute" />
            <Slider label="Key position n" value={posK} min={0} max={16} step={1} onChange={setPosK} format={(v) => `${v}`} kind="memory" />
            <Segmented
              label="RoPE base"
              value={baseId}
              options={BASES.map((b, i) => ({ value: i, label: `${b.name} · ${b.who}` }))}
              onChange={setBaseId}
            />
          </>
        ) : (
          <Slider label="RoPE base" value={baseId} min={0} max={2} step={1} onChange={setBaseId} format={(v) => BASES[v].name} kind="comm" />
        )
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="Token IDs selecting rows of an embedding table to produce vectors, beside a circle diagram showing RoPE rotating query and key vectors by an angle that grows with position.">
        {/* ── the embedding lookup ── */}
        <Label x={24} y={28} tone={2} size={10.5}>{`Llama-3.1-8B · hidden size ${D.toLocaleString('en-US')}`}</Label>
        {SENT.map((s2, i) => (
          <g key={s2.id}>
            <text x={24} y={52 + i * 19} className="v-label t3" style={{ fontSize: 9 }}>{s2.id}</text>
            <text x={86} y={52 + i * 19} className="v-label t1" style={{ fontSize: 10 }}>{s2.piece}</text>
          </g>
        ))}
        <Label x={160} y={52} tone={3} size={9}>IDs</Label>
        <Arrow x1={148} y1={92} x2={186} y2={92} kind="memory" />
        <Label x={167} y={84} tone={3} size={8.5}>look up</Label>

        <Fade show={tblOn}>
          <Label x={200} y={36} tone={2} size={10}>{`embed table ${M.vocab.toLocaleString('en-US')} × ${D.toLocaleString('en-US')}`}</Label>
          <Matrix x={200} y={44} rows={8} cols={8} cell={15} gap={2}
            fill={(r) => (r === 2 ? 'memory' : 'empty')} />
          <rect x={198} y={44 + 2 * 17 - 2} width={8 * 17 - 2 + 4} height={19} rx={2} fill="none" stroke="var(--token)" strokeWidth={1.4} />
          <Arrow x1={344} y1={92} x2={374} y2={92} kind="token" />
        </Fade>

        <Fade show={vecOn}>
          <Label x={388} y={88} tone={3} size={9}>{'row 4734 →'}</Label>
          {ROW.map((v, i) => (
            <g key={i}>
              <rect x={388 + i * 36} y={94} width={34} height={17} rx={2}
                fill={v >= 0 ? 'var(--token)' : 'none'} stroke="var(--token)" strokeWidth={1}
                strokeDasharray={v >= 0 ? undefined : '3 3'} className="v-cell" />
              <text x={405 + i * 36} y={106} textAnchor="middle" className="v-label"
                style={{ fontSize: 9.5, fill: v >= 0 ? '#0e1116' : 'var(--text-3)' }}>{v.toFixed(2)}</text>
            </g>
          ))}
          <Label x={388} y={128} tone={3} size={9}>{`one of ${D.toLocaleString('en-US')} numbers · ${(D * 2 / 1024).toFixed(0)} KB read`}</Label>
          <Label x={388} y={150} tone={2} size={10}>this vector is the token, from here on.</Label>
          <Label x={388} y={168} tone={3} size={9}>{'the same row at position 0 and at 9,999.'}</Label>
        </Fade>

        <line x1={20} y1={182} x2={620} y2={182} className="v-rule faint" />

        {/* ── positions ── */}
        {st < 5 && (
          <g className="v-fade" style={{ opacity: 0.25 }}>
            <text x={388} y={214} className="v-label t2" style={{ fontSize: 10.5 }}>three ways to add position</text>
            {[
              { n: 'Learned absolute', d: 'a position table you must grow to extend', k: 'compute' },
              { n: 'Sinusoidal', d: 'fixed sines and cosines of the position', k: 'compute' },
              { n: 'RoPE', d: 'rotate q and k by an angle per position', k: 'comm' },
            ].map((r, i) => (
              <g key={r.n}>
                <rect x={388} y={228 + i * 30} width={5} height={22} rx={2} fill={r.k === 'comm' ? 'var(--comm)' : 'var(--compute)'} />
                <text x={400} y={240 + i * 30} className="v-label t1" style={{ fontSize: 10, fontWeight: 600 }}>{r.n}</text>
                <text x={400} y={253 + i * 30} className="v-label t3" style={{ fontSize: 8 }}>{r.d}</text>
              </g>
            ))}
            <text x={24} y={220} className="v-label t1" style={{ fontSize: 11, fontWeight: 600 }}>the problem: “dog bites man” and “man bites dog”</text>
            <text x={24} y={238} className="v-label t3" style={{ fontSize: 9.5 }}>have exactly the same tokens. Without positions the model</text>
            <text x={24} y={252} className="v-label t3" style={{ fontSize: 9.5 }}>cannot tell them apart at all — and neither can you.</text>
          </g>
        )}

        <Fade show={ropeOn}>
          <Label x={300} y={200} tone={1} size={10.5} bold>{'score = cos( (m − n) · θ )'}</Label>
          <Label x={300} y={216} tone={3} size={9}>{'θ = base^(-2i/d_head) for pair i — q·k sees only the gap'}</Label>
          <Label x={300} y={236} tone={2} size={9.5}>{`${BASES[baseId].who}: ${DELTAS.filter((d) => Math.cos(d * theta) > 0.3).length} of 5 distances stay near zero`}</Label>

          {/* circle */}
          <circle cx={CX} cy={CY} r={R} fill="none" stroke="var(--rule-2)" strokeWidth={1} />
          <line x1={CX - R - 8} x2={CX + R + 8} y1={CY} y2={CY} className="v-rule faint" />
          <line x1={CX} x2={CX} y1={CY - R - 8} y2={CY + R + 8} className="v-rule faint" />
          <line x1={CX} y1={CY} x2={gx} y2={CY} stroke="var(--text-3)" strokeWidth={1.4} markerEnd="url(#tps-arrow)" />
          <line x1={CX} y1={CY} x2={qx0} y2={qy0} stroke="var(--compute)" strokeWidth={2.2} markerEnd="url(#tps-arrow)" />
          <line x1={CX} y1={CY} x2={kx0} y2={ky0} stroke="var(--memory)" strokeWidth={2.2} markerEnd="url(#tps-arrow)" />
          <path d={`M${ax0},${ay0} A36,36 0 ${arcLarge} ${arcSweep} ${ax1},${ay1}`} fill="none" stroke="var(--comm)" strokeWidth={1.3} strokeDasharray="3 3" />
          <Label x={CX} y={CY - 18} anchor="middle" size={9} kind="comm">{`Δθ = ${(delta * theta).toFixed(2)}`}</Label>
          {[
            { c: 'var(--text-3)', t: 'before RoPE' },
            { c: 'var(--compute)', t: `query, rotated by ${posQ}θ` },
            { c: 'var(--memory)', t: `key, rotated by ${posK}θ` },
          ].map((r, i) => (
            <g key={r.t}>
              <rect x={24} y={322 + i * 12} width={9} height={5} rx={1.5} fill={r.c} />
              <text x={39} y={328 + i * 12} className="v-label t3" style={{ fontSize: 8.5 }}>{r.t}</text>
            </g>
          ))}
          <Label x={24} y={368} tone={2} size={8.5}>{`${PAIRS} pairs: 1 turn/token → 1 turn per ${fmt.si(wavelength(base, PAIRS - 1))}`}</Label>

          {/* score by distance */}
          <line x1={300} x2={618} y1={282} y2={282} className="v-rule" />
          {bars.map((b, i) => {
            const bx = 300 + i * 64;
            const h = Math.abs(b.v) * 38;
            const up = b.v >= 0;
            const hot = delta === b.d;
            return (
              <g key={b.d}>
                <rect x={bx + 22} y={up ? 282 - h : 282} width={20} height={Math.max(1, h)} rx={2}
                  fill={up ? 'var(--compute)' : 'none'} stroke="var(--compute)" strokeWidth={1}
                  strokeDasharray={up ? undefined : '3 3'} className="v-cell"
                  style={{ opacity: hot ? 1 : 0.45 }} />
                <text x={bx + 32} y={334} textAnchor="middle" className="v-label" style={{ fontSize: 9, fill: hot ? 'var(--token)' : 'var(--text-3)', fontWeight: hot ? 700 : 400 }}>{`Δ=${b.d}`}</text>
                <text x={bx + 32} y={348} textAnchor="middle" className="v-label" style={{ fontSize: 9.5, fill: hot ? 'var(--text)' : 'var(--text-3)' }}>{b.v.toFixed(2)}</text>
              </g>
            );
          })}
          <Label x={300} y={368} tone={3} size={8.5}>{'slide both positions together: the score never moves'}</Label>
        </Fade>
      </svg>
    </Stage>
  );
}
