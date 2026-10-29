import { useRef, useState } from 'react';
import { Stage, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade, fmt } from '../lib/svg';
import { arch, type ModelId } from './modelShapes';

type Mode = 'mha' | 'mqa' | 'gqa';
const MODES: { id: Mode; name: string; qh: number; kvh: number; who: string }[] = [
  { id: 'mha', name: 'MHA', qh: 8, kvh: 8, who: 'Llama-2-7B, every LLaMA-1, GPT-2, GPT-3' },
  { id: 'mqa', name: 'MQA', qh: 8, kvh: 1, who: 'PaLM, Falcon-7B, StarCoder, Gemma-2B' },
  { id: 'gqa', name: 'GQA', qh: 8, kvh: 2, who: 'Llama-3.1, Qwen2, Mistral' },
];
const D_HEAD = 128;

// Published configs, straight from each model card.
const TABLE: ModelId[] = ['llama8b', 'llama70b', 'qwen32b', 'llama3b', 'qwen05'];

// One short note per step, along the bottom of the stage.
const NOTES: string[] = [
  '',
  'one 4,096-vector becomes eight 128-vectors, each with its own q_proj, k_proj, v_proj',
  'they run in parallel — one [8 × T × 128] · [128 × T] matmul, not eight in a row',
  'now the interesting part: K and V are a shared resource; nothing requires n_kv = n_q',
  'MQA: one KV head serves all eight queries, so the cache is 1/32 of full multi-head',
  'GQA: eight KV heads shared four ways — almost all of the saving, almost no quality loss',
  'bytes, not FLOPs: the arithmetic is identical; what shrinks is storage, reads, writes',
];

const STEPS = [
  'One head is not enough',
  'Split the hidden state into heads',
  'Many heads in parallel',
  'Then K and V do not have to keep up',
  'MQA: one KV head for everything',
  'GQA: groups in between',
  'What GQA costs and saves',
  'The numbers on real models',
];
const CAPTIONS = [
  'A single attention head returns one average — one number of attention per token. It cannot at once copy the previous token and resolve a pronoun two clauses back.',
  'Llama-3.1-8B hides 4,096 numbers and cuts them into 32 heads of 128. Every head sees the same vector, and the outputs are concatenated back to 4,096.',
  'Thirty-two heads are a batched matrix multiply, so they cost nearly the same wall-clock time as one. Only the weights are extra, and attention is 16.7% of the model.',
  'But every head needs keys and values to look at, and those must be stored for every earlier token. That is where the memory goes, and it is where the design freedom is.',
  'Multi-query attention: one shared key and value head for all 32 queries. The KV cache collapses to 1/32 of full multi-head attention, and quality falls a little.',
  'Grouped-query attention, the compromise almost every current open model uses: 32 query heads sharing 8 key/value heads, four to one. Between MHA and MQA.',
  'GQA changes bytes, not FLOPs. Every query head still scores every key, so the arithmetic is identical — but four times fewer key and value bytes are read and stored.',
  'KV cache per token = 2 × layers × KV heads × 128 × 2. Qwen2.5-32B has 64 layers but 8 KV heads, so it holds less than Llama-3.1-70B from half the parameters.',
];

const X0 = 120;
const COLW = 62;
const HW = COLW - 10;
const KV_Y = 68;
const Q_Y = 210;
const O_Y = 254;

export default function HeadsViz() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [mode, setMode] = useState<Mode>('gqa');
  const m = MODES.find((x) => x.id === mode)!;
  const group = m.qh / m.kvh;
  const shared = m.qh > m.kvh;

  const rows = TABLE.map((id) => {
    const a = arch(id);
    return {
      name: a.name,
      layers: a.layers,
      heads: a.heads,
      kv: a.kvHeads,
      perTok: 2 * a.layers * a.kvHeads * a.headDim * 2,
      at4k: 4096 * 2 * a.layers * a.kvHeads * a.headDim * 2,
    };
  });
  const maxAt4k = Math.max(...rows.map((r) => r.at4k));
  const gib = (b: number) => (b >= 1024 ** 3 ? `${(b / 1024 ** 3).toFixed(2)} GiB` : `${Math.round(b / 1024 ** 2)} MiB`);

  const splitOn = st >= 1 && st < 7;
  const shareOn = st >= 3 && st < 7;
  const note = st >= 1 && st < 7 ? NOTES[st] : '';

  const kvX = (j: number) => X0 + (j * m.qh) / m.kvh * COLW;
  const kvW = (m.qh / m.kvh) * COLW - 10;

  return (
    <Stage
      rootRef={ref}
      title="How many key/value heads, and why it matters more than you think"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', 'memory', ['comm', 'shared key/value head']]}
      {...s}
      readout={
        <Readout
          items={[
            { label: 'query heads', value: `${m.qh}`, kind: 'compute' as const, note: 'each with its own q_proj' },
            { label: 'key/value heads', value: `${m.kvh}`, kind: 'memory' as const, note: shared ? `${group} queries share each` : 'one per query head' },
            { label: 'KV bytes / token / layer', value: fmt.int(m.kvh * D_HEAD * 4), kind: 'memory' as const, note: 'this toy, head_dim 128' },
            { label: 'vs. full multi-head', value: `${(8 / m.kvh).toFixed(0)}×`, kind: 'text' as const, note: m.who },
          ]}
        />
      }
      controls={
        st >= 3 ? (
          <Segmented
            label="Attention style"
            value={mode}
            options={MODES.map((x) => ({ value: x.id, label: `${x.name} · ${x.kvh} KV head${x.kvh > 1 ? 's' : ''}` }))}
            onChange={setMode}
          />
        ) : undefined
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="Eight query heads each projecting its own query, grouped onto one, two or eight shared key-value heads; below, a table of KV cache bytes per token for five real models.">
        {/* headers */}
        <Fade show={splitOn}>
          <Label x={24} y={22} tone={2} size={9.5}>{'hidden 4,096 → viewed as 8 heads × 128'}</Label>
        </Fade>
        <Fade show={shareOn}>
          <Label x={24} y={46} tone={2} size={9.5}>{`${m.kvh} key/value head${m.kvh > 1 ? 's' : ''} × 128`}</Label>
          <Label x={24} y={58} tone={3} size={9}>what the cache stores</Label>
        </Fade>

        {/* note for this step */}
        {note && (
          <text x={24} y={370} className="v-label t3" style={{ fontSize: 9 }}>{note}</text>
        )}

        {/* key/value heads */}
        <Fade show={shareOn}>
          <g>
            {Array.from({ length: m.kvh }, (_, j) => (
              <g key={j}>
                <rect x={kvX(j)} y={KV_Y} width={kvW} height={22} rx={2} fill="var(--memory)" stroke="var(--memory)" strokeWidth={0.8} className="v-cell" />
                <text x={kvX(j) + kvW / 2} y={KV_Y + 15} textAnchor="middle" className="v-label" style={{ fontSize: 9, fontWeight: 700, fill: '#0e1116' }}>{`kv${j}`}</text>
              </g>
            ))}
            {/* fan-out: every query head reads the KV head of its group */}
            {Array.from({ length: m.qh }, (_, i) => {
              const j = Math.floor(i / group);
              const xq = X0 + i * COLW + HW / 2;
              const xk = kvX(j) + kvW / 2;
              return (
                <path key={i} d={`M${xq},${KV_Y + 26} C ${xq},150 ${xk},170 ${xk},${Q_Y - 4}`} fill="none"
                  stroke={shared ? 'var(--comm)' : 'var(--memory)'} strokeWidth={shared ? 1.1 : 0.8}
                  className="v-fade" style={{ opacity: 0.85 }} />
              );
            })}
          </g>
        </Fade>

        {/* query heads */}
        <Fade show={splitOn}>
          <g>
            <Label x={24} y={190} tone={2} size={9.5}>{`${m.qh} query heads`}</Label>
            <Label x={24} y={202} tone={3} size={9}>× 128 wide each</Label>
            {Array.from({ length: m.qh }, (_, i) => (
              <g key={i}>
                <rect x={X0 + i * COLW} y={Q_Y} width={HW} height={22} rx={2} fill="var(--compute)" stroke="var(--compute)" strokeWidth={0.8} className="v-cell" />
                <text x={X0 + i * COLW + HW / 2} y={Q_Y + 15} textAnchor="middle" className="v-label" style={{ fontSize: 9, fontWeight: 700, fill: '#0e1116' }}>{`q${i}`}</text>
              </g>
            ))}
            {Array.from({ length: m.qh }, (_, i) => (
              <rect key={i} x={X0 + i * COLW} y={O_Y} width={HW} height={18} rx={2} fill="var(--compute-dim)" stroke="var(--compute)" strokeWidth={0.8} className="v-cell" />
            ))}
            <Label x={24} y={O_Y + 12} tone={2} size={9.5}>{'their own'}</Label>
            <Label x={24} y={O_Y + 24} tone={3} size={9}>averages</Label>
            <Label x={24} y={292} tone={3} size={9}>{'concatenate the 8 averages back to 4,096, then project with W_O'}</Label>
            {shareOn && (
              <Label x={24} y={312} tone={1} size={9.5} kind="comm">
                {mode === 'mqa' ? 'all 8 queries read one single k and v' : mode === 'gqa' ? `a group of ${group} queries shares one k and v` : 'no sharing: every query head owns its k and v'}
              </Label>
            )}
            {shareOn && (
              <Label x={24} y={332} tone={2} size={9}>{`KV bytes / token / layer = ${m.kvh} × 128 × 2 × 2 = ${fmt.int(m.kvh * D_HEAD * 4)}`}</Label>
            )}
            {shareOn && (
              <Label x={24} y={350} tone={2} size={9}>{`that is ${(8 / m.kvh).toFixed(0)}× smaller than full multi-head attention`}</Label>
            )}
          </g>
        </Fade>

        {/* the real-model table */}
        <Fade show={st >= 7}>
          <g>
            <Label x={24} y={26} tone={1} size={10} bold>KV cache in BF16 — bars show the total at 4,096 tokens</Label>
            {rows.map((r, i) => (
              <g key={r.name}>
                <text x={24} y={62 + i * 30} className="v-label t1" style={{ fontSize: 9.5 }}>{r.name}</text>
                <text x={152} y={62 + i * 30} className="v-label t3" style={{ fontSize: 8.5 }}>{`${r.layers}L · ${r.heads}q/${r.kv}kv`}</text>
                <rect x={252} y={53 + i * 30} width={(r.at4k / maxAt4k) * 186} height={11} rx={2}
                  fill="var(--memory)" className="v-cell" />
                <text x={446} y={62 + i * 30} className="v-label t1" style={{ fontSize: 9 }}>
                  {`${fmt.int(r.perTok / 1024)} KiB/tok · ${gib(r.at4k)}`}
                </text>
              </g>
            ))}
            <line x1={24} y1={196} x2={616} y2={196} className="v-rule faint" />
            <Label x={24} y={216} tone={3} size={9}>{'A 32B model with fewer KV heads holds less than a 70B one.'}</Label>
            <Label x={24} y={232} tone={3} size={9}>{'KV bytes = 2 × layers × kv heads × head_dim × 2. Nothing else enters.'}</Label>
            <Label x={24} y={254} tone={2} size={9}>Qwen2.5-32B: 64 layers but 8 KV heads → 256 KiB/token, against the 70B model’s 320.</Label>
          </g>
        </Fade>

        {st === 0 && (
          <g>
            <Label x={24} y={44} tone={1} size={11} bold>why not just one head?</Label>
            <Label x={24} y={68} tone={2} size={9.5}>One head returns one weighted average of the value</Label>
            <Label x={24} y={84} tone={2} size={9.5}>vectors — one number of attention per token. It cannot</Label>
            <Label x={24} y={100} tone={2} size={9.5}>at once copy the previous token and resolve a pronoun</Label>
            <Label x={24} y={116} tone={2} size={9.5}>two clauses back, because those need different averages.</Label>
            <Label x={24} y={146} tone={2} size={9.5}>Split the hidden state into heads and each head is free</Label>
            <Label x={24} y={162} tone={2} size={9.5}>to learn its own average, in parallel, in one matmul.</Label>
          </g>
        )}
      </svg>
    </Stage>
  );
}
