import { useEffect, useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Span, Label, Bar, Fade, Arrow, fmt } from '../lib/svg';

// Everything here is the arithmetic from the lesson, using the same
// Llama-3.1-8B shapes and H100 spec-sheet numbers as Module 05 and Module 10.
const SYS = 150; // system prompt + tool instructions, tokens
const CHUNK = 512; // tokens per retrieved chunk
const QUESTION = 30; // the user's question, tokens
const TEMPLATE = 20; // chat-template wrappers, tokens
const KB = 100_000; // a small knowledge base, tokens
const KV_PER_TOKEN = 131_072; // 2 × 32 layers × 8 kv heads × 128 dim × 2 bytes
const PARAMS = 8.03e9;
const H100_FLOPS = 989e12;
const MFU = 0.55;
const WEIGHTS_GB = 16.06;
const GPU_GB = 80;
const WORKSPACE_GB = 0.04 * GPU_GB;

const promptTokens = (chunks: number) => SYS + chunks * CHUNK + QUESTION + TEMPLATE;
const prefillMs = (tokens: number) => ((2 * PARAMS * tokens) / (H100_FLOPS * MFU)) * 1000;
const kvGB = (tokens: number) => (tokens * KV_PER_TOKEN) / 1e9;

const CAG_KV_GB = kvGB(KB);
const FREE_GB = (GPU_GB - WEIGHTS_GB - WORKSPACE_GB) * 0.9;
const CAG_PREFIXES = Math.floor(FREE_GB / CAG_KV_GB);
const CAG_PREFILL_MS = prefillMs(QUESTION);
const CAG_REBUILD_S = (KB * 2 * PARAMS) / (H100_FLOPS * MFU);

const STEPS = [
  'Documents become chunks',
  'Chunks become vectors',
  'A question lands nearby',
  'Assemble the prompt',
  'The next query shares only the system prompt',
  'CAG: preload once, reuse for everyone',
  'The bill, side by side',
];
const CAPTIONS = [
  'The offline half of RAG: parse the documents, split each one into chunks of about 512 tokens, and keep the chunk text plus its source metadata. Eight documents become 64 chunks.',
  'Each chunk goes through a separate embedding model, which turns text into a vector. The vectors go into a vector database: one dot per chunk in a high-dimensional space, sketched here in two dimensions.',
  'At query time the same embedding model turns the question into a vector. The vector database returns the chunks whose vectors are closest by cosine similarity: here, eight of them.',
  'The eight chunks are pasted into the prompt after a fixed system prompt, followed by the question. The prompt is now 4,296 tokens, almost all of it retrieved context, and the GPU must prefill every one of them.',
  'A second question retrieves different chunks. The system prompt is still identical, so the prefix cache can reuse its 150 tokens and nothing more: a 3.5% hit rate, about 4 ms saved out of 127 ms.',
  'CAG skips retrieval. The whole 100,000-token knowledge base is prefilled once, its KV is stored as one long prefix, and every query reuses it and appends only its question: about 0.9 ms of prefill.',
  'The trade, in one table: RAG pays retrieval latency and a long prefill per query but keeps memory small and knowledge fresh. CAG pays 13.1 GB of pinned KV per knowledge base but answers with a tiny prefill.',
];

// A deterministic 2-D sketch of 64 chunk vectors. No Math.random: the same
// picture on the server and in the browser.
function seeded(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}
const rnd = seeded(20261029);
const DOTS = Array.from({ length: 64 }, () => ({
  x: 0.04 + rnd() * 0.92,
  y: 0.06 + rnd() * 0.88,
}));
const Q1 = { x: 0.62, y: 0.55 };
const Q2 = { x: 0.3, y: 0.36 };
const nearest = (q: { x: number; y: number }, n: number) =>
  DOTS.map((d, i) => ({ i, d: (d.x - q.x) ** 2 + (d.y - q.y) ** 2 }))
    .sort((a, b) => a.d - b.d)
    .slice(0, n)
    .map((o) => o.i);
const N1 = nearest(Q1, 8);
const N2 = nearest(Q2, 8);

const PX0 = 282;
const PX1 = 614;
const PY0 = 46;
const PY1 = 144;
const planeX = (x: number) => PX0 + x * (PX1 - PX0);
const planeY = (y: number) => PY1 - y * (PY1 - PY0);

const TABLE = [
  { label: 'TTFT', rag: '150–200 ms (retrieval + prefill)', cag: '≈3 ms (0.9 ms prefill)' },
  { label: 'KV memory', rag: '0.56 GB per query, freed after', cag: '13.1 GB pinned per KB' },
  { label: 'Freshness', rag: 'fresh: re-index one chunk', cag: 'stale until ~8 s re-prefill' },
  { label: 'KB size', rag: 'unbounded (vector DB)', cag: '≤100K tokens here; 4 per 80 GB' },
  { label: 'Failure mode', rag: 'bad retrieval: wrong chunk', cag: 'overflow; lost in the middle' },
  { label: 'Multi-tenant', rag: 'per-request KV, trivial', cag: '4 prefixes; eviction ≈ 8 s' },
];

export default function RagVsCag() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [chunks, setChunks] = useState(8);
  useEffect(() => setChunks(8), [s.step]);

  const P = promptTokens(chunks);
  const ms = prefillMs(P);
  const kv = kvGB(P);
  const tMs = useTween(ms, 500);
  const tKv = useTween(kv, 500);
  const tHit = useTween(SYS / P, 500);

  // prompt bar, full width
  const pxTok = 608 / P;
  const sysW = SYS * pxTok;
  const qW = (QUESTION + TEMPLATE) * pxTok;
  const chunkW = CHUNK * pxTok;

  // two-query view, with labels on the left
  const px4 = 554 / P;
  const sysW4 = SYS * px4;
  const chunkW4 = CHUNK * px4;
  const qW4 = (QUESTION + TEMPLATE) * px4;

  const q = s.step >= 4 ? Q2 : Q1;
  const near = s.step >= 4 ? N2 : N1;

  const readout =
    s.step <= 2 ? (
      <Readout
        items={[
          { label: 'Knowledge base', value: '64 chunks', kind: 'text', note: '8 documents × 8 chunks' },
          { label: 'Chunk size', value: '512 tokens', kind: 'text', note: '≈ 380 English words' },
          { label: 'Retrieved', value: `${chunks} chunks`, kind: 'memory', note: 'nearest by cosine' },
          { label: 'Prompt if asked now', value: `${fmt.int(P)} tokens`, kind: 'compute', note: 'system + chunks + question' },
        ]}
      />
    ) : s.step <= 4 ? (
      <Readout
        items={[
          { label: 'Prompt', value: `${fmt.int(P)} tokens`, kind: 'compute', note: `150 + ${chunks}×512 + 50` },
          { label: 'Prefill, H100', value: `${ms.toFixed(0)} ms`, kind: 'compute', note: `${(2 * PARAMS * P / 1e12).toFixed(0)} TFLOP at 55% MFU` },
          { label: 'KV for the prompt', value: `${kv.toFixed(2)} GB`, kind: 'memory', note: '128 KiB per token' },
          { label: 'Prefix hit', value: `${(tHit * 100).toFixed(1)}%`, kind: 'memory', note: 'only the system prompt' },
        ]}
      />
    ) : s.step === 5 ? (
      <Readout
        items={[
          { label: 'CAG prefix', value: '100,000 tokens', kind: 'memory', note: 'the whole knowledge base' },
          { label: 'KV pinned', value: `${CAG_KV_GB.toFixed(1)} GB`, kind: 'memory', note: '128 KiB per token' },
          { label: 'Prefill per query', value: `${CAG_PREFILL_MS.toFixed(1)} ms`, kind: 'compute', note: '30-token question' },
          { label: 'Prefixes per 80 GB', value: `${CAG_PREFIXES}`, kind: 'token', note: 'after weights and workspace' },
        ]}
      />
    ) : (
      <Readout
        items={[
          { label: 'RAG TTFT', value: '≈150–200 ms', kind: 'compute', note: '20–70 ms retrieval + prefill' },
          { label: 'CAG TTFT', value: '≈3 ms', kind: 'token', note: '0.9 ms prefill + overhead' },
          { label: 'RAG KV', value: `${kv.toFixed(2)} GB`, kind: 'memory', note: 'per query, freed after' },
          { label: 'CAG KV', value: `${CAG_KV_GB.toFixed(1)} GB`, kind: 'memory', note: 'pinned per knowledge base' },
        ]}
      />
    );

  return (
    <Stage
      rootRef={ref}
      title="RAG and CAG: what retrieval and caching each cost the serving GPU"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['memory', ['compute', 'prefill / new work'], ['token', 'question'], ['comm', 'embedding call'], ['idle', 'free'], ['error', 'miss / eviction']]}
      {...s}
      readout={readout}
      controls={
        <Slider
          label="Chunks retrieved per question (RAG only; CAG ignores this)"
          value={chunks}
          min={2}
          max={16}
          onChange={setChunks}
          format={(v) => `${v} chunks = ${fmt.int(v * CHUNK)} tokens`}
          kind="compute"
        />
      }
    >
      <svg
        viewBox="0 0 640 360"
        role="img"
        aria-label="Top: documents are split into chunks, embedded as dots in a two-dimensional sketch, and a question vector lights up its eight nearest chunks. Bottom: the assembled 4,296-token prompt with a prefill cost bar and KV memory, then two queries sharing only the system prompt, then CAG where a 100,000-token knowledge base is one reused KV prefix, and finally a side-by-side RAG versus CAG table."
      >
        {/* ── step 0: documents → chunks ── */}
        <Fade show={s.step === 0}>
          <Label x={16} y={28} tone={2} size={11} bold>Documents → chunks (the offline index)</Label>
          {[0, 1, 2].map((i) => (
            <g key={i}>
              <rect x={20 + i * 54} y={46} width={40} height={52} rx={2} fill="var(--ground-3)" stroke="var(--rule-2)" />
              {[0, 1, 2, 3].map((j) => (
                <line key={j} x1={26 + i * 54} x2={52 + i * 54} y1={56 + j * 9} y2={56 + j * 9} className="v-rule" />
              ))}
            </g>
          ))}
          <Label x={20} y={114} size={9.5} tone={3}>8 documents</Label>
          <Arrow x1={180} y1={72} x2={222} y2={72} muted />
          <Label x={201} y={62} anchor="middle" size={9.5} tone={3}>split</Label>
          {Array.from({ length: 8 }, (_, i) => (
            <g key={i}>
              <rect x={232 + (i % 4) * 94} y={48 + Math.floor(i / 4) * 28} width={82} height={18} rx={2} fill="var(--compute)" opacity={0.85} />
              <text
                x={232 + (i % 4) * 94 + 41}
                y={48 + Math.floor(i / 4) * 28 + 12.5}
                textAnchor="middle"
                className="v-label"
                style={{ fontSize: 9, fill: '#0e1116', fontWeight: 600 }}
              >
                {`chunk ${i + 1}`}
              </text>
            </g>
          ))}
          <Label x={232} y={112} size={9.5} tone={3}>each chunk ≈ 512 tokens · 64 chunks in the index</Label>
          <Label x={16} y={140} size={9.5} tone={3}>Chunking is the offline, one-time half: parse, split, embed, store.</Label>
        </Fade>

        {/* ── steps 1-4: the retrieval world ── */}
        <Fade show={s.step >= 1 && s.step <= 4}>
          <Label x={16} y={28} tone={2} size={11} bold>Knowledge base → vectors</Label>
          <rect x={20} y={50} width={30} height={40} rx={2} fill="var(--ground-3)" stroke="var(--rule-2)" />
          {[0, 1, 2].map((j) => (
            <line key={j} x1={25} x2={45} y1={58 + j * 9} y2={58 + j * 9} className="v-rule" />
          ))}
          <Label x={16} y={106} size={9.5} tone={3}>8 docs</Label>
          <Arrow x1={58} y1={70} x2={88} y2={70} kind="memory" />
          <Label x={73} y={60} anchor="middle" size={9} kind="memory">embed</Label>
          {Array.from({ length: 8 }, (_, i) => (
            <rect key={i} x={94 + (i % 4) * 40} y={54 + Math.floor(i / 4) * 18} width={34} height={14} rx={2} fill="var(--memory)" opacity={0.65} />
          ))}
          <Label x={94} y={104} size={9.5} tone={3}>64 chunk vectors</Label>
          <Label x={94} y={118} size={9.5} tone={3}>one per 512-token chunk</Label>

          <rect x={270} y={36} width={354} height={116} rx={3} fill="var(--ground)" stroke="var(--rule)" />
          <Label x={278} y={30} size={9.5} tone={3}>embedding space (2-D sketch of 768+ dimensions)</Label>
          <line x1={PX0} x2={PX1} y1={PY1} y2={PY1} className="v-rule" />
          <line x1={PX0} x2={PX0} y1={PY0} y2={PY1} className="v-rule" />

          {DOTS.map((d, i) => {
            const on = s.step >= 2 && near.includes(i);
            return (
              <circle
                key={i}
                cx={planeX(d.x)}
                cy={planeY(d.y)}
                r={on ? 4 : 2.6}
                fill={on ? 'var(--memory)' : 'var(--text-3)'}
                opacity={on ? 1 : 0.55}
              />
            );
          })}

          <Fade show={s.step >= 2}>
            {near.map((i) => (
              <line
                key={i}
                x1={planeX(q.x)}
                y1={planeY(q.y)}
                x2={planeX(DOTS[i].x)}
                y2={planeY(DOTS[i].y)}
                stroke="var(--memory)"
                strokeWidth={1}
                opacity={0.28}
              />
            ))}
            <circle cx={planeX(q.x)} cy={planeY(q.y)} r={6} fill="var(--token)" stroke="var(--ground-2)" strokeWidth={2} />
            <Label x={planeX(q.x) + 10} y={planeY(q.y) + 3.5} size={9.5} kind="token">
              {s.step >= 4 ? 'question 2' : 'question'}
            </Label>
            <Label x={planeX(q.x)} y={planeY(q.y) - 12} anchor="middle" size={9} kind="memory">
              {`${near.length} nearest`}
            </Label>
          </Fade>
        </Fade>

        {/* ── steps 5-6: CAG removes the retrieval step ── */}
        <Fade show={s.step >= 5}>
          <Label x={16} y={28} tone={2} size={11} bold>CAG: the knowledge base becomes one KV prefix</Label>
          <rect x={20} y={54} width={36} height={46} rx={2} fill="var(--ground-3)" stroke="var(--rule-2)" />
          {[0, 1, 2, 3].map((j) => (
            <line key={j} x1={26} x2={50} y1={64 + j * 9} y2={64 + j * 9} className="v-rule" />
          ))}
          <Label x={16} y={116} size={9.5} tone={3}>100K-token knowledge base</Label>
          <Arrow x1={92} y1={76} x2={146} y2={76} kind="memory" />
          <Label x={119} y={66} anchor="middle" size={9.5} kind="memory">prefill once</Label>
          <rect x={154} y={58} width={446} height={30} rx={2} fill="var(--memory)" opacity={0.85} />
          <text x={377} y={77} textAnchor="middle" className="v-label" style={{ fontSize: 10.5, fill: '#0e1116', fontWeight: 700 }}>
            100,000 tokens of KV · 13.1 GB
          </text>
          <Label x={154} y={104} size={9.5} tone={3}>computed once, stored as prefix blocks</Label>
          <Label x={154} y={120} size={9.5} kind="token">every query reuses this prefix and appends only its question</Label>
          <Label x={16} y={142} size={9.5} tone={3}>No embedding call. No vector search. No per-query retrieval.</Label>
        </Fade>

        {/* ── step 3: assemble the RAG prompt ── */}
        <Fade show={s.step === 3}>
          <Label x={16} y={172} tone={2} size={11} bold>The prompt, and what it costs</Label>
          <rect x={16} y={182} width={608} height={28} rx={3} fill="none" stroke="var(--rule)" />
          <rect x={16} y={182} width={sysW} height={28} fill="var(--memory)" opacity={0.9} />
          {Array.from({ length: chunks }, (_, i) => (
            <rect
              key={i}
              x={16 + (SYS + i * CHUNK) * pxTok + 0.6}
              y={182}
              width={Math.max(1, chunkW - 1.2)}
              height={28}
              fill="var(--compute)"
              opacity={0.9}
            />
          ))}
          <rect x={16 + (SYS + chunks * CHUNK) * pxTok} y={182} width={Math.max(1, qW)} height={28} fill="var(--compute)" opacity={0.55} />
          <Label x={16} y={178} size={9} kind="memory">system 150</Label>
          <Label x={16 + sysW + (chunks * chunkW) / 2} y={178} anchor="middle" size={9} kind="compute">
            {`${chunks} chunks × 512 = ${fmt.int(chunks * CHUNK)} tokens`}
          </Label>
          <Label x={624} y={178} anchor="end" size={9} kind="compute">question + template 50</Label>

          {[0, 1000, 2000, 3000, 4000, P]
            .filter((v, i, a) => v <= P && a.indexOf(v) === i)
            .map((t) => {
              const x = 16 + t * pxTok;
              return (
                <g key={t}>
                  <line x1={x} x2={x} y1={214} y2={219} className="v-rule" />
                  <Label x={x} y={231} anchor={t === P ? 'end' : 'middle'} size={8.5} tone={3}>{fmt.int(t)}</Label>
                </g>
              );
            })}
          <Label x={16} y={243} size={8.5} tone={3}>prompt tokens →</Label>

          <Bar x={80} y={250} w={520} h={14} frac={tMs / 300} kind="compute" label="prefill" value={`${tMs.toFixed(0)} ms`} />
          <Label x={80} y={272} size={8.5} tone={3}>
            {`${fmt.int(P)} tokens × 2 × 8.03B FLOP ÷ (989 TFLOP/s × 0.55) = ${(2 * PARAMS * P / 1e12).toFixed(0)} TFLOP`}
          </Label>

          <Bar x={80} y={284} w={520} h={14} frac={tKv / 2} kind="memory" label="KV" value={`${tKv.toFixed(2)} GB`} />
          <Label x={80} y={306} size={8.5} tone={3}>
            {`${fmt.int(P)} tokens × 128 KiB per token = ${(kv * 1000).toFixed(0)} MB, freed when the request ends`}
          </Label>

          <rect x={80} y={316} width={520} height={14} rx={2} fill="none" stroke="var(--rule)" />
          <Span x={80} y={316} w={520 * (30 / 300)} h={14} kind="comm" opacity={0.85} />
          <Span x={80 + 520 * (30 / 300)} y={316} w={520 * (10 / 300)} h={14} kind="memory" opacity={0.85} />
          <Label x={72} y={327} anchor="end" tone={2} size={10}>retrieval</Label>
          <Label x={80 + 520 * (40 / 300) + 8} y={327} size={9.5} tone={2}>20–70 ms (approximate)</Label>
          <Label x={80} y={343} size={8.5} tone={3}>embedding call + vector search happen before prefill starts; both are part of TTFT.</Label>
          <Label x={80} y={356} size={8.5} tone={3}>TTFT ≈ 150–200 ms. Prefill intensity 4,296 FLOP/byte is about 15× the H100 ridge point.</Label>
        </Fade>

        {/* ── step 4: the second query reuses only the system prompt ── */}
        <Fade show={s.step === 4}>
          <Label x={16} y={172} tone={2} size={11} bold>Two questions, one shared system prompt</Label>
          {[0, 1].map((k) => (
            <g key={k}>
              <Label x={16} y={190 + k * 30} size={9.5} tone={3}>{k === 0 ? 'query A' : 'query B'}</Label>
              <rect x={70} y={180 + k * 30} width={554} height={22} rx={2} fill="none" stroke="var(--rule)" />
              <rect x={70} y={180 + k * 30} width={sysW4} height={22} fill="var(--memory)" opacity={0.9} />
              {Array.from({ length: chunks }, (_, i) => (
                <rect
                  key={i}
                  x={70 + (SYS + i * CHUNK) * px4 + 0.5}
                  y={180 + k * 30}
                  width={Math.max(1, chunkW4 - 1)}
                  height={22}
                  fill="var(--compute)"
                  opacity={k === 0 ? 0.9 : 0.6}
                />
              ))}
              <rect x={70 + (SYS + chunks * CHUNK) * px4} y={180 + k * 30} width={Math.max(1, qW4)} height={22} fill="var(--compute)" opacity={0.5} />
            </g>
          ))}
          <line x1={70 + sysW4} x2={70 + sysW4} y1={176} y2={240} stroke="var(--memory)" strokeWidth={1.4} strokeDasharray="3 3" />
          <Label x={70 + sysW4 + 4} y={176} size={9} kind="memory">
            {`shared ${SYS} tokens = ${((SYS / P) * 100).toFixed(1)}%`}
          </Label>
          <Label x={70 + sysW4 + 4} y={244} size={9} kind="error">everything after this is a miss</Label>

          {[0, 1000, 2000, 3000, 4000, P]
            .filter((v, i, a) => v <= P && a.indexOf(v) === i)
            .map((t) => {
              const x = 70 + t * px4;
              return (
                <g key={t}>
                  <line x1={x} x2={x} y1={254} y2={259} className="v-rule" />
                  <Label x={x} y={271} anchor={t === P ? 'end' : 'middle'} size={8.5} tone={3}>{fmt.int(t)}</Label>
                </g>
              );
            })}

          <Bar x={120} y={282} w={430} h={12} frac={ms / 150} kind="compute" label="A" value={`${ms.toFixed(0)} ms`} />
          <Bar x={120} y={300} w={430} h={12} frac={(ms - SYS * 0.0295) / 150} kind="compute" label="B" value={`${(ms - SYS * 0.0295).toFixed(0)} ms`} />

          <Label x={16} y={324} size={8.5} tone={3}>Only the blue prefix is reused. Different chunks mean the cache walk stops at token 150.</Label>
          <Label x={16} y={338} size={8.5} tone={3}>Same 8 chunks sorted by document ID: 4,246 shared tokens → 98.8% hit, prefill ≈ 1.5 ms.</Label>
          <Label x={16} y={352} size={8.5} tone={3}>Stable first, variable last. Byte-identical or nothing.</Label>
        </Fade>

        {/* ── step 5: CAG memory ── */}
        <Fade show={s.step === 5}>
          <Label x={16} y={172} tone={2} size={11} bold>Memory: four knowledge bases fit beside the weights</Label>
          <Label x={16} y={190} size={9.5} tone={3}>80 GB H100, Llama-3.1-8B in FP16</Label>
          <rect x={16} y={202} width={608} height={22} rx={2} fill="none" stroke="var(--rule)" />
          <rect x={16} y={202} width={608 * (WEIGHTS_GB / GPU_GB)} height={22} fill="var(--text-3)" opacity={0.55} />
          <rect x={16 + 608 * (WEIGHTS_GB / GPU_GB)} y={202} width={608 * (WORKSPACE_GB / GPU_GB)} height={22} fill="var(--idle)" opacity={0.5} />
          {Array.from({ length: CAG_PREFIXES }, (_, i) => {
            const x = 16 + 608 * ((WEIGHTS_GB + WORKSPACE_GB) / GPU_GB) + i * 608 * (CAG_KV_GB / GPU_GB);
            return (
              <g key={i}>
                <rect x={x + 0.6} y={202} width={608 * (CAG_KV_GB / GPU_GB) - 1.2} height={22} fill="var(--memory)" opacity={0.85} />
                <text x={x + (608 * (CAG_KV_GB / GPU_GB)) / 2} y={216.5} textAnchor="middle" className="v-label" style={{ fontSize: 8.5, fill: '#0e1116', fontWeight: 700 }}>
                  {`KB ${i + 1}`}
                </text>
              </g>
            );
          })}
          <text x={16 + (608 * (WEIGHTS_GB / GPU_GB)) / 2} y={216.5} textAnchor="middle" className="v-label" style={{ fontSize: 8.5, fill: 'var(--text)', fontWeight: 600 }}>
            weights 16.1
          </text>
          <Label x={16 + 608 * ((WEIGHTS_GB + WORKSPACE_GB / 2) / GPU_GB)} y={238} anchor="middle" size={8.5} tone={3}>workspace 3.2</Label>
          <Label x={16 + 608 * ((WEIGHTS_GB + WORKSPACE_GB + CAG_PREFIXES * CAG_KV_GB + (GPU_GB - WEIGHTS_GB - WORKSPACE_GB - CAG_PREFIXES * CAG_KV_GB) / 2) / GPU_GB)} y={238} anchor="middle" size={8.5} tone={3}>free 8.3</Label>

          <Label x={16} y={264} size={9} tone={3}>KV per prefix = 100,000 × 131,072 bytes = 13.1 GB (12.2 GiB).</Label>
          <Label x={16} y={280} size={9} tone={3}>Free for KV = (80 − 16.06 − 3.2) × 0.9 = 54.7 GB.</Label>
          <Label x={16} y={296} size={9} kind="memory">54.7 ÷ 13.1 = 4.2 → 4 prefixes; 4 × 13.1 = 52.4 GB, leaving 2.3 GB.</Label>
          <Label x={16} y={312} size={9} kind="compute">Each query appends 30 tokens: 0.9 ms prefill, no retrieval.</Label>
          <Label x={16} y={328} size={9} kind="error">{`Update the KB and you re-prefill 100K tokens: ${CAG_REBUILD_S.toFixed(2)} s of GPU.`}</Label>
          <Label x={16} y={344} size={9} kind="error">{`A fifth tenant evicts one prefix; its next query pays that ${CAG_REBUILD_S.toFixed(2)} s.`}</Label>
        </Fade>

        {/* ── step 6: the comparison table ── */}
        <Fade show={s.step === 6}>
          <Label x={16} y={172} tone={2} size={11} bold>RAG vs CAG: the bill, side by side</Label>
          <line x1={16} x2={624} y1={180} y2={180} className="v-rule" />
          <Label x={180} y={196} size={10} bold kind="compute">RAG</Label>
          <Label x={440} y={196} size={10} bold kind="memory">CAG</Label>
          {TABLE.map((r, i) => (
            <g key={r.label}>
              <line x1={16} x2={624} y1={208 + i * 22} y2={208 + i * 22} className="v-rule faint" />
              <Label x={16} y={222 + i * 22} size={9} tone={3} mono={false}>{r.label}</Label>
              <Label x={180} y={222 + i * 22} size={9} tone={2} mono={false}>{r.rag}</Label>
              <Label x={440} y={222 + i * 22} size={9} tone={2} mono={false}>{r.cag}</Label>
            </g>
          ))}
          <Label x={16} y={352} size={9.5} kind="token" mono={false}>
            Hybrid: CAG the stable core (product docs, policies); RAG the long tail and anything fresh.
          </Label>
        </Fade>
      </svg>
    </Stage>
  );
}
