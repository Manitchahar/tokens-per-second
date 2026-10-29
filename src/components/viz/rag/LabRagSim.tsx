import { useState } from 'react';
import { Stage, Segmented, Readout } from '../lib/Stage';
import { Label, Arrow } from '../lib/svg';

interface QueryCase {
  query: string;
  type: 'concept' | 'code' | 'numeric' | 'acronym';
  bm25Hit: boolean;
  denseHit: boolean;
  hybridHit: boolean;
  rerankHit: boolean;
  topDoc: string;
  answer: string;
}

const QUERIES: Record<string, QueryCase> = {
  bandwidth: {
    query: 'What is the HBM3 memory bandwidth of an NVIDIA H100 GPU?',
    type: 'numeric',
    bm25Hit: true,
    denseHit: true,
    hybridHit: true,
    rerankHit: true,
    topDoc: 'Doc 14: "NVIDIA H100 SXM features 80 GB of HBM3 memory with a peak bandwidth of 3.35 TB/s..."',
    answer: 'The NVIDIA H100 SXM GPU provides approximately 3.35 TB/s of HBM3 memory bandwidth [Doc 14].',
  },
  error_code: {
    query: 'How do I resolve ERR_NCCL_COMM_TIMEOUT_502 on multi-node clusters?',
    type: 'code',
    bm25Hit: true,
    denseHit: false, // Dense fails on exact error code!
    hybridHit: true,
    rerankHit: true,
    topDoc: 'Doc 82: "ERR_NCCL_COMM_TIMEOUT_502 occurs when inter-node InfiniBand links drop sync. Set NCCL_IB_TIMEOUT=22..."',
    answer: 'ERR_NCCL_COMM_TIMEOUT_502 is caused by InfiniBand sync drops; resolve it by increasing NCCL_IB_TIMEOUT [Doc 82].',
  },
  paraphrase: {
    query: 'Why do language models generate words one by one rather than all together?',
    type: 'concept',
    bm25Hit: false, // BM25 fails due to completely different phrasing!
    denseHit: true,
    hybridHit: true,
    rerankHit: true,
    topDoc: 'Doc 03: "Autoregressive decoding produces one token per step because next-token probability is conditioned on all prior tokens..."',
    answer: 'LLMs generate tokens autoregressively because each new token depends causally on the entire prior sequence [Doc 03].',
  },
  kv_cache: {
    query: 'How much KV cache memory does Llama-3.1-8B consume per token in BF16?',
    type: 'numeric',
    bm25Hit: true,
    denseHit: true,
    hybridHit: true,
    rerankHit: true,
    topDoc: 'Doc 21: "Llama-3.1-8B has 32 layers, 8 KV heads, and head dimension 128. In BF16, 2 * 32 * 8 * 128 * 2 = 131,072 bytes (128 KB) per token..."',
    answer: 'Llama-3.1-8B consumes exactly 131,072 bytes (128 KB) of KV cache memory per token in BF16 [Doc 21].',
  },
};

export default function LabRagSim() {
  const [selectedKey, setSelectedKey] = useState<string>('error_code');
  const q = QUERIES[selectedKey];

  return (
    <Stage
      rootRef={{ current: null }}
      title="RAG Pipeline Simulator: BM25, Dense, Hybrid RRF & Reranking"
      steps={['1. Interactive RAG Query Run']}
      step={0}
      go={() => {}}
      playing={false}
      setPlaying={() => {}}
      inScrolly={false}
      hideTransport={true}
      legend={['token', 'memory', 'compute', 'error']}
      readout={
        <Readout items={[
          {
            label: 'BM25 Sparse Hit',
            value: q.bm25Hit ? 'HIT (Found)' : 'MISS',
            kind: q.bm25Hit ? 'token' : 'error',
            note: q.type === 'code' ? 'Wins on exact error codes' : q.type === 'concept' ? 'Fails on synonym phrasing' : 'Keyword match',
          },
          {
            label: 'Dense Vector Hit',
            value: q.denseHit ? 'HIT (Found)' : 'MISS',
            kind: q.denseHit ? 'token' : 'error',
            note: q.type === 'concept' ? 'Wins on semantic concepts' : q.type === 'code' ? 'Fails on specific alphanumeric ID' : 'Embedding match',
          },
          {
            label: 'Hybrid RRF Rank',
            value: q.hybridHit ? 'HIT (Rank 1)' : 'MISS',
            kind: 'token',
            note: 'Reciprocal Rank Fusion captures both',
          },
          {
            label: 'Cross-Encoder Score',
            value: '0.982',
            kind: 'compute',
            note: 'Joint attention verifies exact query-doc fit',
          },
        ]} />
      }
      controls={
        <Segmented
          label="Test Query Case"
          value={selectedKey}
          options={[
            { value: 'error_code', label: '1. Exact ID (ERR_NCCL_COMM)' },
            { value: 'paraphrase', label: '2. Paraphrased (Autoregressive)' },
            { value: 'bandwidth', label: '3. Technical Spec (H100 HBM)' },
            { value: 'kv_cache', label: '4. Memory Math (Llama-8B KV)' },
          ]}
          onChange={setSelectedKey}
        />
      }
    >
      <svg viewBox="0 0 640 330" role="img" aria-label="Interactive simulation of a complete RAG pipeline showing retrieval stage scores and generated answer.">
        {/* User Query Banner */}
        <g transform="translate(30, 20)">
          <rect x={0} y={0} width={580} height={40} rx={4} fill="var(--ground-2)" stroke="var(--rule-2)" strokeWidth={1} />
          <Label x={15} y={18} tone={3} size={8.5} bold>Active Query:</Label>
          <Label x={15} y={32} tone={1} size={9.5} bold kind="token">"{q.query}"</Label>
        </g>

        {/* Pipeline Columns */}
        <g transform="translate(30, 75)">
          {/* Column 1: Sparse BM25 */}
          <g transform="translate(0, 0)">
            <rect x={0} y={0} width={135} height={105} rx={4}
              fill={q.bm25Hit ? 'var(--ground-3)' : 'var(--error-dim)'}
              stroke={q.bm25Hit ? 'var(--rule-2)' : 'var(--error)'} strokeWidth={1} />
            <Label x={12} y={20} tone={1} size={9.5} bold>BM25 Sparse</Label>
            <Label x={12} y={36} tone={3} size={8}>Inverted Index</Label>
            <line x1={10} y1={44} x2={125} y2={44} stroke="var(--rule-2)" strokeWidth={0.6} />
            <Label x={12} y={62} tone={1} size={9} kind={q.bm25Hit ? 'token' : 'error'} bold>
              {q.bm25Hit ? '● Top-5 HIT' : '▲ Top-5 MISS'}
            </Label>
            <Label x={12} y={80} tone={3} size={7.5}>
              {q.type === 'code' ? 'Matches exact code' : q.type === 'concept' ? 'Zero token overlap' : 'Lexical score: 18.4'}
            </Label>
          </g>

          {/* Column 2: Dense Vector */}
          <g transform="translate(148, 0)">
            <rect x={0} y={0} width={135} height={105} rx={4}
              fill={q.denseHit ? 'var(--ground-3)' : 'var(--error-dim)'}
              stroke={q.denseHit ? 'var(--rule-2)' : 'var(--error)'} strokeWidth={1} />
            <Label x={12} y={20} tone={1} size={9.5} bold>Dense Vector</Label>
            <Label x={12} y={36} tone={3} size={8}>all-MiniLM + FAISS</Label>
            <line x1={10} y1={44} x2={125} y2={44} stroke="var(--rule-2)" strokeWidth={0.6} />
            <Label x={12} y={62} tone={1} size={9} kind={q.denseHit ? 'token' : 'error'} bold>
              {q.denseHit ? '● Top-5 HIT' : '▲ Top-5 MISS'}
            </Label>
            <Label x={12} y={80} tone={3} size={7.5}>
              {q.type === 'concept' ? 'Semantic capture' : q.type === 'code' ? 'Embedding blurred' : 'Cosine: 0.841'}
            </Label>
          </g>

          {/* Column 3: Hybrid RRF */}
          <g transform="translate(296, 0)">
            <rect x={0} y={0} width={135} height={105} rx={4} fill="var(--memory-dim)" stroke="var(--memory)" strokeWidth={1.2} />
            <Label x={12} y={20} tone={1} size={9.5} bold kind="memory">Hybrid RRF</Label>
            <Label x={12} y={36} tone={3} size={8}>Rank Fusion (k=60)</Label>
            <line x1={10} y1={44} x2={125} y2={44} stroke="var(--rule-2)" strokeWidth={0.6} />
            <Label x={12} y={62} tone={1} size={9} kind="token" bold>● Top-5 HIT (Rank 1)</Label>
            <Label x={12} y={80} tone={2} size={7.5}>Fuses sparse + dense</Label>
            <Label x={12} y={94} tone={1} size={7.5} kind="token">RRF: 0.0322</Label>
          </g>

          {/* Column 4: Cross-Encoder */}
          <g transform="translate(445, 0)">
            <rect x={0} y={0} width={135} height={105} rx={4} fill="var(--compute-dim)" stroke="var(--compute)" strokeWidth={1.2} />
            <Label x={12} y={20} tone={1} size={9.5} bold kind="compute">Cross-Encoder</Label>
            <Label x={12} y={36} tone={3} size={8}>ms-marco-MiniLM</Label>
            <line x1={10} y1={44} x2={125} y2={44} stroke="var(--rule-2)" strokeWidth={0.6} />
            <Label x={12} y={62} tone={1} size={9} kind="compute" bold>Joint Attention</Label>
            <Label x={12} y={80} tone={2} size={7.5}>Rerank 20 → 5</Label>
            <Label x={12} y={94} tone={1} size={7.5} kind="token">Confidence: 98.2%</Label>
          </g>
        </g>

        {/* Down Arrow to Generator */}
        <Arrow x1={320} y1={185} x2={320} y2={205} kind="token" width={1.4} />

        {/* End-to-End Generated Output Box */}
        <g transform="translate(30, 208)">
          <rect x={0} y={0} width={580} height={105} rx={5} fill="var(--ground-2)" stroke="var(--rule-2)" strokeWidth={1} />
          <Label x={15} y={20} tone={1} size={10} bold kind="token">
            Generator Output (Qwen2.5-1.5B-Instruct + Grounded Context):
          </Label>
          <rect x={15} y={30} width={550} height={38} rx={3} fill="var(--ground-3)" />
          <Label x={25} y={48} tone={1} size={9} bold>
            "{q.answer}"
          </Label>
          <Label x={25} y={60} tone={3} size={8}>
            Grounded in: {q.topDoc.substring(0, 80)}...
          </Label>

          <g transform="translate(15, 78)">
            <Label x={0} y={14} tone={3} size={8.5}>
              Benchmark Recall@5 across 30 test questions:
            </Label>
            <Label x={225} y={14} tone={2} size={8.5}>
              BM25: <tspan fill="var(--text)">76.7%</tspan> | Dense: <tspan fill="var(--text)">80.0%</tspan> | Hybrid: <tspan fill="var(--token)" fontWeight="600">93.3%</tspan> | Reranked: <tspan fill="var(--token)" fontWeight="600">96.7%</tspan>
            </Label>
          </g>
        </g>
      </svg>
    </Stage>
  );
}
