import { useRef, useState } from 'react';
import { Stage, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Arrow } from '../lib/svg';

const STEPS = [
  '1. Chunk Size Trade-Off',
  '2. Anthropic Contextual Retrieval',
  '3. Multi-Turn Query Rewriting',
  '4. Bi-Encoder vs Cross-Encoder',
  '5. The Two-Stage Funnel',
  '6. Cross-Encoder Token Attention',
  '7. The End-to-End Pipeline',
];

const CAPTIONS = [
  'Tiny chunks (50 tokens) lose antecedents; massive chunks (2,000 tokens) dilute embeddings and bloat the KV cache.',
  'Anthropic Contextual Retrieval prepends 50–100 tokens of document summary, cutting retrieval failure rates by up to 49%.',
  'Conversational follow-ups like "What about in Europe?" lack keywords. Query rewriting expands them into standalone search queries.',
  'Bi-encoders embed query and document independently (no cross-token attention). Cross-encoders run full bidirectional self-attention.',
  'Production uses a two-stage funnel: Stage 1 retrieves top 100 candidates cheaply; Stage 2 reranks them down to the top 5 with cross-attention.',
  'Cross-encoders inspect token interactions: recognizing negations, subject-verb relations, and subtle nuances bi-encoders miss.',
  'The resulting pipeline balances throughput and precision: fast search over 1M docs, accurate reranking of 100, clean context for the LLM.',
];

export default function ChunkingReranking() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);

  const [chunkSize, setChunkSize] = useState<50 | 250 | 1000>(250);

  return (
    <Stage
      rootRef={ref}
      title="Chunking, Query Rewriting & Cross-Encoder Reranking"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['token', 'memory', 'compute', 'text']}
      {...s}
      readout={
        <Readout items={[
          {
            label: 'Chunk Size',
            value: `${chunkSize} tokens`,
            kind: chunkSize === 250 ? 'token' : chunkSize === 50 ? 'error' : 'compute',
            note: chunkSize === 50 ? 'Lost antecedents & context' : chunkSize === 1000 ? 'Diluted embeddings & large prompt' : 'Optimal semantic balance',
          },
          {
            label: 'Contextual Retrieval',
            value: '-49% failures',
            kind: 'token',
            note: 'Anthropic (2024): prepending doc summary to chunk + hybrid search',
          },
          {
            label: 'Stage 1 Candidate Pool',
            value: 'Top 100 Chunks',
            kind: 'memory',
            note: 'Retrieved via Bi-Encoder + BM25 in ~3 ms',
          },
          {
            label: 'Stage 2 Rerank Target',
            value: 'Top 5 Chunks',
            kind: 'compute',
            note: 'Cross-encoder full self-attention in ~15 ms',
          },
        ]} />
      }
      controls={
        <Segmented
          label="Chunk Size Simulation"
          value={chunkSize}
          options={[
            { value: 50, label: '50 tokens (Small)' },
            { value: 250, label: '250 tokens (Sweet spot)' },
            { value: 1000, label: '1000 tokens (Large)' },
          ]}
          onChange={(v) => setChunkSize(v as 50 | 250 | 1000)}
        />
      }
    >
      <svg viewBox="0 0 640 330" role="img" aria-label="Visual explanation of chunking trade-offs, contextual retrieval headers, bi-encoder vs cross-encoder attention, and the two-stage retrieval funnel.">
        {/* Step 1 & 2: CHUNKING & CONTEXTUAL RETRIEVAL */}
        {(s.step <= 1 || s.step === 2) && (
          <g transform="translate(30, 25)">
            <rect x={0} y={0} width={580} height={235} rx={6} fill="var(--ground-2)" stroke="var(--rule-2)" strokeWidth={1} />
            <Label x={20} y={24} tone={1} size={12} bold>
              {s.step === 2 ? 'Multi-Turn Query Rewriting' : s.step === 1 ? 'Anthropic Contextual Retrieval (2024)' : 'Chunking Size & Context Fragmentation'}
            </Label>
            <Label x={20} y={40} tone={3} size={9.5}>
              {s.step === 2
                ? 'Resolving pronouns and omitted context from chat history before retrieval'
                : s.step === 1
                ? 'Prepending 50-100 tokens of document-level context to prevent lost meaning'
                : 'Balancing semantic completeness against context window consumption'}
            </Label>

            {s.step === 0 && (
              <g transform="translate(20, 60)">
                {/* Chunk comparison boxes */}
                <g transform="translate(0, 0)">
                  <rect x={0} y={0} width={165} height={120} rx={4} fill={chunkSize === 50 ? 'var(--error-dim)' : 'var(--ground-3)'} stroke={chunkSize === 50 ? 'var(--error)' : 'var(--rule-2)'} strokeWidth={1.2} />
                  <Label x={15} y={22} tone={1} size={10} bold kind={chunkSize === 50 ? 'error' : undefined}>50 Tokens (Small)</Label>
                  <Label x={15} y={42} tone={3} size={8.5}>"It was updated to 4.2%."</Label>
                  <Label x={15} y={58} tone={3} size={8.5}>"This caused an outage."</Label>
                  <line x1={15} y1={72} x2={150} y2={72} stroke="var(--rule-2)" strokeWidth={0.8} />
                  <Label x={15} y={88} tone={2} size={8} kind="error">Who is "It"?</Label>
                  <Label x={15} y={104} tone={3} size={7.5}>Pronoun antecedent lost</Label>
                </g>

                <g transform="translate(180, 0)">
                  <rect x={0} y={0} width={175} height={120} rx={4} fill={chunkSize === 250 ? 'var(--token-dim)' : 'var(--ground-3)'} stroke={chunkSize === 250 ? 'var(--token)' : 'var(--rule-2)'} strokeWidth={1.4} />
                  <Label x={15} y={22} tone={1} size={10} bold kind={chunkSize === 250 ? 'token' : undefined}>250 Tokens (Optimal)</Label>
                  <Label x={15} y={42} tone={2} size={8.5}>"In Q3, ACME fee rate was</Label>
                  <Label x={15} y={56} tone={2} size={8.5}>updated to 4.2% for EU cards.</Label>
                  <Label x={15} y={70} tone={2} size={8.5}>This caused an API outage..."</Label>
                  <line x1={15} y1={78} x2={160} y2={78} stroke="var(--rule-2)" strokeWidth={0.8} />
                  <Label x={15} y={94} tone={1} size={8} kind="token" bold>Self-contained concept</Label>
                  <Label x={15} y={108} tone={3} size={7.5}>Fits tight prompt budget</Label>
                </g>

                <g transform="translate(370, 0)">
                  <rect x={0} y={0} width={165} height={120} rx={4} fill={chunkSize === 1000 ? 'var(--compute-dim)' : 'var(--ground-3)'} stroke={chunkSize === 1000 ? 'var(--compute)' : 'var(--rule-2)'} strokeWidth={1.2} />
                  <Label x={15} y={22} tone={1} size={10} bold kind={chunkSize === 1000 ? 'compute' : undefined}>1000 Tokens (Large)</Label>
                  <Label x={15} y={42} tone={3} size={8.5}>Covers 4 different topics:</Label>
                  <Label x={15} y={58} tone={3} size={8.5}>Billing, KYC, Datacenters...</Label>
                  <line x1={15} y1={72} x2={150} y2={72} stroke="var(--rule-2)" strokeWidth={0.8} />
                  <Label x={15} y={88} tone={2} size={8} kind="error">Diluted vector embedding</Label>
                  <Label x={15} y={104} tone={3} size={7.5}>Wastes 128KB/token KV cache</Label>
                </g>
              </g>
            )}

            {s.step === 1 && (
              <g transform="translate(20, 60)">
                <rect x={0} y={0} width={540} height={125} rx={4} fill="var(--ground-3)" stroke="var(--rule-2)" strokeWidth={1} />
                <rect x={15} y={15} width={510} height={38} rx={3} fill="var(--token-dim)" stroke="var(--token)" strokeWidth={1} />
                <Label x={25} y={32} tone={1} size={9} kind="token" bold>
                  [Contextual Header]: "This chunk is from SEC 10-K (2024) for ACME Corp, describing payment processor fees."
                </Label>
                <Label x={25} y={46} tone={3} size={8}>
                  Generated by prompt-cached LLM at ingestion time (~50–100 tokens)
                </Label>

                <rect x={15} y={62} width={510} height={48} rx={3} fill="var(--memory-dim)" stroke="var(--memory)" strokeWidth={0.8} />
                <Label x={25} y={80} tone={1} size={9} kind="memory" bold>
                  [Original Raw Chunk]: "The transaction fee rate was raised by 0.5% in July, reaching a ceiling of 4.2%."
                </Label>
                <Label x={25} y={98} tone={2} size={8.5}>
                  Result: Both BM25 and vector embeddings now index "ACME Corp", "SEC 10-K", and "payment processor".
                </Label>
              </g>
            )}

            {s.step === 2 && (
              <g transform="translate(20, 60)">
                <rect x={0} y={0} width={540} height={125} rx={4} fill="var(--ground-3)" stroke="var(--rule-2)" strokeWidth={1} />
                {/* Chat History */}
                <g transform="translate(15, 15)">
                  <Label x={0} y={12} tone={3} size={8.5} bold>Conversation History:</Label>
                  <Label x={0} y={28} tone={2} size={8.5}>User: "Tell me about the payment gateway timeout in US-East."</Label>
                  <Label x={0} y={42} tone={2} size={8.5}>Assistant: "It is configured to 1,200 ms with 3 retries."</Label>
                  <Label x={0} y={60} tone={1} size={9} bold kind="error">Follow-up: "What about in Europe?"</Label>
                </g>

                <Arrow x1={260} y1={55} x2={295} y2={55} kind="token" width={1.4} />

                {/* Rewritten Query */}
                <g transform="translate(305, 20)">
                  <rect x={0} y={0} width={220} height={75} rx={3} fill="var(--token-dim)" stroke="var(--token)" strokeWidth={1.2} />
                  <Label x={10} y={18} tone={1} size={9} bold kind="token">Rewritten Query for Retriever:</Label>
                  <Label x={10} y={36} tone={2} size={8.5}>"What is the payment gateway</Label>
                  <Label x={10} y={50} tone={2} size={8.5}>timeout configuration for the</Label>
                  <Label x={10} y={64} tone={1} size={8.5} bold kind="token">Europe (EU) region?"</Label>
                </g>
              </g>
            )}
          </g>
        )}

        {/* Steps 3, 4, 5, 6: BI-ENCODER VS CROSS-ENCODER & TWO-STAGE FUNNEL */}
        {s.step >= 3 && (
          <g transform="translate(30, 25)">
            <rect x={0} y={0} width={580} height={235} rx={6} fill="var(--ground-2)" stroke="var(--rule-2)" strokeWidth={1} />
            <Label x={20} y={24} tone={1} size={12} bold>
              {s.step === 4 ? 'The Two-Stage Retrieval & Reranking Funnel' : 'Bi-Encoder vs Cross-Encoder Attention Architecture'}
            </Label>
            <Label x={20} y={40} tone={3} size={9.5}>
              {s.step === 4
                ? 'Funneling 1,000,000 documents down to 100 with Bi-Encoder + BM25, then down to 5 with Cross-Encoder'
                : 'Comparing independent vector projection against full cross-token self-attention'}
            </Label>

            {s.step === 3 && (
              <g transform="translate(20, 55)">
                {/* Left: Bi-encoder */}
                <g transform="translate(0, 0)">
                  <rect x={0} y={0} width={255} height={135} rx={4} fill="var(--ground-3)" stroke="var(--rule-2)" strokeWidth={1} />
                  <Label x={15} y={20} tone={1} size={10} bold kind="memory">Bi-Encoder (Dual Tower)</Label>
                  <rect x={15} y={32} width={100} height={22} rx={2} fill="var(--ground-2)" stroke="var(--rule-2)" />
                  <Label x={65} y={46} anchor="middle" tone={2} size={8}>Query q</Label>
                  <rect x={140} y={32} width={100} height={22} rx={2} fill="var(--ground-2)" stroke="var(--rule-2)" />
                  <Label x={190} y={46} anchor="middle" tone={2} size={8}>Doc d</Label>

                  <Arrow x1={65} y1={54} x2={65} y2={76} kind="token" width={1} />
                  <Arrow x1={190} y1={54} x2={190} y2={76} kind="token" width={1} />

                  <rect x={15} y={76} width={100} height={22} rx={2} fill="var(--token-dim)" stroke="var(--token)" strokeWidth={0.8} />
                  <Label x={65} y={90} anchor="middle" tone={1} size={8}>Vector u</Label>
                  <rect x={140} y={76} width={100} height={22} rx={2} fill="var(--token-dim)" stroke="var(--token)" strokeWidth={0.8} />
                  <Label x={190} y={90} anchor="middle" tone={1} size={8}>Vector v</Label>

                  <Label x={127} y={118} anchor="middle" tone={1} size={9} bold kind="token">Score = u · v (Fast, O(1))</Label>
                </g>

                {/* Right: Cross-encoder */}
                <g transform="translate(285, 0)">
                  <rect x={0} y={0} width={255} height={135} rx={4} fill="var(--compute-dim)" stroke="var(--compute)" strokeWidth={1.2} />
                  <Label x={15} y={20} tone={1} size={10} bold kind="compute">Cross-Encoder (Joint Attention)</Label>
                  <rect x={15} y={32} width={225} height={24} rx={2} fill="var(--ground-2)" stroke="var(--rule-2)" />
                  <Label x={127} y={48} anchor="middle" tone={1} size={8.5}>
                    [CLS] Query tokens [SEP] Document tokens
                  </Label>

                  <Arrow x1={127} y1={56} x2={127} y2={76} kind="compute" width={1.2} />

                  <rect x={15} y={76} width={225} height={26} rx={2} fill="var(--ground-2)" stroke="var(--compute)" strokeWidth={1} />
                  <Label x={127} y={92} anchor="middle" tone={1} size={8.5} kind="compute" bold>
                    Full Self-Attention Across All Token Pairs
                  </Label>

                  <Label x={127} y={120} anchor="middle" tone={2} size={8.5}>
                    Catches negation & precision (Slow, O(L²))
                  </Label>
                </g>
              </g>
            )}

            {s.step >= 4 && (
              <g transform="translate(30, 55)">
                {/* Stage 1: Ingestion pool */}
                <rect x={0} y={15} width={120} height={95} rx={4} fill="var(--ground-3)" stroke="var(--rule-2)" strokeWidth={1} />
                <Label x={60} y={38} anchor="middle" tone={1} size={9.5} bold>1,000,000 Docs</Label>
                <Label x={60} y={54} anchor="middle" tone={3} size={8}>Inverted + ANN</Label>
                <Label x={60} y={78} anchor="middle" tone={2} size={8}>Bi-Encoder + BM25</Label>
                <Label x={60} y={94} anchor="middle" tone={1} size={8} kind="token">~3 ms search</Label>

                <Arrow x1={120} y1={62} x2={175} y2={62} kind="token" width={1.6} />
                <Label x={148} y={52} anchor="middle" tone={3} size={8}>Top 100</Label>

                {/* Stage 2: Reranker pool */}
                <rect x={175} y={25} width={150} height={75} rx={4} fill="var(--compute-dim)" stroke="var(--compute)" strokeWidth={1.2} />
                <Label x={250} y={48} anchor="middle" tone={1} size={9.5} bold kind="compute">Cross-Encoder</Label>
                <Label x={250} y={64} anchor="middle" tone={2} size={8}>Scores 100 pairs</Label>
                <Label x={250} y={82} anchor="middle" tone={1} size={8} kind="compute" bold>~15 ms compute</Label>

                <Arrow x1={325} y1={62} x2={380} y2={62} kind="compute" width={1.6} />
                <Label x={352} y={52} anchor="middle" tone={3} size={8}>Top 5</Label>

                {/* Stage 3: LLM Context */}
                <rect x={380} y={30} width={135} height={65} rx={4} fill="var(--token-dim)" stroke="var(--token)" strokeWidth={1.2} />
                <Label x={447} y={52} anchor="middle" tone={1} size={9.5} bold kind="token">Generator LLM</Label>
                <Label x={447} y={68} anchor="middle" tone={2} size={8}>5 High-Purity Chunks</Label>
                <Label x={447} y={82} anchor="middle" tone={3} size={7.5}>Clean prompt (~1.2k tok)</Label>
              </g>
            )}
          </g>
        )}

        {/* Bottom Takeaway Banner */}
        <g transform="translate(30, 275)">
          <rect x={0} y={0} width={580} height={45} rx={5} fill="var(--ground-2)" stroke="var(--rule-2)" strokeWidth={1} />
          <Label x={15} y={18} tone={1} size={10} bold>The 100-to-5 Golden Ratio:</Label>
          <Label x={15} y={32} tone={2} size={9.5}>
            Never send 100 chunks to your generator LLM. Cast a wide net at Stage 1 (<tspan fill="var(--memory)" fontWeight="600">retrieve 100</tspan>), then filter with a cross-encoder (<tspan fill="var(--token)" fontWeight="600">rerank to 5</tspan>).
          </Label>
        </g>
      </svg>
    </Stage>
  );
}
