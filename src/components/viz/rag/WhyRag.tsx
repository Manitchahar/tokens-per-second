import { useRef, useState } from 'react';
import { Stage, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Arrow, Chip, Fade } from '../lib/svg';

const STEPS = [
  '1. The Cutoff Problem',
  '2. Attempt: Finetuning',
  '3. The Non-Parametric Index',
  '4. Retrieve Relevant Chunks',
  '5. Augment the Prompt',
  '6. Grounded Generation',
  '7. The Serving Trade-Off',
];

const CAPTIONS = [
  'A user query requires facts outside the weights: private company data, real-time status, or events past the training cutoff.',
  'Finetuning forces facts into model weights. It is expensive, slow to update, and causes hallucination when paraphrased.',
  'RAG decouples facts from weights: documents are stored verbatim in an external searchable non-parametric index.',
  'When a query arrives, the retriever runs a fast lookup over the index, pulling only the top-k relevant text passages.',
  'The prompt is augmented: system prompt + retrieved passages + user query are concatenated into the model context.',
  'The generator reads the retrieved chunks during prefill and attends to verified facts, generating cited, grounded answers.',
  'Compare the architectures: RAG trades prompt tokens and retrieval latency for zero weight updates and zero hallucinations.',
];

export default function WhyRag() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [arch, setArch] = useState<'naive' | 'ft' | 'rag'>('rag');

  // Sync tab with step progression if user hasn't explicitly chosen
  const activeMode = s.step <= 0 ? 'naive' : s.step === 1 ? 'ft' : arch;

  return (
    <Stage
      rootRef={ref}
      title="Finetuning vs RAG: Parametric vs External Memory"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['token', 'memory', 'compute', 'error']}
      {...s}
      readout={
        <Readout items={[
          {
            label: 'Knowledge Source',
            value: activeMode === 'naive' ? 'Static Weights' : activeMode === 'ft' ? 'Retrained Weights' : 'External Vector Store',
            kind: activeMode === 'rag' ? 'memory' : 'compute',
            note: activeMode === 'rag' ? 'Non-parametric, live updates' : 'Parametric, frozen at training',
          },
          {
            label: 'Update Cost',
            value: activeMode === 'naive' ? '$0 (Stale)' : activeMode === 'ft' ? '~$50–500 (GPU run)' : '~$0.001 (Embed doc)',
            kind: activeMode === 'ft' ? 'error' : 'memory',
            note: activeMode === 'rag' ? 'O(1) append to database' : activeMode === 'ft' ? 'Gradient descent over corpus' : 'Cannot update',
          },
          {
            label: 'Hallucination Risk',
            value: activeMode === 'rag' ? 'Low (Grounded in text)' : 'High (Probabilistic recall)',
            kind: activeMode === 'rag' ? 'token' : 'error',
            note: activeMode === 'rag' ? 'Attends directly to retrieved context' : 'Guessed from compressed weights',
          },
          {
            label: 'Context Overhead',
            value: activeMode === 'rag' ? '+1,024 to 2,048 tokens' : '0 extra tokens',
            kind: activeMode === 'rag' ? 'compute' : 'token',
            note: activeMode === 'rag' ? 'Increases prefill FLOPs & KV memory' : 'Shortest prompt, but wrong answer',
          },
        ]} />
      }
      controls={
        <Segmented
          label="Architecture View"
          value={activeMode}
          options={[
            { value: 'naive', label: '1. Frozen LLM' },
            { value: 'ft', label: '2. Finetuning' },
            { value: 'rag', label: '3. RAG Pipeline' },
          ]}
          onChange={(v) => setArch(v as 'naive' | 'ft' | 'rag')}
        />
      }
    >
      <svg viewBox="0 0 640 340" role="img" aria-label="Comparison between frozen LLM inference, finetuning knowledge into weights, and retrieval-augmented generation with external vector index.">
        {/* User Query Box */}
        <g transform="translate(30, 40)">
          <rect x={0} y={0} width={130} height={50} rx={6} fill="var(--ground-2)" stroke="var(--rule-2)" strokeWidth={1.2} />
          <Label x={65} y={20} anchor="middle" tone={1} size={11} bold>User Query</Label>
          <Label x={65} y={36} anchor="middle" tone={3} size={9.5}>"What is our Q3 refund policy?"</Label>
        </g>

        {/* MODE 1: NAIVE FROZEN LLM */}
        {activeMode === 'naive' && (
          <g>
            <Arrow x1={160} y1={65} x2={250} y2={65} kind="token" />
            {/* Frozen Model Box */}
            <g transform="translate(250, 30)">
              <rect x={0} y={0} width={160} height={70} rx={6} fill="var(--ground-2)" stroke="var(--rule-2)" strokeWidth={1.2} />
              <Label x={80} y={24} anchor="middle" tone={1} size={12} bold>Frozen LLM</Label>
              <Label x={80} y={42} anchor="middle" tone={3} size={10}>Knowledge cutoff: 2023</Label>
              <Label x={80} y={58} anchor="middle" tone={3} size={9}>No access to private docs</Label>
            </g>

            <Arrow x1={410} y1={65} x2={480} y2={65} kind="error" dashed />

            {/* Hallucinated Output */}
            <g transform="translate(480, 35)">
              <rect x={0} y={0} width={135} height={60} rx={6} fill="var(--error-dim)" stroke="var(--error)" strokeWidth={1.2} />
              <Label x={67} y={22} anchor="middle" tone={1} size={11} bold kind="error">Hallucinated Output</Label>
              <Label x={67} y={38} anchor="middle" tone={2} size={9}>"Refunds take 90 days..."</Label>
              <Label x={67} y={50} anchor="middle" tone={3} size={8.5}>(Factually fabricated)</Label>
            </g>
          </g>
        )}

        {/* MODE 2: FINETUNING */}
        {activeMode === 'ft' && (
          <g>
            {/* Training data flow */}
            <g transform="translate(30, 140)">
              <rect x={0} y={0} width={130} height={60} rx={6} fill="var(--ground-2)" stroke="var(--compute)" strokeWidth={1} />
              <Label x={65} y={22} anchor="middle" tone={1} size={10.5} bold kind="compute">10M Token Corpus</Label>
              <Label x={65} y={38} anchor="middle" tone={3} size={9}>Company Wiki & Policies</Label>
              <Label x={65} y={50} anchor="middle" tone={3} size={8.5}>Updates daily</Label>
            </g>

            <Arrow x1={160} y1={170} x2={250} y2={170} kind="compute" />

            {/* Backprop / Weight Update */}
            <g transform="translate(250, 130)">
              <rect x={0} y={0} width={160} height={80} rx={6} fill="var(--compute-dim)" stroke="var(--compute)" strokeWidth={1.4} />
              <Label x={80} y={24} anchor="middle" tone={1} size={11} bold kind="compute">LoRA / SFT Training</Label>
              <Label x={80} y={42} anchor="middle" tone={2} size={9.5}>$4.8 \times 10^{'{'}17{'}'}$ FLOPs</Label>
              <Label x={80} y={58} anchor="middle" tone={3} size={9}>Hours on 8× A100 GPUs</Label>
              <Label x={80} y={72} anchor="middle" tone={3} size={8.5} kind="error">Catastrophic forgetting risk</Label>
            </g>

            <Arrow x1={330} y1={130} x2={330} y2={100} kind="compute" dashed />

            {/* Model box */}
            <g transform="translate(250, 30)">
              <rect x={0} y={0} width={160} height={70} rx={6} fill="var(--ground-2)" stroke="var(--rule-2)" strokeWidth={1.2} />
              <Label x={80} y={24} anchor="middle" tone={1} size={12} bold>Finetuned LLM</Label>
              <Label x={80} y={42} anchor="middle" tone={3} size={10}>Facts baked into weights</Label>
              <Label x={80} y={58} anchor="middle" tone={3} size={9} kind="error">Static until next retrain</Label>
            </g>

            <Arrow x1={160} y1={65} x2={250} y2={65} kind="token" />
            <Arrow x1={410} y1={65} x2={480} y2={65} kind="token" />

            {/* Weakly recalled output */}
            <g transform="translate(480, 35)">
              <rect x={0} y={0} width={135} height={60} rx={6} fill="var(--ground-2)" stroke="var(--rule-2)" strokeWidth={1.2} />
              <Label x={67} y={22} anchor="middle" tone={1} size={11} bold>Stale / Unverified</Label>
              <Label x={67} y={38} anchor="middle" tone={3} size={9}>Cannot cite exact clause</Label>
              <Label x={67} y={50} anchor="middle" tone={3} size={8.5}>Changes tomorrow = invalid</Label>
            </g>
          </g>
        )}

        {/* MODE 3: RAG PIPELINE */}
        {activeMode === 'rag' && (
          <g>
            {/* Step 3: Non-parametric index */}
            <g transform="translate(190, 160)">
              <rect x={0} y={0} width={150} height={90} rx={6} fill="var(--memory-dim)" stroke="var(--memory)" strokeWidth={1.4} />
              <Label x={75} y={22} anchor="middle" tone={1} size={11} bold kind="memory">Non-Parametric Index</Label>
              <Label x={75} y={38} anchor="middle" tone={2} size={9}>Chunks + Vectors</Label>
              <line x1={15} y1={46} x2={135} y2={46} stroke="var(--rule-2)" strokeWidth={0.8} />
              <Label x={75} y={60} anchor="middle" tone={3} size={8.5}>Doc 402: "Q3 Refund Policy..."</Label>
              <Label x={75} y={74} anchor="middle" tone={3} size={8.5}>Doc 819: "Return SLA 14 days"</Label>
              <Label x={75} y={86} anchor="middle" tone={3} size={8} kind="token">Instant updates: O(1)</Label>
            </g>

            {/* Query to retriever arrow */}
            <path d="M 95 90 L 95 205 L 190 205" fill="none" stroke="var(--token)" strokeWidth={1.4} markerEnd="url(#tps-arrow)" />
            <Label x={135} y={195} tone={3} size={9} anchor="middle">Search</Label>

            {/* Retrieved chunks to context */}
            <path d="M 340 205 L 390 205 L 390 145" fill="none" stroke="var(--memory)" strokeWidth={1.4} markerEnd="url(#tps-arrow)" />
            <Label x={365} y={195} tone={2} size={9} anchor="middle" kind="memory">Top-k Chunks</Label>

            {/* Prompt Assembly (Context window) */}
            <g transform="translate(320, 30)">
              <rect x={0} y={0} width={140} height={110} rx={6} fill="var(--ground-2)" stroke="var(--rule-2)" strokeWidth={1.2} />
              <Label x={70} y={20} anchor="middle" tone={1} size={11} bold>Prompt Assembly</Label>
              <rect x={10} y={30} width={120} height={20} rx={3} fill="var(--ground-3)" />
              <Label x={70} y={44} anchor="middle" tone={3} size={8.5}>System: You are an assistant</Label>
              <rect x={10} y={54} width={120} height={26} rx={3} fill="var(--memory-dim)" stroke="var(--memory)" strokeWidth={0.8} />
              <Label x={70} y={66} anchor="middle" tone={1} size={8.5} kind="memory" bold>Context: Doc 402 + 819</Label>
              <Label x={70} y={76} anchor="middle" tone={3} size={7.5}>"Refunds issued within 14 days"</Label>
              <rect x={10} y={84} width={120} height={18} rx={3} fill="var(--token-dim)" stroke="var(--token)" strokeWidth={0.8} />
              <Label x={70} y={96} anchor="middle" tone={1} size={8.5} kind="token">Query: "What is refund policy?"</Label>
            </g>

            {/* Arrow from query box to prompt assembly */}
            <path d="M 160 65 L 320 65" fill="none" stroke="var(--token)" strokeWidth={1.4} markerEnd="url(#tps-arrow)" strokeDasharray="3 3" />

            {/* Arrow from prompt assembly to Generator */}
            <Arrow x1={460} y1={85} x2={500} y2={85} kind="compute" />

            {/* Generator LLM Box */}
            <g transform="translate(500, 45)">
              <rect x={0} y={0} width={120} height={80} rx={6} fill="var(--ground-2)" stroke="var(--compute)" strokeWidth={1.4} />
              <Label x={60} y={22} anchor="middle" tone={1} size={11} bold kind="compute">Generator LLM</Label>
              <Label x={60} y={38} anchor="middle" tone={3} size={8.5}>General Pretrained</Label>
              <Label x={60} y={52} anchor="middle" tone={2} size={8.5}>Attends to Context</Label>
              <line x1={10} y1={58} x2={110} y2={58} stroke="var(--rule-2)" strokeWidth={0.6} />
              <Label x={60} y={70} anchor="middle" tone={1} size={8.5} kind="token" bold>Cites [Doc 402]</Label>
            </g>
          </g>
        )}

        {/* Bottom Comparative Summary Banner */}
        <g transform="translate(30, 275)">
          <rect x={0} y={0} width={580} height={50} rx={5} fill="var(--ground-2)" stroke="var(--rule-2)" strokeWidth={1} />
          <Label x={15} y={20} tone={1} size={10} bold>Core Engineering Principle:</Label>
          <Label x={15} y={36} tone={2} size={9.5}>
            Finetuning alters the <tspan fill="var(--compute)" fontWeight="600">form</tspan> (style, syntax, task reasoning). Retrieval supplies the <tspan fill="var(--memory)" fontWeight="600">facts</tspan> (dynamic, verifiable, zero retrain).
          </Label>
          <Label x={565} y={28} anchor="end" tone={3} size={9}>Ovadia et al. (2023)</Label>
        </g>
      </svg>
    </Stage>
  );
}
