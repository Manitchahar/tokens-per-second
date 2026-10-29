import { useRef, useState } from 'react';
import { Stage, Segmented, Readout } from '../lib/Stage';
import { Label, Span, COLOR } from '../lib/svg';

interface QAItem {
  id: number;
  prompt: string;
  reference: string;
  candidateA: string;
  candidateB: string;
  emA: number;
  emB: number;
  embedA: number;
  embedB: number;
  judgePass1: 'A' | 'B';
  judgePass2: 'A' | 'B'; // when B is in position 1
}

const ITEMS: QAItem[] = [
  {
    id: 1,
    prompt: 'What is the default port for PostgreSQL?',
    reference: '5432',
    candidateA: 'The default port is 5432.',
    candidateB: 'PostgreSQL listens on port 5432 by default.',
    emA: 1,
    emB: 1,
    embedA: 0.94,
    embedB: 0.96,
    judgePass1: 'A',
    judgePass2: 'B',
  },
  {
    id: 2,
    prompt: 'How many bytes per token does Llama-3.1-8B use for KV cache in BF16?',
    reference: '131072 bytes (128 KB)',
    candidateA: '128 KiB per token (2 * 32 layers * 8 heads * 128 dim * 2 bytes).',
    candidateB: 'Roughly 16 GB for the model weights.',
    emA: 1,
    emB: 0,
    embedA: 0.92,
    embedB: 0.41,
    judgePass1: 'A',
    judgePass2: 'A',
  },
  {
    id: 3,
    prompt: 'Explain what happens when a CUDA thread encounters a branch divergence.',
    reference: 'Threads in the warp execute different code paths sequentially while inactive threads are masked, reducing compute utilization.',
    candidateA: 'Execution serializes along each path with SIMT masking until the paths reconverge.',
    candidateB: 'The GPU immediately crashes with an out of memory error.',
    emA: 0,
    emB: 0,
    embedA: 0.88,
    embedB: 0.25,
    judgePass1: 'A',
    judgePass2: 'A',
  },
  {
    id: 4,
    prompt: 'What HTTP status code represents Too Many Requests for rate limiting?',
    reference: '429',
    candidateA: '429 Too Many Requests.',
    candidateB: 'HTTP 429.',
    emA: 1,
    emB: 1,
    embedA: 0.98,
    embedB: 0.97,
    judgePass1: 'A',
    judgePass2: 'B',
  },
  {
    id: 5,
    prompt: 'How do you prevent SQL injection when writing database queries in Python?',
    reference: 'Use parameterized queries or prepared statements rather than string concatenation.',
    candidateA: 'Use parameterized queries with bind variables in the database driver.',
    candidateB: 'Sanitize strings by replacing single quotes with empty strings.',
    emA: 0,
    emB: 0,
    embedA: 0.89,
    embedB: 0.58,
    judgePass1: 'A',
    judgePass2: 'A',
  },
];

export default function LabEvalSimulator() {
  const ref = useRef<HTMLDivElement>(null);
  const [selectedId, setSelectedId] = useState(1);
  const [enableSwap, setEnableSwap] = useState(true);

  const item = ITEMS.find((x) => x.id === selectedId) || ITEMS[0];

  // Judge decision with or without position swap
  let judgeWinner: string;
  if (!enableSwap) {
    // Single pass: Pass 1 decision (heavily biased toward position 1 = Candidate A)
    judgeWinner = item.judgePass1 === 'A' ? 'Model A' : 'Model B';
  } else {
    // Bidirectional swap reconciliation
    if (item.judgePass1 === 'A' && item.judgePass2 === 'A') {
      judgeWinner = 'Model A (Robust Win)';
    } else if (item.judgePass1 === 'B' && item.judgePass2 === 'B') {
      judgeWinner = 'Model B (Robust Win)';
    } else {
      judgeWinner = 'Tie (Position Artifact Removed)';
    }
  }

  return (
    <Stage
      rootRef={ref}
      title="Lab Harness Simulator: 3-Way Scoring with Qwen2.5-1.5B Judge"
      steps={['Dataset Inspection', 'Exact Match', 'Embedding Similarity', 'LLM Judge', 'Debiased Reconciliation']}
      step={selectedId - 1}
      captions={[
        'Item 1: Normalized Exact Match succeeds on both, Embedding Similarity is near 0.95, and Position Swap resolves a tie.',
        'Item 2: Model A derives exact 128 KiB; Model B hallucinates weight memory instead of KV cache.',
        'Item 3: Open-ended prose explanation: EM is 0% for both, but Embedding and LLM Judge correctly award victory to Model A.',
        'Item 4: Short HTTP status code: Both models produce correct answer 429; LLM Judge reconciles to a Tie.',
        'Item 5: Security query: Model A recommends parameterized queries; Model B suggests vulnerable regex stripping.',
      ]}
      legend={['token', 'error', 'memory', 'compute']}
      readout={
        <Readout items={[
          { label: 'Active Item', value: `Sample #${item.id} of 5`, kind: 'text' },
          { label: 'Exact Match (EM)', value: `A: ${item.emA} | B: ${item.emB}`, kind: item.emA === item.emB ? 'memory' : 'token', note: 'normalized string check' },
          { label: 'Embedding Cosine', value: `A: ${item.embedA.toFixed(2)} | B: ${item.embedB.toFixed(2)}`, kind: 'compute', note: 'semantic similarity' },
          { label: 'Qwen Judge Verdict', value: judgeWinner, kind: judgeWinner.includes('Robust') ? 'token' : judgeWinner.includes('Tie') ? 'memory' : 'error' },
        ]} />
      }
      controls={
        <>
          <Segmented
            label="Select QA item"
            value={selectedId}
            options={ITEMS.map((it) => ({ value: it.id, label: `Item #${it.id}` }))}
            onChange={(val) => setSelectedId(Number(val))}
          />
          <Segmented
            label="Position swap debiasing"
            value={enableSwap ? 'on' : 'off'}
            options={[
              { value: 'on', label: 'Bidirectional Swap (Debiased)' },
              { value: 'off', label: 'Single Pass (Biased)' },
            ]}
            onChange={(val) => setEnableSwap(val === 'on')}
          />
        </>
      }
    >
      <svg viewBox="0 0 640 330" role="img" aria-label="Interactive demonstration of the laboratory evaluation harness comparing normalized exact match, sentence embedding cosine similarity, and Qwen2.5-1.5B judge scoring with position swapping.">
        {/* Prompt Card */}
        <rect x={24} y={36} width={592} height={46} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
        <Label x={36} y={54} tone={1} size={10} bold>PROMPT #{item.id}:</Label>
        <text x={36} y={72} fill="var(--text)" fontSize={10} fontFamily="var(--font-sans)">
          {item.prompt}
        </text>

        {/* Candidate A Card */}
        <rect x={24} y={92} width={285} height={130} rx={4} fill="var(--surface-1)" stroke={item.emA ? COLOR.token : 'var(--rule-2)'} strokeWidth={item.emA ? 1.5 : 1} />
        <Label x={36} y={112} kind="token" size={10.5} bold>Candidate A Output:</Label>
        <text x={36} y={132} fill="var(--text)" fontSize={9} fontFamily="var(--font-sans)">
          "{item.candidateA}"
        </text>
        <Label x={36} y={178} tone={2} size={9} mono>Exact Match: {item.emA ? '1 (PASS)' : '0 (FAIL)'}</Label>
        <Label x={36} y={196} kind="compute" size={9} mono>Embedding Cosine: {item.embedA.toFixed(2)}</Label>
        <Span x={36} y={204} w={item.embedA * 180} h={6} kind="compute" />

        {/* Candidate B Card */}
        <rect x={331} y={92} width={285} height={130} rx={4} fill="var(--surface-1)" stroke={item.emB ? COLOR.token : 'var(--rule-2)'} strokeWidth={item.emB ? 1.5 : 1} />
        <Label x={343} y={112} kind="memory" size={10.5} bold>Candidate B Output:</Label>
        <text x={343} y={132} fill="var(--text)" fontSize={9} fontFamily="var(--font-sans)">
          "{item.candidateB}"
        </text>
        <Label x={343} y={178} tone={2} size={9} mono>Exact Match: {item.emB ? '1 (PASS)' : '0 (FAIL)'}</Label>
        <Label x={343} y={196} kind="compute" size={9} mono>Embedding Cosine: {item.embedB.toFixed(2)}</Label>
        <Span x={343} y={204} w={item.embedB * 180} h={6} kind="compute" />

        {/* Bottom Judge Result Card */}
        <rect x={24} y={232} width={592} height={60} rx={4} fill="var(--surface-2)" stroke="var(--rule-2)" />
        <Label x={40} y={254} tone={1} size={11} bold>Qwen2.5-1.5B Judge Decision:</Label>
        <text x={40} y={274} fill="var(--text-2)" fontSize={10} fontFamily="var(--font-sans)">
          Pass 1 (A in Pos 1): Voted {item.judgePass1} | Pass 2 (B in Pos 1): Voted {item.judgePass2} →
        </text>
        <Label x={380} y={274} kind={judgeWinner.includes('Robust') ? 'token' : judgeWinner.includes('Tie') ? 'memory' : 'error'} size={11} bold>
          Final Verdict: {judgeWinner}
        </Label>
      </svg>
    </Stage>
  );
}
