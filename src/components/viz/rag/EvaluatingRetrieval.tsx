import { useRef, useState } from 'react';
import { Stage, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Arrow } from '../lib/svg';

const STEPS = [
  '1. Two Different Failure Modes',
  '2. Binary Metrics: Precision@k & Recall@k',
  '3. Rank Matters: Lost in the Middle',
  '4. Mean Reciprocal Rank (MRR)',
  '5. Discounted Cumulative Gain (DCG)',
  '6. Ideal Ranking & NDCG Normalization',
  '7. The RAG Triad: End-to-End Quality',
];

const CAPTIONS = [
  'When a RAG system fails, isolate the cause: did the retriever fail to find the fact, or did the generator fail to read it?',
  'Precision@k measures what fraction of retrieved chunks are relevant; Recall@k measures what fraction of total relevant chunks were retrieved.',
  'Unranked retrieval is not enough. Transformers pay attention to position; relevant chunks buried at rank 15 suffer from context degradation.',
  'Mean Reciprocal Rank (MRR) evaluates the position of the first relevant document: 1 / rank1. Rank 1 is 1.0, Rank 2 is 0.5, Rank 4 is 0.25.',
  'Discounted Cumulative Gain (DCG) supports graded relevance (0 to 3), heavily discounting documents that appear low in the list.',
  'NDCG normalizes DCG by the Ideal DCG (IDCG), producing a score between 0.0 and 1.0. An NDCG of 1.0 represents a flawless ranking.',
  'The RAG Triad measures end-to-end production health: Context Relevance (retriever), Faithfulness, and Answer Relevance (generator).',
];

export default function EvaluatingRetrieval() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);

  // 5 retrieved documents with relevance grades (0 to 3)
  const [grades, setGrades] = useState<[number, number, number, number, number]>([3, 0, 2, 1, 0]);
  const totalRelevantInCorpus = 4; // Assume 4 true relevant docs exist

  // Calculate Precision@5 & Recall@5 (Binary: grade >= 1 is relevant)
  const relevantRetrieved = grades.filter((g) => g >= 1).length;
  const precision5 = relevantRetrieved / 5;
  const recall5 = relevantRetrieved / totalRelevantInCorpus;

  // Calculate MRR (first item with grade >= 1)
  const firstRelIndex = grades.findIndex((g) => g >= 1);
  const mrr = firstRelIndex >= 0 ? 1 / (firstRelIndex + 1) : 0;

  // Calculate DCG@5: sum (2^rel - 1) / log2(i + 1) for i = 1..5
  const discounts = [
    Math.log2(2), // i=1 -> log2(2) = 1.0
    Math.log2(3), // i=2 -> 1.58496
    Math.log2(4), // i=3 -> 2.0
    Math.log2(5), // i=4 -> 2.32193
    Math.log2(6), // i=5 -> 2.58496
  ];
  let dcg = 0;
  for (let i = 0; i < 5; i++) {
    const gain = Math.pow(2, grades[i]) - 1;
    dcg += gain / discounts[i];
  }

  // Calculate IDCG@5: sort grades descending
  const sortedGrades = [...grades].sort((a, b) => b - a);
  let idcg = 0;
  for (let i = 0; i < 5; i++) {
    const gain = Math.pow(2, sortedGrades[i]) - 1;
    idcg += gain / discounts[i];
  }
  const ndcg = idcg > 0 ? dcg / idcg : 1.0;

  return (
    <Stage
      rootRef={ref}
      title="Evaluating Retrieval: Precision, Recall, MRR & NDCG"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['token', 'memory', 'compute', 'error']}
      {...s}
      readout={
        <Readout items={[
          {
            label: 'Precision@5',
            value: `${(precision5 * 100).toFixed(0)}%`,
            kind: precision5 >= 0.6 ? 'token' : 'error',
            note: `${relevantRetrieved} relevant of 5 retrieved chunks`,
          },
          {
            label: 'Recall@5',
            value: `${(recall5 * 100).toFixed(0)}%`,
            kind: recall5 >= 0.75 ? 'token' : 'compute',
            note: `${relevantRetrieved} found of ${totalRelevantInCorpus} total relevant in corpus`,
          },
          {
            label: 'Reciprocal Rank (RR)',
            value: mrr.toFixed(3),
            kind: mrr >= 0.5 ? 'token' : 'error',
            note: firstRelIndex >= 0 ? `First relevant at Rank ${firstRelIndex + 1} (1/${firstRelIndex + 1})` : 'No relevant found',
          },
          {
            label: 'NDCG@5',
            value: ndcg.toFixed(3),
            kind: ndcg >= 0.85 ? 'token' : 'compute',
            note: `DCG: ${dcg.toFixed(2)} / IDCG: ${idcg.toFixed(2)}`,
          },
        ]} />
      }
      controls={
        <Segmented
          label="Preset Retrieval Ranking"
          value={grades.join(',')}
          options={[
            { value: '3,0,2,1,0', label: 'Preset A: [3, 0, 2, 1, 0]' },
            { value: '0,0,3,2,1', label: 'Preset B: [0, 0, 3, 2, 1] (Delayed)' },
            { value: '3,2,1,0,0', label: 'Preset C: [3, 2, 1, 0, 0] (Ideal)' },
          ]}
          onChange={(v) => {
            const arr = v.split(',').map(Number) as [number, number, number, number, number];
            setGrades(arr);
          }}
        />
      }
    >
      <svg viewBox="0 0 640 330" role="img" aria-label="Visual demonstration of Precision@k, Recall@k, MRR reciprocal rank, and NDCG discounted cumulative gain calculations.">
        {/* Left Side: Ranked Results with Interactive Grades */}
        <g transform="translate(30, 25)">
          <rect x={0} y={0} width={280} height={235} rx={6} fill="var(--ground-2)" stroke="var(--rule-2)" strokeWidth={1} />
          <Label x={16} y={22} tone={1} size={11} bold>Retrieved Top-5 Candidate Chunks</Label>
          <Label x={16} y={38} tone={3} size={9}>Ranked by initial retriever score</Label>

          <g transform="translate(16, 50)">
            {grades.map((grade, idx) => {
              const rank = idx + 1;
              const isFirstHit = rank === firstRelIndex + 1;
              const gain = Math.pow(2, grade) - 1;
              const discount = discounts[idx];
              const discountedGain = gain / discount;
              const isRel = grade >= 1;

              return (
                <g key={idx} transform={`translate(0, ${idx * 34})`}>
                  <rect
                    x={0}
                    y={0}
                    width={248}
                    height={28}
                    rx={3}
                    fill={isRel ? (grade === 3 ? 'var(--token-dim)' : 'var(--memory-dim)') : 'var(--ground-3)'}
                    stroke={isRel ? (grade === 3 ? 'var(--token)' : 'var(--memory)') : 'var(--rule-2)'}
                    strokeWidth={isFirstHit ? 1.4 : 0.8}
                  />
                  <Label x={10} y={18} tone={1} size={9} bold kind={isFirstHit ? 'token' : undefined}>
                    #{rank}
                  </Label>
                  <Label x={35} y={18} tone={2} size={8.5}>
                    Doc {String.fromCharCode(65 + idx)} (Grade: {grade}/3)
                  </Label>

                  <Label x={180} y={18} tone={3} size={8}>
                    +{discountedGain.toFixed(2)} DCG
                  </Label>

                  {isFirstHit && (
                    <circle cx={235} cy={14} r={3} fill="var(--token)" />
                  )}
                </g>
              );
            })}
          </g>

          <Label x={16} y={224} tone={3} size={8.5}>
            ● First relevant hit (defines Reciprocal Rank)
          </Label>
        </g>

        {/* Right Side: Step-by-Step Metric Formulas */}
        <g transform="translate(325, 25)">
          <rect x={0} y={0} width={285} height={235} rx={6} fill="var(--ground-2)" stroke="var(--rule-2)" strokeWidth={1} />
          <Label x={16} y={22} tone={1} size={11} bold>Metric Formulas & Breakdown</Label>
          <Label x={16} y={38} tone={3} size={9}>Exact numeric derivations for this ranking</Label>

          {/* Precision & Recall Box */}
          <g transform="translate(16, 50)">
            <rect x={0} y={0} width={253} height={42} rx={3} fill="var(--ground-3)" />
            <Label x={10} y={16} tone={2} size={8.5} bold>Precision@5 & Recall@5:</Label>
            <Label x={10} y={32} tone={1} size={8.5} kind="token">
              P@5 = {relevantRetrieved}/5 = {(precision5 * 100).toFixed(0)}%  |  R@5 = {relevantRetrieved}/{totalRelevantInCorpus} = {(recall5 * 100).toFixed(0)}%
            </Label>
          </g>

          {/* MRR Box */}
          <g transform="translate(16, 100)">
            <rect x={0} y={0} width={253} height={42} rx={3} fill="var(--ground-3)" />
            <Label x={10} y={16} tone={2} size={8.5} bold>Mean Reciprocal Rank (MRR):</Label>
            <Label x={10} y={32} tone={1} size={8.5} kind="token">
              RR = 1 / Rank₁ = 1 / {firstRelIndex + 1} = {mrr.toFixed(3)}
            </Label>
          </g>

          {/* NDCG Box */}
          <g transform="translate(16, 150)">
            <rect x={0} y={0} width={253} height={68} rx={3} fill="var(--ground-3)" />
            <Label x={10} y={16} tone={2} size={8.5} bold>NDCG@5 (Normalized DCG):</Label>
            <Label x={10} y={32} tone={3} size={8}>
              DCG = Σ (2^rel - 1) / log₂(i + 1) = {dcg.toFixed(3)}
            </Label>
            <Label x={10} y={46} tone={3} size={8}>
              IDCG (Ideal: [{sortedGrades.join(',')}]) = {idcg.toFixed(3)}
            </Label>
            <Label x={10} y={60} tone={1} size={9} bold kind="token">
              NDCG@5 = {dcg.toFixed(2)} / {idcg.toFixed(2)} = {ndcg.toFixed(3)}
            </Label>
          </g>
        </g>

        {/* Bottom Takeaway */}
        <g transform="translate(30, 275)">
          <rect x={0} y={0} width={580} height={45} rx={5} fill="var(--ground-2)" stroke="var(--rule-2)" strokeWidth={1} />
          <Label x={15} y={18} tone={1} size={10} bold>The Evaluation Rule:</Label>
          <Label x={15} y={32} tone={2} size={9.5}>
            Evaluate the <tspan fill="var(--memory)" fontWeight="600">retriever</tspan> with NDCG and Recall@k; evaluate the <tspan fill="var(--compute)" fontWeight="600">generator</tspan> with Faithfulness and Answer Relevance. Never bundle them into a single opaque test.
          </Label>
        </g>
      </svg>
    </Stage>
  );
}
