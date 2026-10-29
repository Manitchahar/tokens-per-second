import { useRef, useState } from 'react';
import { Stage, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Span, Label, Fade } from '../lib/svg';

interface StrategyMetrics {
  name: string;
  setupCost: number; // 0-100
  queryCost: number; // 0-100 (tokens/latency)
  factuality: number; // 0-100
  formAdherence: number; // 0-100
  summary: string;
}

const STRATEGIES: StrategyMetrics[] = [
  {
    name: 'Prompting',
    setupCost: 5,
    queryCost: 20,
    factuality: 40,
    formAdherence: 45,
    summary: 'Zero training cost. High variance on strict syntax; prone to hallucinating niche facts.',
  },
  {
    name: 'Few-Shot ICL',
    setupCost: 15,
    queryCost: 65,
    factuality: 55,
    formAdherence: 75,
    summary: 'Burns 500–2,000 prompt tokens per call. Increases TTFT latency; format improves but eats KV cache.',
  },
  {
    name: 'RAG',
    setupCost: 35,
    queryCost: 70,
    factuality: 95,
    formAdherence: 60,
    summary: 'Best for dynamic ground truth. Perfect citation & auditability; prompt overhead remains high.',
  },
  {
    name: 'Finetuning',
    setupCost: 80,
    queryCost: 15,
    factuality: 35,
    formAdherence: 98,
    summary: 'Bakes syntax & tone directly into weights. Zero added prompt tokens, ultra-fast TTFT; bad for volatile facts.',
  },
];

const SCENARIOS = [
  {
    title: 'Dynamic Docs / FAQs',
    rec: 'RAG',
    why: 'Docs change weekly. Finetuning weights cannot guarantee zero hallucinations or exact citations.',
  },
  {
    title: 'High-Volume JSON Extractor',
    rec: 'Finetuning (LoRA)',
    why: 'Save millions of prompt schema tokens. 100% valid JSON output at 1/5th the serving latency.',
  },
  {
    title: 'Domain Persona / Medical Tone',
    rec: 'Finetuning (SFT)',
    why: 'Bake empathetic, concise triage behavior into weights without burning system prompt context.',
  },
  {
    title: 'Distillation: 70B → 1.5B',
    rec: 'Finetuning',
    why: 'Transfer teacher reasoning into an edge-deployable small model for 10× cheaper throughput.',
  },
];

const STEPS = [
  'Level 1: System prompt & zero-shot',
  'Level 2: Few-shot in-context demonstrations',
  'Level 3: Retrieval-Augmented Generation (RAG)',
  'The pitfall: The knowledge injection fallacy',
  'Level 4: Finetuning (SFT / LoRA)',
  'The complete decision matrix',
];

const CAPTIONS = [
  'Always start here. Modify instructions in plain text. Zero training cost, immediate iterations, but reliability degrades on complex domain schemas.',
  'Include 3–5 input-output examples in context. Format adherence jumps from 45% to 75%, but every request now carries 1,000+ prompt tokens into the KV cache.',
  'When knowledge changes or citations are required, retrieve factual chunks at runtime. Factuality hits 95%+, but prompt length and prefill compute remain high.',
  'Pitfall: retraining weights to memorize facts. Backpropagation is sample-inefficient, suffers catastrophic forgetting, and still hallucinates obscure details.',
  'Finetuning changes the weights to master form, style, and syntax. System prompt tokens drop to zero, TTFT plummets, and schema validity reaches 98%+.',
  'The golden rule: Finetune for form, behaviour, and distillation; retrieve for volatile facts. Select scenarios below to inspect the trade-off profile.',
];

export default function WhenToFinetuneViz() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;

  const [activeScenario, setActiveScenario] = useState<number>(0);

  // Active highlighted strategy depending on step or scenario
  const activeStrategyIdx =
    st === 0 ? 0 : st === 1 ? 1 : st === 2 ? 2 : st === 3 ? 2 : st === 4 ? 3 : activeScenario;

  const activeStrategy = STRATEGIES[Math.min(activeStrategyIdx, STRATEGIES.length - 1)];
  const scen = SCENARIOS[activeScenario];

  return (
    <Stage
      rootRef={ref}
      title="The Customization Ladder: Prompt vs RAG vs Finetune"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['compute', 'setup / training cost'],
        ['memory', 'query prompt tokens & TTFT'],
        ['token', 'form / syntax adherence'],
        ['error', 'hallucination risk on facts'],
      ]}
      controls={
        st === 5 ? (
          <Segmented
            options={SCENARIOS.map((sc) => sc.title)}
            value={SCENARIOS[activeScenario].title}
            onChange={(val) => {
              const idx = SCENARIOS.findIndex((s) => s.title === val);
              if (idx !== -1) setActiveScenario(idx);
            }}
          />
        ) : undefined
      }
      readout={
        st === 5 ? (
          <Readout
            items={[
              { label: 'Recommended', value: scen.rec },
              { label: 'Rationale', value: scen.why },
            ]}
          />
        ) : (
          <Readout
            items={[
              { label: 'Strategy', value: activeStrategy.name },
              { label: 'Key Tradeoff', value: activeStrategy.summary },
            ]}
          />
        )
      }
      {...s}
    >
      <svg viewBox="0 0 640 320" role="img" aria-label="Tradeoff matrix comparing Prompting, Few-Shot, RAG, and Finetuning">
        {/* Background track & ladder columns */}
        <g className="ladder-columns">
          {STRATEGIES.map((strat, idx) => {
            const colX = 24 + idx * 150;
            const isSelected =
              st === 5
                ? strat.name.toLowerCase().includes(scen.rec.toLowerCase().slice(0, 4))
                : activeStrategyIdx === idx;
            const isFaded = st < 5 && activeStrategyIdx !== idx;

            return (
              <g key={strat.name} style={{ opacity: isFaded ? 0.35 : 1, transition: 'opacity 0.25s ease' }}>
                {/* Column Card Frame */}
                <rect
                  x={colX}
                  y={24}
                  width={142}
                  height={150}
                  rx={6}
                  fill={isSelected ? 'rgba(255, 255, 255, 0.05)' : 'rgba(255, 255, 255, 0.015)'}
                  stroke={isSelected ? 'var(--text-1)' : 'var(--rule-2)'}
                  strokeWidth={isSelected ? 1.5 : 1}
                />

                {/* Strategy Title */}
                <Label x={colX + 12} y={45} tone={isSelected ? 1 : 2} bold size={13} mono={false}>
                  {strat.name}
                </Label>

                {/* Subtitle tag */}
                <Label x={colX + 12} y={60} tone={3} size={9.5} mono>
                  {idx === 0
                    ? '0 extra tokens'
                    : idx === 1
                    ? '+1k tokens'
                    : idx === 2
                    ? '+2k ctx tokens'
                    : 'weights updated'}
                </Label>

                {/* Bar Metrics */}
                {/* 1. Setup cost */}
                <Label x={colX + 12} y={80} tone={3} size={8.5} mono>
                  Setup FLOPs
                </Label>
                <Span
                  x={colX + 12}
                  y={84}
                  w={(strat.setupCost / 100) * 118}
                  h={6}
                  kind="compute"
                  dim={!isSelected}
                />

                {/* 2. Query overhead */}
                <Label x={colX + 12} y={104} tone={3} size={8.5} mono>
                  Prompt Overhead (TTFT)
                </Label>
                <Span
                  x={colX + 12}
                  y={108}
                  w={(strat.queryCost / 100) * 118}
                  h={6}
                  kind="memory"
                  dim={!isSelected}
                />

                {/* 3. Form reliability */}
                <Label x={colX + 12} y={128} tone={3} size={8.5} mono>
                  Format Adherence
                </Label>
                <Span
                  x={colX + 12}
                  y={132}
                  w={(strat.formAdherence / 100) * 118}
                  h={6}
                  kind="token"
                  dim={!isSelected}
                />

                {/* 4. Factual grounding */}
                <Label x={colX + 12} y={152} tone={3} size={8.5} mono>
                  Fact Grounding
                </Label>
                <Span
                  x={colX + 12}
                  y={156}
                  w={(strat.factuality / 100) * 118}
                  h={6}
                  kind={strat.factuality < 50 ? 'error' : 'token'}
                  dim={!isSelected}
                />
              </g>
            );
          })}
        </g>

        {/* Lower Detail Panel / Callout */}
        <rect
          x={24}
          y={186}
          width={592}
          height={118}
          rx={6}
          fill="rgba(0, 0, 0, 0.25)"
          stroke="var(--rule-2)"
          strokeWidth={1}
        />

        <Fade show={st === 0}>
          <Label x={42} y={212} tone={1} bold size={13} mono={false}>
            Stage 1 · Prompt Engineering: The Free Baseline
          </Label>
          <Label x={42} y={234} tone={2} size={11} mono={false}>
            - Cost: \$0 compute, 0 training infrastructure, instant feedback.
          </Label>
          <Label x={42} y={252} tone={2} size={11} mono={false}>
            - Limitation: Schema compliance drifts (5–15% JSON parse errors under high temperature).
          </Label>
          <Label x={42} y={270} tone={3} size={10.5} mono={false}>
            Always start here. Never train a model until you have hit the ceiling of prompt engineering.
          </Label>
        </Fade>

        <Fade show={st === 1}>
          <Label x={42} y={212} tone={1} bold size={13} mono={false}>
            Stage 2 · Few-Shot In-Context Learning (ICL)
          </Label>
          <Label x={42} y={234} tone={2} size={11} mono={false}>
            - Mechanism: Supply 3 to 10 golden input-output pairs inside the prompt.
          </Label>
          <Label x={42} y={252} tone={2} size={11} mono={false}>
            - Cost: Linear memory penalty! 1,000 extra prompt tokens = ~16 ms added TTFT + 16 MB KV cache.
          </Label>
          <Label x={42} y={270} tone={3} size={10.5} mono={false}>
            Great for low-volume prototypes. Expensive at 10M requests/day due to compounding token fees.
          </Label>
        </Fade>

        <Fade show={st === 2}>
          <Label x={42} y={212} tone={1} bold size={13} mono={false}>
            Stage 3 · Retrieval-Augmented Generation (RAG)
          </Label>
          <Label x={42} y={234} tone={2} size={11} mono={false}>
            - Mechanism: Non-parametric lookup (BM25 + FAISS) inserts authoritative snippets at runtime.
          </Label>
          <Label x={42} y={252} tone={2} size={11} mono={false}>
            - Factuality: 95%+ factual precision, verifiable source URLs, zero retraining when docs update.
          </Label>
          <Label x={42} y={270} tone={3} size={10.5} mono={false}>
            Rule: If your documents change faster than your training cycle, RAG is strictly mandatory.
          </Label>
        </Fade>

        <Fade show={st === 3}>
          <Label x={42} y={212} kind="error" bold size={13} mono={false}>
            Warning · The Knowledge Injection Fallacy
          </Label>
          <Label x={42} y={234} tone={1} size={11} mono={false}>
            - Trying to teach an LLM internal corporate facts via finetuning fails reliably.
          </Label>
          <Label x={42} y={252} tone={2} size={11} mono={false}>
            - Ovadia et al. (2023): SFT recall on factual questions drops 30–50% behind simple RAG.
          </Label>
          <Label x={42} y={270} tone={3} size={10.5} mono={false}>
            Backprop treats all facts as fuzzy statistical priors. Weight updates are not a database.
          </Label>
        </Fade>

        <Fade show={st === 4}>
          <Label x={42} y={212} tone={1} bold size={13} mono={false}>
            Stage 4 · Finetuning (SFT / LoRA): Mastery of Form
          </Label>
          <Label x={42} y={234} tone={2} size={11} mono={false}>
            - Mechanism: Gradient descent adapts weights $W + \Delta W$ for specific format or persona.
          </Label>
          <Label x={42} y={252} tone={2} size={11} mono={false}>
            - Serving efficiency: System prompt shrinks to 10 tokens! Minimal TTFT and minimal KV cache.
          </Label>
          <Label x={42} y={270} tone={3} size={10.5} mono={false}>
            Perfect for strict JSON schemas, SQL generation, domain tone, and distilling 70B down to 1.5B.
          </Label>
        </Fade>

        <Fade show={st === 5}>
          <Label x={42} y={212} tone={1} bold size={13} mono={false}>
            The Engineering Synthesis: Form vs Facts
          </Label>
          <Label x={42} y={234} tone={2} size={11} mono={false}>
            - Active Scenario: <tspan style={{ fill: 'var(--text-1)', fontWeight: 600 }}>{scen.title}</tspan>
          </Label>
          <Label x={42} y={252} tone={2} size={11} mono={false}>
            - Decision: Use <tspan style={{ fill: 'var(--token)', fontWeight: 600 }}>{scen.rec}</tspan>. {scen.why}
          </Label>
          <Label x={42} y={270} tone={3} size={10.5} mono={false}>
            Toggle the scenario buttons above to explore how production teams choose between RAG and Finetuning.
          </Label>
        </Fade>
      </svg>
    </Stage>
  );
}
