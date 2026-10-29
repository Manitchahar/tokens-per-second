import { useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Arrow, Fade, Span, Chip, COLOR } from '../lib/svg';

const STEPS = [
  'The planning dilemma: Blind execution vs deliberation',
  'ReAct: Interleaving reasoning and acting',
  'Plan-then-execute: Decoupling planner and executor',
  'DAG execution: Parallel tool dispatch',
  'Reflexion: Episodic verbal self-correction',
  'Systems comparison: Wall-clock time and token cost',
];

const CAPTIONS = [
  'Without planning, an agent acts blindly. Chain-of-thought reasons without grounding; pure action acts without strategic synthesis. Effective agents require deliberate planning structures.',
  'ReAct interleaves Thought, Action, and Observation in a strict sequence. Each thought synthesizes the last observation and chooses the next action, adapting dynamically to unexpected results.',
  'Plan-then-execute separates the architect from the builder. A large planner model generates a structured subtask DAG once; lightweight executor models or deterministic functions carry them out.',
  'Independent subtasks in a DAG run concurrently. Instead of waiting for 4 sequential tool turns, parallel dispatch collapses wall-clock time from 4x to 1x.',
  'Reflexion adds an episodic memory buffer across trials. When an execution fails, a self-reflection pass diagnoses the root cause, injecting verbal critiques into the next trial.',
  'Systems trade-off: ReAct offers highest adaptability at high sequential latency. Plan-then-execute offers lowest wall-clock time via parallelism. Reflexion offers maximum accuracy at 3-5x token cost.',
];

export default function PlanningStrategiesViz() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;

  // Selected strategy for interactive inspection
  const [strategy, setStrategy] = useState<'react' | 'plan_exec' | 'reflexion'>('react');

  // Active strategy depending on step:
  // Step 1: ReAct
  // Step 2 & 3: Plan-then-Execute
  // Step 4: Reflexion
  // Step 5: Interactive comparison
  const activeStrat = st === 1 ? 'react' : st === 2 || st === 3 ? 'plan_exec' : st === 4 ? 'reflexion' : strategy;

  return (
    <Stage
      rootRef={ref}
      title="Agent Planning Architectures: ReAct, Plan-then-Execute & Reflexion"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['compute', 'thought / planning'],
        ['token', 'action / tool call'],
        ['comm', 'observation / env'],
        ['memory', 'reflexion / state'],
      ]}
      readout={
        <Readout
          items={[
            {
              label: 'Architecture',
              value: activeStrat === 'react' ? 'ReAct' : activeStrat === 'plan_exec' ? 'Plan-Execute' : 'Reflexion',
              kind: activeStrat === 'react' ? 'compute' : activeStrat === 'plan_exec' ? 'token' : 'memory',
              note: activeStrat === 'react' ? 'Yao et al. 2022' : activeStrat === 'plan_exec' ? 'DAG Dispatch' : 'Shinn et al. 2023',
            },
            {
              label: 'Execution Style',
              value: activeStrat === 'react' ? 'Sequential' : activeStrat === 'plan_exec' ? 'Parallel DAG' : 'Multi-Episode',
              kind: 'text',
              note: activeStrat === 'plan_exec' ? 'Batched concurrency' : 'Interleaved turns',
            },
            {
              label: 'Wall-Clock Latency',
              value: activeStrat === 'react' ? '6.8s (High)' : activeStrat === 'plan_exec' ? '2.1s (Low)' : '18.4s (Very High)',
              kind: activeStrat === 'plan_exec' ? 'token' : activeStrat === 'react' ? 'compute' : 'error',
              note: 'Normalized 4-subtask job',
            },
            {
              label: 'Token Volume',
              value: activeStrat === 'react' ? '4,800 tokens' : activeStrat === 'plan_exec' ? '2,900 tokens' : '14,200 tokens',
              kind: activeStrat === 'reflexion' ? 'error' : 'memory',
              note: 'Prompt + generation tokens',
            },
          ]}
        />
      }
      controls={
        st === 0 || st === 5 ? (
          <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
            <span style={{ fontSize: 11, color: 'var(--text-2)' }}>Compare Architecture:</span>
            {[
              { id: 'react', label: 'ReAct' },
              { id: 'plan_exec', label: 'Plan-then-Execute' },
              { id: 'reflexion', label: 'Reflexion' },
            ].map((item) => (
              <button
                key={item.id}
                type="button"
                className={`t-btn ${strategy === item.id ? 'active' : ''}`}
                style={{
                  padding: '4px 12px',
                  borderRadius: 4,
                  fontSize: 11,
                  background: strategy === item.id ? 'var(--ground-3)' : 'transparent',
                  border: `1px solid ${strategy === item.id ? 'var(--token)' : 'var(--rule)'}`,
                  color: strategy === item.id ? 'var(--token)' : 'var(--text-2)',
                  cursor: 'pointer',
                }}
                onClick={() => setStrategy(item.id as any)}
              >
                {item.label}
              </button>
            ))}
          </div>
        ) : undefined
      }
      {...s}
    >
      <svg viewBox="0 0 640 320" role="img" aria-label="Comparison of ReAct, Plan-then-Execute, and Reflexion planning architectures">
        {/* Step 0: The Planning Dilemma */}
        {st === 0 && (
          <g>
            <rect x="40" y="20" width="560" height="280" rx="8" fill="var(--ground-2)" stroke="var(--rule-2)" />
            <Label x="60" y="45" tone={1} size={13} bold>The Core Dilemma: Hallucination vs Aimless Action</Label>
            <Label x="60" y="65" tone={2} size={10.5} mono={false}>
              Why neither pure Chain-of-Thought nor pure Tool Calling solves multi-step engineering tasks:
            </Label>

            {/* Path A: Reasoning only */}
            <rect x="60" y="90" width="250" height="185" rx="6" fill="var(--ground-3)" stroke="var(--rule-2)" />
            <Chip x="75" y="105" text="Reasoning Only (Chain-of-Thought)" kind="compute" mono={false} size={10.5} />
            <Label x="75" y="140" tone={2} size={10} mono={false}>Model ponders internally without tools:</Label>
            <Label x="75" y="160" tone={3} size={9.5}>Thought 1: "Maybe replica 3 is out of RAM."</Label>
            <Label x="75" y="178" tone={3} size={9.5}>Thought 2: "I estimate memory was 82 GB."</Label>
            <Label x="75" y="196" tone={3} size={9.5}>Thought 3: "I conclude node failure."</Label>
            <Chip x="75" y="225" text="Failure: Hallucinates ungrounded facts" kind="error" mono={false} size={9.5} />

            {/* Path B: Act only */}
            <rect x="330" y="90" width="250" height="185" rx="6" fill="var(--ground-3)" stroke="var(--rule-2)" />
            <Chip x="345" y="105" text="Acting Only (No Reflection)" kind="token" mono={false} size={10.5} />
            <Label x="345" y="140" tone={2} size={10} mono={false}>Model emits tools without synthesis:</Label>
            <Label x="345" y="160" tone={3} size={9.5}>Action 1: query_logs("rep-3") → 500 lines</Label>
            <Label x="345" y="178" tone={3} size={9.5}>Action 2: restart_pod("rep-3") → restarts</Label>
            <Label x="345" y="196" tone={3} size={9.5}>Action 3: query_logs("rep-3") → still dead</Label>
            <Chip x="345" y="225" text="Failure: Loops blindly without plan" kind="error" mono={false} size={9.5} />
          </g>
        )}

        {/* Step 1 or (Step 5 && strategy === 'react'): ReAct Interleaving */}
        {(st === 1 || (st === 5 && strategy === 'react')) && (
          <g>
            <rect x="40" y="20" width="560" height="280" rx="8" fill="var(--ground-2)" stroke="var(--rule-2)" />
            <Label x="60" y="45" tone={1} size={13} bold>ReAct: Synergizing Reasoning and Acting</Label>
            <Label x="60" y="65" tone={2} size={10.5} mono={false}>
              Thought → Action → Observation cycle. Dynamic adaptation at each step.
            </Label>

            {/* ReAct Sequence trace */}
            <g transform="translate(60, 90)">
              {/* Turn 1 */}
              <Span x="0" y="0" w="140" h="30" kind="compute" label="Thought 1: Inspect logs" />
              <Arrow x1="145" y1="15" x2="175" y2="15" />
              <Span x="180" y="0" w="150" h="30" kind="token" label="Action 1: get_logs()" />
              <Arrow x1="335" y1="15" x2="365" y2="15" />
              <Span x="370" y="0" w="150" h="30" kind="comm" label="Obs 1: OOMKilled 137" />

              {/* Connector down */}
              <Arrow x1="445" y1="35" x2="445" y2="55" />
              <Arrow x1="445" y1="55" x2="60" y2="55" />
              <Arrow x1="60" y1="55" x2="60" y2="70" />

              {/* Turn 2 */}
              <Span x="0" y="75" w="140" h="30" kind="compute" label="Thought 2: Check HBM" />
              <Arrow x1="145" y1="90" x2="175" y2="90" />
              <Span x="180" y="75" w="150" h="30" kind="token" label="Action 2: read_metrics()" />
              <Arrow x1="335" y1="90" x2="365" y2="90" />
              <Span x="370" y="75" w="150" h="30" kind="comm" label="Obs 2: KV cache 98%" />

              {/* Connector down to Final Answer */}
              <Arrow x1="445" y1="110" x2="445" y2="130" />
              <Arrow x1="445" y1="130" x2="260" y2="130" />
              <Arrow x1="260" y1="130" x2="260" y2="145" />

              <Span x="180" y="145" w="160" h="30" kind="compute" label="Final Answer: Root Cause" />
            </g>

            <rect x="60" y="270" width="520" height="20" rx="3" fill="var(--ground-3)" />
            <Label x="75" y="284" tone={3} size={9} mono={false}>
              Profile: Sequential roundtrips (High latency). Can react instantly to unexpected tool errors.
            </Label>
          </g>
        )}

        {/* Step 2 & 3 or (Step 5 && strategy === 'plan_exec'): Plan-then-Execute */}
        {(st === 2 || st === 3 || (st === 5 && strategy === 'plan_exec')) && (
          <g>
            <rect x="40" y="20" width="560" height="280" rx="8" fill="var(--ground-2)" stroke="var(--rule-2)" />
            <Label x="60" y="45" tone={1} size={13} bold>Plan-then-Execute: Decoupled DAG Architecture</Label>
            <Label x="60" y="65" tone={2} size={10.5} mono={false}>
              Planner (70B) generates subtasks once; Executors run independent tasks concurrently.
            </Label>

            {/* Stage 1: The Planner */}
            <rect x="60" y="90" width="130" height="110" rx="6" fill="var(--ground-3)" stroke="var(--compute)" strokeWidth="1.5" />
            <Label x="125" y="112" anchor="middle" tone={1} size={11} bold kind="compute">1. PLANNER (70B)</Label>
            <Label x="125" y="132" anchor="middle" tone={3} size={9} mono={false}>Decomposes Goal</Label>
            <Label x="125" y="148" anchor="middle" tone={3} size={9} mono={false}>into Subtask DAG</Label>
            <Chip x="75" y="165" text="DAG Schema" kind="token" mono={false} size={9} />

            <Arrow x1="195" y1="145" x2="245" y2="110" />
            <Arrow x1="195" y1="145" x2="245" y2="150" />
            <Arrow x1="195" y1="145" x2="245" y2="190" />

            {/* Stage 2: Parallel Workers (DAG) */}
            <g transform="translate(250, 90)">
              <rect x="0" y="0" width="160" height="32" rx="4" fill="var(--ground-3)" stroke="var(--rule-2)" />
              <Label x="10" y="20" tone={1} size={9.5}>Task A: Fetch Pod Logs</Label>

              <rect x="0" y="40" width="160" height="32" rx="4" fill="var(--ground-3)" stroke="var(--rule-2)" />
              <Label x="10" y="60" tone={1} size={9.5}>Task B: Query HBM Metrics</Label>

              <rect x="0" y="80" width="160" height="32" rx="4" fill="var(--ground-3)" stroke="var(--rule-2)" />
              <Label x="10" y="100" tone={1} size={9.5}>Task C: Inspect Node Alerts</Label>

              <Label x="80" y="125" anchor="middle" tone={2} size={9} bold>
                PARALLEL DISPATCH (1x Wall-Clock)
              </Label>
            </g>

            <Arrow x1="415" y1="105" x2="455" y2="145" />
            <Arrow x1="415" y1="145" x2="455" y2="145" />
            <Arrow x1="415" y1="185" x2="455" y2="145" />

            {/* Stage 3: Synthesizer */}
            <rect x="460" y="90" width="120" height="110" rx="6" fill="var(--ground-3)" stroke="var(--token)" strokeWidth="1.5" />
            <Label x="520" y="112" anchor="middle" tone={1} size={11} bold kind="token">3. SYNTHESIZE</Label>
            <Label x="520" y="132" anchor="middle" tone={3} size={9} mono={false}>Fast 8B Model</Label>
            <Label x="520" y="152" anchor="middle" tone={3} size={9} mono={false}>Combines Results</Label>
            <Chip x="475" y="168" text="Final Answer" kind="memory" mono={false} size={9} />

            {/* Latency comparison bar */}
            <rect x="60" y="230" width="520" height="60" rx="6" fill="var(--ground-3)" stroke="var(--rule)" />
            <Label x="75" y="250" tone={1} size={10} bold>Wall-Clock Advantage:</Label>
            <Label x="75" y="270" tone={2} size={9.5} mono={false}>
              ReAct runs 3 tasks in 3 sequential steps (3 × 1.5s = 4.5s). DAG runs all 3 in parallel (1 × 1.5s = 1.5s). 3x faster!
            </Label>
          </g>
        )}

        {/* Step 4 or (Step 5 && strategy === 'reflexion'): Reflexion */}
        {(st === 4 || (st === 5 && strategy === 'reflexion')) && (
          <g>
            <rect x="40" y="20" width="560" height="280" rx="8" fill="var(--ground-2)" stroke="var(--rule-2)" />
            <Label x="60" y="45" tone={1} size={13} bold>Reflexion: Verbal Reinforcement Learning</Label>
            <Label x="60" y="65" tone={2} size={10.5} mono={false}>
              Trial → Evaluate → Self-Reflection → Memory Buffer → Retry with Verbal Critique.
            </Label>

            {/* Trial 1: Failure */}
            <rect x="60" y="90" width="220" height="85" rx="6" fill="var(--ground-3)" stroke="var(--error)" strokeWidth="1" />
            <Label x="75" y="110" tone={1} size={10.5} bold kind="error">TRIAL 1 (Failed)</Label>
            <Label x="75" y="128" tone={3} size={9}>Action: reboot_worker("worker-3")</Label>
            <Label x="75" y="144" tone={3} size={9}>Evaluator: Unit test failed (Exit 1)</Label>
            <Chip x="75" y="152" text="Self-Eval: REJECTED" kind="error" mono={false} size={8.5} />

            <Arrow x1="285" y1="130" x2="335" y2="130" />

            {/* Verbal Reflection Memory */}
            <rect x="340" y="90" width="240" height="85" rx="6" fill="var(--ground-3)" stroke="var(--memory)" strokeWidth="1.5" />
            <Label x="355" y="110" tone={1} size={10.5} bold kind="memory">EPISODIC REFLECTION MEMORY</Label>
            <Label x="355" y="128" tone={2} size={8.5} mono={false}>
              Critique: "Rebooting failed because disk
            </Label>
            <Label x="355" y="142" tone={2} size={8.5} mono={false}>
              was full, not corrupted memory. In Trial 2,
            </Label>
            <Label x="355" y="156" tone={2} size={8.5} mono={false}>
              I must check df -h and clear temp files."
            </Label>

            <Arrow x1="460" y1="180" x2="460" y2="200" />
            <Arrow x1="460" y1="200" x2="200" y2="200" />
            <Arrow x1="200" y1="200" x2="200" y2="215" />

            {/* Trial 2: Success with Critique */}
            <rect x="60" y="215" width="520" height="75" rx="6" fill="var(--ground-3)" stroke="var(--token)" strokeWidth="1.5" />
            <Label x="75" y="235" tone={1} size={10.5} bold kind="token">TRIAL 2 (Informed by Memory)</Label>
            <Label x="75" y="253" tone={2} size={9} mono={false}>
              Context includes past critique: Agent executes <tspan fill={COLOR.token}>clean_temp_cache()</tspan> instead of reboot.
            </Label>
            <Label x="75" y="271" tone={2} size={9} mono={false}>
              Evaluator: Tests pass 100%. Task Solved!
            </Label>
          </g>
        )}
      </svg>
    </Stage>
  );
}
