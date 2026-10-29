import { useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Arrow, Fade, Span, Chip, COLOR } from '../lib/svg';

const STEPS = [
  'The spectrum of failure: Six fatal modes',
  'Malformed parameters & schema mismatch',
  'The infinite loop trap: State repetition',
  'Loop detection: Hashes and circuit breakers',
  'Defensive guards: Step & budget circuit breakers',
  'Agent evaluation: The Cost vs Success Pareto frontier',
];

const CAPTIONS = [
  'Single-turn models fail by hallucinating. Agents fail by looping, blowing budgets, mutating wrong tables, drifting from goals, or crashing on malformed JSON parameters.',
  'Parameter errors occur when the model invents arguments, omits required keys, or outputs invalid types. Without schema validation, the execution environment throws an unhandled crash.',
  'Infinite loops are the most common agent killer: the model receives an error, tries the exact same tool with the exact same arguments, and repeats until the context window overflows.',
  'Loop detection calculates a hash of (tool_name, argument_json). If the same state occurs three consecutive times, a circuit breaker trips and halts the runaway process.',
  'Production defensive engineering requires hard boundaries: max_steps caps, cumulative token budgets, per-tool timeouts, and automatic fallback to human operators.',
  'Evaluating agents requires multi-dimensional metrics: Task Success Rate, Steps-to-Completion, Cost per Solved Task, and Goodput under benchmarks like SWE-bench and GAIA.',
];

export default function FailureModesViz() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;

  // Interactive controls
  const [selectedFailure, setSelectedFailure] = useState<number>(2); // 0..5
  const [stepLimit, setStepLimit] = useState(8);

  const failureModes = [
    { name: 'Invalid Tool Selection', cat: 'Syntax', risk: 'Medium', desc: 'Calling non-existent function or hallucinated name' },
    { name: 'Malformed Parameters', cat: 'Schema', risk: 'High', desc: 'Type mismatch, missing required keys, unescaped JSON' },
    { name: 'Infinite State Loops', cat: 'Logic', risk: 'Critical', desc: 'Repeating identical failing action over and over' },
    { name: 'Goal Drift & Rabbit Holes', cat: 'Reasoning', risk: 'High', desc: 'Wandering into irrelevant sub-explorations' },
    { name: 'Compounding Error Cascade', cat: 'Math', risk: 'Critical', desc: '0.95^10 = 59.9% compound reliability cliff' },
    { name: 'Runaway Cost Blowup', cat: 'Economic', risk: 'Critical', desc: 'Uncapped API retry loops burning hundreds of dollars' },
  ];

  const currentFail = failureModes[selectedFailure];

  return (
    <Stage
      rootRef={ref}
      title="Agent Failure Taxonomy, Circuit Breakers & Evaluation"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['error', 'failure mode / loop'],
        ['token', 'guard / circuit breaker'],
        ['compute', 'hash detector / state'],
        ['memory', 'evaluation / metrics'],
      ]}
      readout={
        <Readout
          items={[
            {
              label: 'Inspected Failure',
              value: failureModes[selectedFailure].name.split(' ')[0] + ' ' + failureModes[selectedFailure].name.split(' ')[1],
              kind: 'error',
              note: failureModes[selectedFailure].cat,
            },
            {
              label: 'Step Budget Limit',
              value: `${stepLimit} steps`,
              kind: 'token',
              note: 'Hard ceiling',
            },
            {
              label: 'Circuit Breaker',
              value: st >= 3 ? 'ARMED' : 'DISABLED',
              kind: st >= 3 ? 'token' : 'idle',
              note: st >= 3 ? 'Trips at 3 repeats' : 'Runaway risk',
            },
            {
              label: 'Benchmark Target',
              value: st === 5 ? 'SWE-bench / GAIA' : 'Production Fleet',
              kind: 'memory',
              note: 'Success vs Cost',
            },
          ]}
        />
      }
      controls={
        st === 0 ? (
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {failureModes.map((fm, idx) => (
              <button
                key={fm.name}
                type="button"
                className={`t-btn ${selectedFailure === idx ? 'active' : ''}`}
                style={{
                  padding: '3px 8px',
                  borderRadius: 4,
                  fontSize: 10,
                  background: selectedFailure === idx ? 'var(--ground-3)' : 'transparent',
                  border: `1px solid ${selectedFailure === idx ? COLOR.error : 'var(--rule)'}`,
                  color: selectedFailure === idx ? COLOR.error : 'var(--text-2)',
                  cursor: 'pointer',
                }}
                onClick={() => setSelectedFailure(idx)}
              >
                {fm.name.split(' ')[0]}
              </button>
            ))}
          </div>
        ) : st === 4 ? (
          <Slider
            label="Hard Step Limit (max_steps)"
            value={stepLimit}
            min={3}
            max={15}
            step={1}
            format={(v) => `${v} steps max`}
            onChange={setStepLimit}
          />
        ) : undefined
      }
      {...s}
    >
      <svg viewBox="0 0 640 320" role="img" aria-label="Visualizing agent failure modes, loop detectors, and benchmark metrics">
        {/* Step 0: The Six Failure Modes */}
        {st === 0 && (
          <g>
            <rect x="40" y="20" width="560" height="280" rx="8" fill="var(--ground-2)" stroke="var(--rule-2)" />
            <Label x="60" y="45" tone={1} size={13} bold>The Six Fatal Agent Failure Modes</Label>
            <Label x="60" y="65" tone={2} size={10.5} mono={false}>
              Click a failure mode below to inspect its mechanism and prevention strategy:
            </Label>

            {/* Grid of 6 Failure Modes */}
            {failureModes.map((fm, idx) => {
              const col = idx % 2;
              const row = Math.floor(idx / 2);
              const xPos = 60 + col * 265;
              const yPos = 85 + row * 65;
              const isSel = selectedFailure === idx;

              return (
                <g key={fm.name} onClick={() => setSelectedFailure(idx)} style={{ cursor: 'pointer' }}>
                  <rect
                    x={xPos}
                    y={yPos}
                    width="250"
                    height="55"
                    rx="6"
                    fill={isSel ? 'var(--ground-3)' : 'var(--ground-2)'}
                    stroke={isSel ? COLOR.error : 'var(--rule-2)'}
                    strokeWidth={isSel ? 2 : 1}
                  />
                  <Label x={xPos + 12} y={yPos + 20} tone={isSel ? 1 : 2} size={10.5} bold kind={isSel ? 'error' : undefined}>
                    {idx + 1}. {fm.name}
                  </Label>
                  <Label x={xPos + 12} y={yPos + 38} tone={3} size={8.5} mono={false}>
                    {fm.desc}
                  </Label>
                  <Chip x={xPos + 185} y={yPos + 10} text={fm.risk} kind={fm.risk === 'Critical' ? 'error' : 'compute'} mono={false} size={8} />
                </g>
              );
            })}
          </g>
        )}

        {/* Step 1: Malformed Parameters & Schema Mismatch */}
        {st === 1 && (
          <g>
            <rect x="40" y="20" width="560" height="280" rx="8" fill="var(--ground-2)" stroke="var(--rule-2)" />
            <Label x="60" y="45" tone={1} size={13} bold>Mode 2: Malformed Parameters & Type Violations</Label>
            <Label x="60" y="65" tone={2} size={10.5} mono={false}>
              The model calls the correct tool, but passes invalid types, missing keys, or unescaped strings.
            </Label>

            {/* Bad call trace */}
            <rect x="60" y="90" width="250" height="185" rx="6" fill="var(--ground-3)" stroke="var(--error)" strokeWidth="1" />
            <Chip x="75" y="105" text="Emitted Tool Call (Broken)" kind="error" mono={false} size={10} />
            <Label x="75" y="140" tone={1} size={9}>tool: "query_database"</Label>
            <Label x="75" y="158" tone={1} size={9}>arguments: &#123;</Label>
            <Label x="75" y="174" tone={1} size={9} style={{ fill: COLOR.error }}>  "timeout": "5000ms", // Expected int!</Label>
            <Label x="75" y="190" tone={1} size={9} style={{ fill: COLOR.error }}>  // Missing required "query" key!</Label>
            <Label x="75" y="206" tone={1} size={9}>&#125;</Label>
            <Chip x="75" y="235" text="Crash: TypeError / KeyError" kind="error" mono={false} size={9} />

            {/* Defensive Validation */}
            <rect x="330" y="90" width="250" height="185" rx="6" fill="var(--ground-3)" stroke="var(--token)" strokeWidth="1.5" />
            <Chip x="345" y="105" text="Pydantic / Schema Guard" kind="token" mono={false} size={10} />
            <Label x="345" y="140" tone={2} size={9.5} mono={false}>1. Intercept before execution</Label>
            <Label x="345" y="158" tone={2} size={9.5} mono={false}>2. Catch pydantic.ValidationError</Label>
            <Label x="345" y="176" tone={2} size={9.5} mono={false}>3. Return structured error feedback:</Label>
            <rect x="345" y="190" width="220" height="40" rx="3" fill="var(--ground-2)" />
            <Label x="350" y="205" tone={1} size={8}>"Error: field 'query' is required;</Label>
            <Label x="350" y="218" tone={1} size={8}>field 'timeout' must be an integer."</Label>
            <Chip x="345" y="245" text="Self-Correction on Turn 2" kind="token" mono={false} size={9} />
          </g>
        )}

        {/* Step 2: The Infinite Loop Trap */}
        {st === 2 && (
          <g>
            <rect x="40" y="20" width="560" height="280" rx="8" fill="var(--ground-2)" stroke="var(--rule-2)" />
            <Label x="60" y="45" tone={1} size={13} bold>Mode 3: The Infinite Loop Trap</Label>
            <Label x="60" y="65" tone={2} size={10.5} mono={false}>
              When a tool returns an error, a naive agent repeats the exact same call in a mindless loop.
            </Label>

            {/* Loop Circle */}
            <circle cx="200" cy="180" r="60" fill="none" stroke="var(--error)" strokeWidth="2" strokeDasharray="6 4" />
            <Arrow x1="200" y1="120" x2="230" y2="125" curve={0.3} kind="error" />
            <Arrow x1="200" y1="240" x2="170" y2="235" curve={0.3} kind="error" />

            <rect x="150" y="105" width="100" height="25" rx="3" fill="var(--ground-3)" stroke="var(--error)" />
            <Label x="200" y="122" anchor="middle" tone={1} size={9} bold>read_file("app.py")</Label>

            <rect x="150" y="225" width="100" height="25" rx="3" fill="var(--ground-3)" stroke="var(--error)" />
            <Label x="200" y="242" anchor="middle" tone={1} size={9} bold>FileNotFoundError</Label>

            <Label x="200" y="175" anchor="middle" tone={1} size={14} bold kind="error">LOOPING</Label>
            <Label x="200" y="192" anchor="middle" tone={3} size={9}>Repeated 8 times</Label>

            {/* Explanation box */}
            <rect x="330" y="90" width="250" height="185" rx="6" fill="var(--ground-3)" stroke="var(--rule-2)" />
            <Label x="345" y="115" tone={1} size={11} bold>Why Models Loop:</Label>
            <Label x="345" y="140" tone={2} size={9.5} mono={false}>
              1. The model expects the file to exist.
            </Label>
            <Label x="345" y="160" tone={2} size={9.5} mono={false}>
              2. It believes the failure was a transient glitch.
            </Label>
            <Label x="345" y="180" tone={2} size={9.5} mono={false}>
              3. Without loop detection, it repeats the action
            </Label>
            <Label x="345" y="195" tone={2} size={9.5} mono={false}>
              until context fills and dollars burn.
            </Label>
            <Chip x="345" y="230" text="Result: 100% Task Failure" kind="error" mono={false} size={9.5} />
          </g>
        )}

        {/* Step 3: Loop Detection via State Hashing */}
        {st === 3 && (
          <g>
            <rect x="40" y="20" width="560" height="280" rx="8" fill="var(--ground-2)" stroke="var(--rule-2)" />
            <Label x="60" y="45" tone={1} size={13} bold>Defensive Guard: State Hashing & Circuit Breakers</Label>
            <Label x="60" y="65" tone={2} size={10.5} mono={false}>
              Track hash(tool_name, argument_json). Trip a circuit breaker on 3 consecutive identical calls.
            </Label>

            {/* Hash ring visual */}
            <g transform="translate(60, 95)">
              <rect x="0" y="0" width="520" height="35" rx="4" fill="var(--ground-3)" stroke="var(--rule-2)" />
              <Label x="15" y="22" tone={2} size={10}>Turn 1: hash("cat", "app.py") = 0x8a3f</Label>
              <Chip x="420" y="8" text="Count: 1" kind="compute" mono={false} size={9} />

              <rect x="0" y="45" width="520" height="35" rx="4" fill="var(--ground-3)" stroke="var(--rule-2)" />
              <Label x="15" y="67" tone={2} size={10}>Turn 2: hash("cat", "app.py") = 0x8a3f</Label>
              <Chip x="420" y="53" text="Count: 2 (Warning)" kind="compute" mono={false} size={9} />

              <rect x="0" y="90" width="520" height="40" rx="4" fill="var(--ground-3)" stroke="var(--error)" strokeWidth="1.5" />
              <Label x="15" y="115" tone={1} size={10} bold kind="error">Turn 3: hash("cat", "app.py") = 0x8a3f → CIRCUIT BREAKER TRIPS!</Label>
              <Chip x="420" y="98" text="BREAKER TRIPPED" kind="error" mono={false} size={9} />
            </g>

            <rect x="60" y="245" width="520" height="40" rx="4" fill="var(--ground-3)" stroke="var(--token)" />
            <Label x="75" y="270" tone={1} size={10} bold kind="token">
              Remediation: Return synthetic observation "Loop detected: tool failed 3 times. You MUST try an alternate strategy."
            </Label>
          </g>
        )}

        {/* Step 4: Step Limits & Token Budgets */}
        {st === 4 && (
          <g>
            <rect x="40" y="20" width="560" height="280" rx="8" fill="var(--ground-2)" stroke="var(--rule-2)" />
            <Label x="60" y="45" tone={1} size={13} bold>Hard Execution Ceilings: Step & Token Limits</Label>
            <Label x="60" y="65" tone={2} size={10.5} mono={false}>
              Always enforce a maximum step budget. Slide the control to inspect the cutoff boundary.
            </Label>

            {/* Progress tracks up to stepLimit */}
            {[...Array(12)].map((_, i) => {
              const xPos = 60 + i * 42;
              const isPastLimit = i >= stepLimit;
              return (
                <g key={i}>
                  <rect
                    x={xPos}
                    y="110"
                    width="36"
                    height="50"
                    rx="4"
                    fill={isPastLimit ? 'var(--ground-3)' : 'var(--ground-2)'}
                    stroke={isPastLimit ? 'var(--rule-2)' : COLOR.token}
                    strokeDasharray={isPastLimit ? '3 3' : 'none'}
                  />
                  <Label x={xPos + 18} y="132" anchor="middle" tone={isPastLimit ? 3 : 1} size={10} bold>
                    {i + 1}
                  </Label>
                  <Label x={xPos + 18} y="148" anchor="middle" tone={3} size={8}>
                    {isPastLimit ? 'HALT' : 'Step'}
                  </Label>
                </g>
              );
            })}

            {/* Cutoff barrier */}
            <line x1={60 + stepLimit * 42 - 3} x2={60 + stepLimit * 42 - 3} y1="95" y2="175" stroke="var(--error)" strokeWidth="2" />
            <Label x={60 + stepLimit * 42 - 3} y="90" anchor="middle" tone={1} size={9.5} kind="error">
              max_steps = {stepLimit}
            </Label>

            <rect x="60" y="195" width="520" height="75" rx="6" fill="var(--ground-3)" stroke="var(--rule)" />
            <Label x="75" y="218" tone={1} size={10.5} bold>Production Defense Checklist:</Label>
            <Label x="75" y="238" tone={2} size={9.5} mono={false}>
              1. Hard step limit: `max_steps = 8` (guarantees loop termination).
            </Label>
            <Label x="75" y="255" tone={2} size={9.5} mono={false}>
              2. Cumulative cost budget: `max_cost = \$0.15` (prevents \$100 runaway bills).
            </Label>
          </g>
        )}

        {/* Step 5: Evaluation & Pareto Frontier */}
        {st === 5 && (
          <g>
            <rect x="40" y="20" width="560" height="280" rx="8" fill="var(--ground-2)" stroke="var(--rule-2)" />
            <Label x="60" y="45" tone={1} size={13} bold>Agent Evaluation: Success Rate vs Cost Pareto Frontier</Label>
            <Label x="60" y="65" tone={2} size={10.5} mono={false}>
              Evaluating agents is not BLEU or perplexity. It is task completion, trajectory efficiency, and cost.
            </Label>

            {/* Pareto Chart */}
            <g transform="translate(80, 85)">
              {/* Axes */}
              <line x1="0" x2="300" y1="140" y2="140" stroke="var(--rule)" strokeWidth="1" />
              <line x1="0" x2="0" y1="0" y2="140" stroke="var(--rule)" strokeWidth="1" />
              <Label x="150" y="160" anchor="middle" tone={3} size={9.5}>Cost per Task ($)</Label>
              <Label x="-10" y="70" anchor="middle" tone={3} size={9.5} style={{ transform: 'rotate(-90deg)', transformOrigin: '-10px 70px' }}>Pass@1 (%)</Label>

              {/* Data points */}
              {/* Point 1: Naive ReAct 8B */}
              <circle cx="50" cy="95" r="5" fill={COLOR.compute} />
              <Label x="55" y="90" tone={2} size={8.5}>ReAct 8B (38%, \$0.01)</Label>

              {/* Point 2: ReAct 70B */}
              <circle cx="160" cy="50" r="5" fill={COLOR.token} />
              <Label x="165" y="45" tone={1} size={8.5} bold>ReAct 70B (64%, \$0.08)</Label>

              {/* Point 3: Reflexion 70B */}
              <circle cx="260" cy="25" r="5" fill={COLOR.memory} />
              <Label x="260" y="18" anchor="middle" tone={1} size={8.5} bold>Reflexion 70B (82%, \$0.24)</Label>

              {/* Pareto line */}
              <path d="M 50 95 Q 110 65 160 50 T 260 25" fill="none" stroke="var(--token)" strokeWidth="1.5" strokeDasharray="4 3" />
            </g>

            {/* Benchmark highlights box */}
            <rect x="400" y="90" width="180" height="180" rx="6" fill="var(--ground-3)" stroke="var(--rule-2)" />
            <Label x="415" y="112" tone={1} size={10.5} bold>Core Benchmarks:</Label>
            <Label x="415" y="135" tone={2} size={9} bold>• SWE-bench:</Label>
            <Label x="415" y="150" tone={3} size={8.5} mono={false}>Solve real GitHub issues</Label>
            <Label x="415" y="172" tone={2} size={9} bold>• GAIA:</Label>
            <Label x="415" y="187" tone={3} size={8.5} mono={false}>Multimodal general assistant</Label>
            <Label x="415" y="209" tone={2} size={9} bold>• AgentBench:</Label>
            <Label x="415" y="224" tone={3} size={8.5} mono={false}>OS & DB interaction</Label>
            <Chip x="415" y="242" text="Metric: Pass@1, Cost/Task" kind="token" mono={false} size={8.5} />
          </g>
        )}
      </svg>
    </Stage>
  );
}
