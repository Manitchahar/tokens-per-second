import { useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Arrow, Fade, Span, Chip, COLOR } from '../lib/svg';

const STEPS = [
  'The static prompt-response wall',
  'Perceive: Ingest state and user intent',
  'Plan: Formulate reasoning and action',
  'Act: Issue a structured tool call',
  'Observe: Ingest environment feedback',
  'The closed loop: State accumulation',
  'Compounding errors: The 0.95^10 cliff',
  'The serving cost: Prefix cache reuse',
];

const CAPTIONS = [
  'A standard LLM call is stateless: user sends prompt, model responds once. When the task requires live inspection, computation, or verification, a single pass inevitably fails or hallucinates.',
  'An agent begins with perception: the model reads the user goal, available system tools, and current environment state into its working context window.',
  'Planning: Before acting, an autonomous agent reasons over what is known and what is missing, deciding whether to conclude or invoke a tool.',
  'Action: The LLM emits a structured tool invocation (e.g., querying cluster logs). The model pauses generation while the runtime executes the call.',
  'Observation: The tool environment returns raw results (e.g. exit code 137, OOMKilled). This observation is appended to the agent’s working history.',
  'The loop repeats: Perceive -> Plan -> Act -> Observe. Each iteration refines the agent’s belief state until the primary goal is satisfied.',
  'The compounding error trap: If each step has a 95% success rate, a 10-step trajectory succeeds only 0.95^10 = 59.9% of the time. At 20 steps, reliability drops to 35.9%.',
  'Inference engineering reality: Each turn appends new tokens to context. Without prefix caching, every turn re-prefills all earlier history, creating quadratic FLOP waste.',
];

export default function AgentLoopViz() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;

  // State for interactive sliders
  const [stepAccuracy, setStepAccuracy] = useState(0.95);
  const [numSteps, setNumSteps] = useState(10);

  // Arithmetic for compounding errors
  const overallSuccess = Math.pow(stepAccuracy, numSteps);

  // Coordinates for the loop cycle nodes
  const cx = 320;
  const cy = 145;
  const rx = 150;
  const ry = 80;

  // Node positions on ellipse:
  // Perceive: Top (cy - ry = 65)
  // Plan: Right (cx + rx = 470)
  // Act: Bottom (cy + ry = 225)
  // Observe: Left (cx - rx = 170)
  const nodes = [
    { name: 'Perceive', x: cx, y: cy - ry, kind: 'memory' as const, desc: 'Read state & context' },
    { name: 'Plan', x: cx + rx, y: cy, kind: 'compute' as const, desc: 'Reason next move' },
    { name: 'Act', x: cx, y: cy + ry, kind: 'token' as const, desc: 'Invoke tool action' },
    { name: 'Observe', x: cx - rx, y: cy, kind: 'comm' as const, desc: 'Capture tool output' },
  ];

  // Active node index based on step
  let activeNode = -1;
  if (st === 1) activeNode = 0; // Perceive
  if (st === 2) activeNode = 1; // Plan
  if (st === 3) activeNode = 2; // Act
  if (st === 4) activeNode = 3; // Observe
  if (st === 5) activeNode = -2; // Full loop spinning

  return (
    <Stage
      rootRef={ref}
      title="The Agent Loop & Compounding Error Arithmetic"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['memory', 'state / context'],
        ['compute', 'planning / LLM'],
        ['token', 'action / tool call'],
        ['comm', 'observation / env'],
        ['error', 'failure probability'],
      ]}
      readout={
        <Readout
          items={[
            {
              label: 'Loop Stage',
              value: st === 0 ? 'Stateless' : st <= 4 ? nodes[st - 1].name : st === 5 ? 'Iterating' : st === 6 ? 'Compounding' : 'Prefix Cache',
              kind: st === 1 ? 'memory' : st === 2 ? 'compute' : st === 3 ? 'token' : st === 4 ? 'comm' : st === 6 ? 'error' : 'text',
              note: st === 6 ? `k = ${numSteps} steps` : 'Active agent phase',
            },
            {
              label: 'Step Accuracy (p)',
              value: `${(stepAccuracy * 100).toFixed(1)}%`,
              kind: 'token',
              note: 'per-turn reliability',
            },
            {
              label: 'Task Success P(k)',
              value: `${(overallSuccess * 100).toFixed(1)}%`,
              kind: overallSuccess < 0.5 ? 'error' : overallSuccess < 0.8 ? 'compute' : 'token',
              note: `p^${numSteps} = (${stepAccuracy.toFixed(2)})^${numSteps}`,
            },
            {
              label: 'Cache Status',
              value: st === 7 ? 'Shared Prefix' : 'Uncached',
              kind: st === 7 ? 'memory' : 'idle',
              note: st === 7 ? 'Turn 1..k KV reused' : 'Turn overhead',
            },
          ]}
        />
      }
      controls={
        st === 6 ? (
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', width: '100%' }}>
            <Slider
              label="Per-step accuracy (p)"
              value={stepAccuracy}
              min={0.80}
              max={0.99}
              step={0.01}
              format={(v) => `${(v * 100).toFixed(0)}%`}
              onChange={setStepAccuracy}
            />
            <Slider
              label="Trajectory steps (k)"
              value={numSteps}
              min={1}
              max={20}
              step={1}
              format={(v) => `${v} steps`}
              onChange={setNumSteps}
            />
          </div>
        ) : undefined
      }
      {...s}
    >
      <svg viewBox="0 0 640 320" role="img" aria-label="Agent Perceive Plan Act Observe loop and compounding error curve">
        {/* Step 0: Static Prompt-Response vs Agent */}
        {st === 0 && (
          <g>
            <rect x="60" y="40" width="520" height="100" rx="8" fill="var(--ground-2)" stroke="var(--rule-2)" />
            <Label x="80" y="70" tone={1} size={14} bold>Stateless Single-Turn LLM</Label>
            <Label x="80" y="92" tone={2} size={11.5} mono={false}>User Prompt → Model Forward Pass → Single Text Answer</Label>
            <Chip x="80" y="105" text="Prompt: 'Why is pod 3 failing?'" kind="text" mono={false} size={10} />
            <Arrow x1="320" y1="116" x2="360" y2="116" />
            <Chip x="370" y="105" text="Answer: 'Check your logs... (Hallucinated guesswork)'" kind="error" mono={false} size={10} />

            <rect x="60" y="160" width="520" height="130" rx="8" fill="var(--ground-2)" stroke="var(--rule-2)" />
            <Label x="80" y="190" tone={1} size={14} bold>Autonomous Agentic System</Label>
            <Label x="80" y="212" tone={2} size={11.5} mono={false}>Goal-driven loop: Model controls environment tools, inspects real outputs, updates plan</Label>
            <Chip x="80" y="235" text="Goal: 'Fix replica 3'" kind="compute" mono={false} size={10.5} />
            <Arrow x1="210" y1="246" x2="245" y2="246" />
            <Chip x="250" y="235" text="Perceive → Plan → Act → Observe" kind="token" mono={false} size={10.5} />
            <Arrow x1="475" y1="246" x2="510" y2="246" />
            <Chip x="515" y="235" text="Verified Fix" kind="memory" mono={false} size={10.5} />
          </g>
        )}

        {/* Steps 1 to 5: The Four-Phase Agent Loop */}
        {st >= 1 && st <= 5 && (
          <g>
            {/* Background loop ring */}
            <ellipse
              cx={cx}
              cy={cy}
              rx={rx}
              ry={ry}
              fill="none"
              stroke="var(--rule-2)"
              strokeWidth="2"
              strokeDasharray={st === 5 ? '6 4' : 'none'}
            />

            {/* Direction arrows on the loop */}
            <Arrow x1={cx + 10} y1={cy - ry} x2={cx + 70} y2={cy - ry + 12} curve={0.2} muted={st !== 5 && activeNode !== 0} kind={activeNode === 0 ? 'memory' : undefined} />
            <Arrow x1={cx + rx} y1={cy + 10} x2={cx + rx - 20} y2={cy + 55} curve={0.2} muted={st !== 5 && activeNode !== 1} kind={activeNode === 1 ? 'compute' : undefined} />
            <Arrow x1={cx - 10} y1={cy + ry} x2={cx - 70} y2={cy + ry - 12} curve={0.2} muted={st !== 5 && activeNode !== 2} kind={activeNode === 2 ? 'token' : undefined} />
            <Arrow x1={cx - rx} y1={cy - 10} x2={cx - rx + 20} y2={cy - 55} curve={0.2} muted={st !== 5 && activeNode !== 3} kind={activeNode === 3 ? 'comm' : undefined} />

            {/* Central Agent Controller Node */}
            <circle cx={cx} cy={cy} r="32" fill="var(--ground-3)" stroke="var(--rule)" strokeWidth="1.5" />
            <Label x={cx} y={cy - 6} anchor="middle" tone={1} size={11} bold>LLM Core</Label>
            <Label x={cx} y={cy + 10} anchor="middle" tone={3} size={9.5} mono={false}>Controller</Label>

            {/* The 4 Loop Nodes */}
            {nodes.map((n, idx) => {
              const isHighlight = activeNode === idx || activeNode === -2;
              return (
                <g key={n.name}>
                  <rect
                    x={n.x - 55}
                    y={n.y - 20}
                    width="110"
                    height="40"
                    rx="6"
                    fill={isHighlight ? 'var(--ground-3)' : 'var(--ground-2)'}
                    stroke={isHighlight ? COLOR[n.kind] : 'var(--rule-2)'}
                    strokeWidth={isHighlight ? 2 : 1}
                  />
                  <Label x={n.x} y={n.y - 3} anchor="middle" tone={isHighlight ? 1 : 2} size={12} bold kind={isHighlight ? n.kind : undefined}>
                    {n.name}
                  </Label>
                  <Label x={n.x} y={n.y + 11} anchor="middle" tone={3} size={9} mono={false}>
                    {n.desc}
                  </Label>
                </g>
              );
            })}

            {/* Trace context box at bottom */}
            <rect x="50" y="245" width="540" height="65" rx="6" fill="var(--ground-2)" stroke="var(--rule-2)" />
            <Label x="65" y="263" tone={2} size={10.5} bold>LIVE AGENT WORKING TRACE (TURN 1):</Label>
            {st === 1 && (
              <Label x="65" y="285" tone={1} size={10} mono={false}>
                Context loaded: [User: "Replica 3 crashed"] + [Tools: get_logs, restart_pod, inspect_hbm]
              </Label>
            )}
            {st === 2 && (
              <Label x="65" y="285" tone={1} size={10} mono={false}>
                Thought: "Before taking actions, I must verify exit status and logs of replica 3."
              </Label>
            )}
            {st === 3 && (
              <Label x="65" y="285" tone={1} size={10} mono={false}>
                Tool Call Emitted: <tspan fill={COLOR.token}>get_logs(pod="replica-3", tail=50)</tspan> → Execution paused.
              </Label>
            )}
            {st === 4 && (
              <Label x="65" y="285" tone={1} size={10} mono={false}>
                Observation Ingested: <tspan fill={COLOR.comm}>[system] Exit code 137: Out of Memory (OOMKilled)</tspan>
              </Label>
            )}
            {st === 5 && (
              <Label x="65" y="285" tone={1} size={10} mono={false}>
                Turn 2 Plan formulated: "Exit code 137 indicates OOM. Invoking inspect_hbm to find memory leak."
              </Label>
            )}
          </g>
        )}

        {/* Step 6: Compounding Errors Waterfall */}
        {st === 6 && (
          <g>
            <rect x="40" y="20" width="560" height="275" rx="8" fill="var(--ground-2)" stroke="var(--rule-2)" />
            <Label x="60" y="45" tone={1} size={13} bold>
              Compound Reliability: P(success) = p^k
            </Label>
            <Label x="60" y="65" tone={2} size={10.5} mono={false}>
              Each step is an independent Bernoulli trial. Watch how small errors compound exponentially over long horizons.
            </Label>

            {/* Error curve bar chart */}
            {[1, 2, 3, 5, 8, 10, 15, 20].map((kVal, idx) => {
              const prob = Math.pow(stepAccuracy, kVal);
              const barH = prob * 140;
              const xPos = 80 + idx * 60;
              const yPos = 230 - barH;
              const isSelected = kVal === numSteps;
              const col = prob > 0.75 ? COLOR.token : prob > 0.5 ? COLOR.compute : COLOR.error;

              return (
                <g key={kVal}>
                  <rect
                    x={xPos}
                    y={yPos}
                    width="40"
                    height={barH}
                    rx="3"
                    fill={col}
                    opacity={isSelected ? 1 : 0.65}
                    stroke={isSelected ? '#ffffff' : 'none'}
                    strokeWidth={isSelected ? 1.5 : 0}
                  />
                  <Label x={xPos + 20} y={yPos - 6} anchor="middle" tone={1} size={9} bold>
                    {(prob * 100).toFixed(0)}%
                  </Label>
                  <Label x={xPos + 20} y={245} anchor="middle" tone={2} size={9}>
                    k={kVal}
                  </Label>
                </g>
              );
            })}

            {/* Axis base line */}
            <line x1="60" x2="560" y1="230" y2="230" stroke="var(--rule)" strokeWidth="1" />
            <Label x="60" y="270" tone={3} size={9.5} mono={false}>
              Formula: P(k = {numSteps}) = ({stepAccuracy.toFixed(2)})^{numSteps} = {(overallSuccess * 100).toFixed(1)}% success probability
            </Label>
          </g>
        )}

        {/* Step 7: Inference Cost & Prefix Cache */}
        {st === 7 && (
          <g>
            <rect x="40" y="20" width="560" height="280" rx="8" fill="var(--ground-2)" stroke="var(--rule-2)" />
            <Label x="60" y="45" tone={1} size={13} bold>The Inference Engineer's Perspective: Context Accumulation</Label>
            <Label x="60" y="65" tone={2} size={10.5} mono={false}>
              Turn n includes all previous turns: Prompt(n) = System + Tools + Turn_1 + ... + Turn_n-1
            </Label>

            {/* Turn 1 track */}
            <Label x="60" y="105" tone={2} size={10} bold>Turn 1 (Prefill)</Label>
            <Span x="160" y="92" w="80" h="20" kind="compute" label="Sys+Tools" />
            <Span x="245" y="92" w="70" h="20" kind="memory" label="Goal" />
            <Span x="320" y="92" w="50" h="20" kind="token" label="Act 1" />
            <Span x="375" y="92" w="65" h="20" kind="comm" label="Obs 1" />

            {/* Turn 2 track */}
            <Label x="60" y="145" tone={2} size={10} bold>Turn 2 (Prefix hit!)</Label>
            <Span x="160" y="132" w="280" h="20" kind="memory" label="Cached KV (Turn 1 Prefix: 440 tokens)" />
            <Span x="445" y="132" w="55" h="20" kind="token" label="Act 2" />
            <Span x="505" y="132" w="70" h="20" kind="comm" label="Obs 2" />

            {/* Turn 3 track */}
            <Label x="60" y="185" tone={2} size={10} bold>Turn 3 (Prefix hit!)</Label>
            <Span x="160" y="172" w="415" h="20" kind="memory" label="Cached KV (Turn 1 + 2 Prefix: 810 tokens)" />
            <Span x="580" y="172" w="45" h="20" kind="token" label="Final" />

            {/* Summary comparison */}
            <rect x="60" y="215" width="520" height="70" rx="6" fill="var(--ground-3)" stroke="var(--rule-2)" />
            <Label x="75" y="235" tone={1} size={10.5} bold>Why Radix / Prefix Caching is Non-Negotiable for Agents:</Label>
            <Label x="75" y="255" tone={2} size={9.5} mono={false}>
              Without prefix cache: Turn 1 (440T) + Turn 2 (810T) + Turn 3 (1100T) = 2,350 prefill FLOPs (O(k²) waste).
            </Label>
            <Label x="75" y="272" tone={2} size={9.5} mono={false}>
              With prefix cache: Turn 1 (440T) + Turn 2 (370 new T) + Turn 3 (290 new T) = 1,100 prefill FLOPs (53% savings!).
            </Label>
          </g>
        )}
      </svg>
    </Stage>
  );
}
