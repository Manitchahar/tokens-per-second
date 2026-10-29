import { useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Arrow, Fade, Span, Chip, COLOR } from '../lib/svg';

const STEPS = [
  'The three tiers: Working context, external store, weights',
  'The unmanaged context overflow trap',
  'Naive sliding window: The amnesia problem',
  'Recursive summarization: Compaction buffers',
  'The hybrid context architecture',
  'Token budget arithmetic: Capping quadratic growth',
];

const CAPTIONS = [
  'Agent memory spans three distinct tiers: high-bandwidth working context (RAM), external vector/key-value stores (disk/database), and frozen parametric weights (ROM).',
  'In an unmanaged agent loop, every turn appends actions and observations. By turn 20, context size explodes, causing latency spikes, high costs, and eventual OOM.',
  'A naive sliding window discards oldest tokens to stay within budget. But dropping Turn 1 deletes the system prompt and original user goal, causing immediate agent amnesia.',
  'Recursive summarization compacts past turns into a concise structured state representation, shrinking thousands of observation tokens down to a dense 250-token summary.',
  'Production hybrid memory: Anchor the static system prompt and initial goal at the start, inject the compacted summary in the middle, and keep the last M turns raw.',
  'Token budget management: Compare unmanaged quadratic prefill vs compacted linear budget. Compaction caps context at 4,000 tokens, saving 65% of compute.',
];

export default function AgentMemoryViz() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;

  // Interactive controls
  const [turnCount, setTurnCount] = useState(12);
  const [compactionStrategy, setCompactionStrategy] = useState<'unmanaged' | 'sliding' | 'summary'>('summary');

  const staticPrefix = 1200; // system prompt + goal
  const tokensPerTurn = 450; // tool call + observation

  // Arithmetic for context sizes
  const unmanagedSize = staticPrefix + turnCount * tokensPerTurn;
  const slidingSize = Math.min(unmanagedSize, 4000);
  const summarySize = staticPrefix + 300 + Math.min(turnCount, 4) * tokensPerTurn;

  const currentSize =
    st === 1 ? unmanagedSize :
    st === 2 ? slidingSize :
    st === 3 || st === 4 ? summarySize :
    compactionStrategy === 'unmanaged' ? unmanagedSize :
    compactionStrategy === 'sliding' ? slidingSize : summarySize;

  const maxBudget = 4000;
  const isOverBudget = currentSize > maxBudget;

  return (
    <Stage
      rootRef={ref}
      title="Agent Memory Tiers & Context Budget Compaction"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['memory', 'working context (RAM)'],
        ['comm', 'external store (Disk)'],
        ['compute', 'parametric weights (ROM)'],
        ['error', 'context overflow / amnesia'],
      ]}
      readout={
        <Readout
          items={[
            {
              label: 'Memory Policy',
              value: st === 0 ? '3 Tiers' : st === 1 ? 'Unmanaged' : st === 2 ? 'Sliding Window' : st === 3 || st === 4 ? 'Compacted' : compactionStrategy,
              kind: st === 1 ? 'error' : st === 2 ? 'compute' : 'memory',
              note: `Turn ${turnCount} of 20`,
            },
            {
              label: 'Active Context Size',
              value: `${currentSize} tokens`,
              kind: isOverBudget ? 'error' : 'token',
              note: `Budget: ${maxBudget} tokens`,
            },
            {
              label: 'Budget Utilization',
              value: `${((currentSize / maxBudget) * 100).toFixed(0)}%`,
              kind: isOverBudget ? 'error' : ((currentSize / maxBudget) > 0.85 ? 'compute' : 'token'),
              note: isOverBudget ? 'OVERFLOW DANGER' : 'Within GPU limit',
            },
            {
              label: 'Goal Preservation',
              value: st === 2 && turnCount > 8 ? 'LOST (Amnesia)' : 'Preserved (Anchored)',
              kind: st === 2 && turnCount > 8 ? 'error' : 'token',
              note: 'Initial user intent status',
            },
          ]}
        />
      }
      controls={
        st >= 1 ? (
          <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
            <Slider
              label="Agent Trajectory Turns"
              value={turnCount}
              min={2}
              max={20}
              step={1}
              format={(v) => `Turn ${v}`}
              onChange={setTurnCount}
            />
            {st === 5 && (
              <div style={{ display: 'flex', gap: 8 }}>
                {(['unmanaged', 'sliding', 'summary'] as const).map((strat) => (
                  <button
                    key={strat}
                    type="button"
                    className={`t-btn ${compactionStrategy === strat ? 'active' : ''}`}
                    style={{
                      padding: '4px 10px',
                      borderRadius: 4,
                      fontSize: 10.5,
                      background: compactionStrategy === strat ? 'var(--ground-3)' : 'transparent',
                      border: `1px solid ${compactionStrategy === strat ? 'var(--token)' : 'var(--rule)'}`,
                      color: compactionStrategy === strat ? 'var(--token)' : 'var(--text-2)',
                      cursor: 'pointer',
                    }}
                    onClick={() => setCompactionStrategy(strat)}
                  >
                    {strat === 'unmanaged' ? 'Unmanaged' : strat === 'sliding' ? 'Sliding Window' : 'Compacted'}
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : undefined
      }
      {...s}
    >
      <svg viewBox="0 0 640 320" role="img" aria-label="Visualizing agent memory tiers and context window compaction">
        {/* Step 0: The Three Tiers */}
        {st === 0 && (
          <g>
            <rect x="40" y="20" width="560" height="280" rx="8" fill="var(--ground-2)" stroke="var(--rule-2)" />
            <Label x="60" y="45" tone={1} size={13} bold>The Three Tiers of Agent Memory</Label>
            <Label x="60" y="65" tone={2} size={10.5} mono={false}>
              An agent coordinates memory across three fundamentally different physical hardware layers:
            </Label>

            {/* Tier 1: Working Context */}
            <rect x="60" y="90" width="160" height="180" rx="6" fill="var(--ground-3)" stroke="var(--memory)" strokeWidth="1.5" />
            <Chip x="75" y="105" text="1. Working Context" kind="memory" mono={false} size={10} />
            <Label x="75" y="135" tone={1} size={10} bold>GPU HBM / KV Cache</Label>
            <Label x="75" y="155" tone={2} size={9} mono={false}>• Ultra-fast access (TB/s)</Label>
            <Label x="75" y="172" tone={2} size={9} mono={false}>• Strict capacity limit (4k-128k)</Label>
            <Label x="75" y="189" tone={2} size={9} mono={false}>• Short-term working scratchpad</Label>
            <Label x="75" y="206" tone={2} size={9} mono={false}>• Lost when session ends</Label>
            <Chip x="75" y="235" text="Analogy: Fast RAM" kind="text" mono={false} size={9} />

            {/* Tier 2: External Store */}
            <rect x="240" y="90" width="160" height="180" rx="6" fill="var(--ground-3)" stroke="var(--comm)" strokeWidth="1.5" />
            <Chip x="255" y="105" text="2. External Store" kind="comm" mono={false} size={10} />
            <Label x="255" y="135" tone={1} size={10} bold>Vector DB / SQL / Disk</Label>
            <Label x="255" y="155" tone={2} size={9} mono={false}>• Infinite persistent storage</Label>
            <Label x="255" y="172" tone={2} size={9} mono={false}>• Millisecond retrieval latency</Label>
            <Label x="255" y="189" tone={2} size={9} mono={false}>• Long-term episodic memory</Label>
            <Label x="255" y="206" tone={2} size={9} mono={false}>• Retrieved on-demand</Label>
            <Chip x="255" y="235" text="Analogy: Hard Disk / NVMe" kind="text" mono={false} size={9} />

            {/* Tier 3: Model Weights */}
            <rect x="420" y="90" width="160" height="180" rx="6" fill="var(--ground-3)" stroke="var(--compute)" strokeWidth="1.5" />
            <Chip x="435" y="105" text="3. Parametric Memory" kind="compute" mono={false} size={10} />
            <Label x="435" y="135" tone={1} size={10} bold>Transformer Weights (θ)</Label>
            <Label x="435" y="155" tone={2} size={9} mono={false}>• Embedded world knowledge</Label>
            <Label x="435" y="172" tone={2} size={9} mono={false}>• Zero retrieval latency</Label>
            <Label x="435" y="189" tone={2} size={9} mono={false}>• Frozen at inference time</Label>
            <Label x="435" y="206" tone={2} size={9} mono={false}>• Only modified by training</Label>
            <Chip x="435" y="235" text="Analogy: Burned ROM" kind="text" mono={false} size={9} />
          </g>
        )}

        {/* Steps 1 to 5: Context Window Management & Compaction */}
        {st >= 1 && (
          <g>
            <rect x="40" y="20" width="560" height="280" rx="8" fill="var(--ground-2)" stroke="var(--rule-2)" />
            <Label x="60" y="45" tone={1} size={13} bold>
              {st === 1 ? 'The Unmanaged Context Accumulation Trap' :
               st === 2 ? 'Naive Sliding Window: The Catastrophic Amnesia Problem' :
               st === 3 ? 'Recursive Summarization: Compacting Observations' :
               st === 4 ? 'The Production Hybrid Context Architecture' :
               'Token Budget Management: Quadratic vs Compacted'}
            </Label>
            <Label x="60" y="65" tone={2} size={10.5} mono={false}>
              Context Window Capacity Limit: 4,000 Tokens (Budget threshold).
            </Label>

            {/* Context bar container */}
            <rect x="60" y="95" width="520" height="40" rx="4" fill="var(--ground-3)" stroke="var(--rule)" />
            <line x1={60 + (maxBudget / 10000) * 520} x2={60 + (maxBudget / 10000) * 520} y1="85" y2="145" stroke="var(--error)" strokeWidth="2" strokeDasharray="4 3" />
            <Label x={60 + (maxBudget / 10000) * 520} y="80" anchor="middle" tone={1} size={9} kind="error">Max Budget (4,000 T)</Label>

            {/* Rendering the active context blocks */}
            {st === 1 && (
              <g>
                {/* Unmanaged growth */}
                <Span x="60" y="98" w={(staticPrefix / 10000) * 520} h="34" kind="compute" label="Prefix (1.2k)" />
                <Span
                  x={60 + (staticPrefix / 10000) * 520}
                  y="98"
                  w={(turnCount * tokensPerTurn / 10000) * 520}
                  h="34"
                  kind={isOverBudget ? 'error' : 'memory'}
                  label={`Turns 1..${turnCount} (${turnCount * tokensPerTurn} T)`}
                />
              </g>
            )}

            {st === 2 && (
              <g>
                {/* Sliding window: FIFO truncation */}
                {turnCount > 8 ? (
                  <g>
                    <Span x="60" y="98" w="100" h="34" kind="idle" ghost label="LOST GOAL" />
                    <Span x="165" y="98" w={(slidingSize / 10000) * 520 - 105} h="34" kind="memory" label={`Recent Turns ${turnCount - 6}..${turnCount}`} />
                  </g>
                ) : (
                  <g>
                    <Span x="60" y="98" w={(staticPrefix / 10000) * 520} h="34" kind="compute" label="Prefix (1.2k)" />
                    <Span x={60 + (staticPrefix / 10000) * 520} y="98" w={(turnCount * tokensPerTurn / 10000) * 520} h="34" kind="memory" label={`Turns 1..${turnCount}`} />
                  </g>
                )}
              </g>
            )}

            {(st === 3 || st === 4 || (st === 5 && compactionStrategy === 'summary')) && (
              <g>
                {/* Hybrid compacted layout */}
                <Span x="60" y="98" w={(staticPrefix / 10000) * 520} h="34" kind="compute" label="Anchored Goal (1.2k)" />
                <Span x={60 + (staticPrefix / 10000) * 520} y="98" w={(300 / 10000) * 520} h="34" kind="token" label="Summary (300 T)" />
                <Span
                  x={60 + ((staticPrefix + 300) / 10000) * 520}
                  y="98"
                  w={(Math.min(turnCount, 4) * tokensPerTurn / 10000) * 520}
                  h="34"
                  kind="memory"
                  label={`Last ${Math.min(turnCount, 4)} Turns (${Math.min(turnCount, 4) * tokensPerTurn} T)`}
                />
              </g>
            )}

            {/* Diagnostic callout box */}
            <rect x="60" y="155" width="520" height="125" rx="6" fill="var(--ground-3)" stroke="var(--rule-2)" />
            {st === 1 && (
              <g>
                <Label x="75" y="180" tone={1} size={11} bold kind={isOverBudget ? 'error' : 'compute'}>
                  {isOverBudget ? 'CRITICAL: Context Window Budget Exceeded!' : 'Context Expanding Linearly:'}
                </Label>
                <Label x="75" y="202" tone={2} size={10} mono={false}>
                  Current length: {currentSize} tokens. At turn 20, context reaches {staticPrefix + 20 * tokensPerTurn} tokens.
                </Label>
                <Label x="75" y="222" tone={2} size={10} mono={false}>
                  Inference penalties: Prefill latency (TTFT) grows by {((currentSize / staticPrefix)).toFixed(1)}x; KV cache fills GPU VRAM.
                </Label>
                <Label x="75" y="242" tone={3} size={9.5} mono={false}>
                  Risk: If context exceeds the model limit (or allocated paged blocks), the engine raises an OOM crash.
                </Label>
              </g>
            )}

            {st === 2 && (
              <g>
                <Label x="75" y="180" tone={1} size={11} bold kind={turnCount > 8 ? 'error' : 'compute'}>
                  {turnCount > 8 ? 'FATAL: System Prompt & User Goal Evicted!' : 'Sliding Window Active (FIFO):'}
                </Label>
                <Label x="75" y="202" tone={2} size={10} mono={false}>
                  FIFO truncation drops the oldest tokens. At turn {turnCount}, the initial user goal was pushed out of the window!
                </Label>
                <Label x="75" y="222" tone={2} size={10} mono={false}>
                  Result: The agent forgets who it is, what problem it was solving, and why it is calling tools.
                </Label>
                <Label x="75" y="242" tone={3} size={9.5} mono={false}>
                  Rule: NEVER use simple FIFO truncation on agent message histories.
                </Label>
              </g>
            )}

            {(st === 3 || st === 4 || st === 5) && (
              <g>
                <Label x="75" y="180" tone={1} size={11} bold kind="token">
                  The Production Hybrid Memory Pattern:
                </Label>
                <Label x="75" y="202" tone={2} size={10} mono={false}>
                  1. <tspan fill={COLOR.compute}>Anchor Head:</tspan> System prompt + original user goal (NEVER truncated).
                </Label>
                <Label x="75" y="222" tone={2} size={10} mono={false}>
                  2. <tspan fill={COLOR.token}>Compaction Buffer:</tspan> Fast background summarization turns past 10 turns into a dense state JSON.
                </Label>
                <Label x="75" y="242" tone={2} size={10} mono={false}>
                  3. <tspan fill={COLOR.memory}>Raw Recent Tail:</tspan> Exact recent 3-5 turns preserved for step-by-step reasoning.
                </Label>
                <Label x="75" y="262" tone={3} size={9.5} mono={false}>
                  Total Context: Capped at ~{summarySize} tokens regardless of whether the trajectory runs 10, 50, or 100 turns!
                </Label>
              </g>
            )}
          </g>
        )}
      </svg>
    </Stage>
  );
}
