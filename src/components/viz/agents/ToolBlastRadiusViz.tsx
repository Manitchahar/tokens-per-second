import { useRef, useState } from 'react';
import { Stage, Slider, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Arrow, Fade, Span, Chip, COLOR } from '../lib/svg';

const STEPS = [
  'The tool spectrum: Read, Compute, Write',
  'Blast radius: Mutation vs Idempotency',
  'JSON Schema: Token weight in the prefix',
  'Constrained decoding: Grammar enforcement',
  'The Model Context Protocol (MCP) abstraction',
  'Human-in-the-loop: The approval checkpoint',
  'Security & Blast Radius Matrix',
];

const CAPTIONS = [
  'Not all tools are created equal. Knowledge tools read facts safely; compute tools run sandboxed calculations; write actions permanently mutate databases, cloud infrastructure, or money.',
  'Blast radius defines the maximum damage of an erroneous tool call. Read tools have zero persistent side effects. Write actions cannot be undone by hitting backspace.',
  'Tool calling is not magic: every tool description and JSON Schema must be injected into the system prompt. Ten schemas can consume 1,500+ tokens before the user even speaks.',
  'During decode, standard sampling risks malformed JSON syntax. Constrained grammars (outlines / regex masking) force the model logits to conform strictly to the schema.',
  'The Model Context Protocol (MCP) standardizes how models connect to external data: Host (IDE/App), Client (Orchestrator), and Server (DB/GitHub/Terminal).',
  'High-risk write actions require a Human-In-The-Loop (HITL) gate. The agent pauses, presents a structured dry-run diff, and awaits explicit authorization.',
  'Systematic tool categorization: evaluate blast radius, idempotency, privilege level, and token cost to harden production agent deployments.',
];

export default function ToolBlastRadiusViz() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;

  // Interactive controls for blast radius inspection
  const [selectedTier, setSelectedTier] = useState<0 | 1 | 2>(1); // 0: Read, 1: Compute, 2: Write
  const [schemaCount, setSchemaCount] = useState(6);

  const tokensPerSchema = 185;
  const totalSchemaTokens = schemaCount * tokensPerSchema;

  const tiers = [
    {
      name: 'Tier 1: Read (Knowledge)',
      kind: 'memory' as const,
      examples: 'vector_search, db_select, get_weather',
      sideEffects: 'None (Idempotent)',
      blastRadius: 'Low (Data leakage risk)',
      approval: 'Autonomous (Auto-run)',
    },
    {
      name: 'Tier 2: Compute (Capability)',
      kind: 'compute' as const,
      examples: 'python_sandbox, ast_eval, wolfram',
      sideEffects: 'Transient CPU/RAM',
      blastRadius: 'Medium (Timeouts, OOM)',
      approval: 'Autonomous with strict sandbox',
    },
    {
      name: 'Tier 3: Write (Action)',
      kind: 'error' as const,
      examples: 'sql_delete, aws_terminate, send_payment',
      sideEffects: 'Permanent State Mutation',
      blastRadius: 'CRITICAL (Data loss, Financial)',
      approval: 'MANDATORY HITL Gate',
    },
  ];

  const curr = tiers[selectedTier];

  return (
    <Stage
      rootRef={ref}
      title="Tools: Taxonomy, Blast Radius & Schema Overhead"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['memory', 'read / knowledge'],
        ['compute', 'compute / sandbox'],
        ['error', 'write / high blast radius'],
        ['token', 'grammar / schema'],
      ]}
      readout={
        <Readout
          items={[
            {
              label: 'Selected Category',
              value: st <= 1 || st === 6 ? curr.name.split(':')[1].trim() : st === 2 ? 'JSON Schema' : st === 3 ? 'Grammar' : st === 4 ? 'MCP Protocol' : 'HITL Gate',
              kind: curr.kind,
              note: curr.sideEffects,
            },
            {
              label: 'Tool Schemas',
              value: `${schemaCount} tools`,
              kind: 'text',
              note: `~${tokensPerSchema} tokens each`,
            },
            {
              label: 'Prefix Injection',
              value: `${totalSchemaTokens} tokens`,
              kind: 'token',
              note: `${(totalSchemaTokens * 2 * 8e9 / 1e12).toFixed(2)} TFLOPs (8B prefill)`,
            },
            {
              label: 'Safety Policy',
              value: selectedTier === 2 ? 'HITL Required' : 'Autonomous',
              kind: selectedTier === 2 ? 'error' : 'token',
              note: curr.approval,
            },
          ]}
        />
      }
      controls={
        st <= 1 || st === 6 ? (
          <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
            <span style={{ fontSize: 11, color: 'var(--text-2)' }}>Privilege Tier:</span>
            {[0, 1, 2].map((idx) => (
              <button
                key={idx}
                type="button"
                className={`t-btn ${selectedTier === idx ? 'active' : ''}`}
                style={{
                  padding: '4px 12px',
                  borderRadius: 4,
                  fontSize: 11,
                  background: selectedTier === idx ? 'var(--ground-3)' : 'transparent',
                  border: `1px solid ${selectedTier === idx ? COLOR[tiers[idx].kind] : 'var(--rule)'}`,
                  color: selectedTier === idx ? COLOR[tiers[idx].kind] : 'var(--text-2)',
                  cursor: 'pointer',
                }}
                onClick={() => setSelectedTier(idx as 0 | 1 | 2)}
              >
                {tiers[idx].name.split(':')[0]}
              </button>
            ))}
          </div>
        ) : st === 2 ? (
          <Slider
            label="Injected Tool Schemas"
            value={schemaCount}
            min={1}
            max={20}
            step={1}
            format={(v) => `${v} tools (${v * tokensPerSchema} tokens)`}
            onChange={setSchemaCount}
          />
        ) : undefined
      }
      {...s}
    >
      <svg viewBox="0 0 640 320" role="img" aria-label="Tool classification spectrum and blast radius architecture">
        {/* Step 0 & 1 & 6: The Three Privilege Tiers */}
        {(st <= 1 || st === 6) && (
          <g>
            {/* The 3 Cards */}
            {tiers.map((t, idx) => {
              const isSel = selectedTier === idx;
              const yPos = 25 + idx * 90;
              return (
                <g key={t.name} onClick={() => setSelectedTier(idx as 0 | 1 | 2)} style={{ cursor: 'pointer' }}>
                  <rect
                    x="50"
                    y={yPos}
                    width="540"
                    height="80"
                    rx="6"
                    fill={isSel ? 'var(--ground-3)' : 'var(--ground-2)'}
                    stroke={isSel ? COLOR[t.kind] : 'var(--rule-2)'}
                    strokeWidth={isSel ? 2 : 1}
                  />
                  <Chip x="65" y={yPos + 12} text={t.name} kind={t.kind} mono={false} size={11} />
                  <Label x="260" y={yPos + 26} tone={2} size={10.5}>
                    Examples: <tspan fill="var(--text-1)">{t.examples}</tspan>
                  </Label>
                  <Label x="65" y={yPos + 52} tone={3} size={10} mono={false}>
                    Side Effects: <tspan fill={COLOR[t.kind]}>{t.sideEffects}</tspan>
                  </Label>
                  <Label x="300" y={yPos + 52} tone={3} size={10} mono={false}>
                    Blast Radius: <tspan fill="var(--text-1)">{t.blastRadius}</tspan>
                  </Label>
                  <Label x="65" y={yPos + 68} tone={2} size={9.5} mono={false}>
                    Policy: <tspan fill={t.kind === 'error' ? COLOR.error : COLOR.token}>{t.approval}</tspan>
                  </Label>
                </g>
              );
            })}
          </g>
        )}

        {/* Step 2: JSON Schema Token Weight in Prompt */}
        {st === 2 && (
          <g>
            <rect x="40" y="20" width="560" height="280" rx="8" fill="var(--ground-2)" stroke="var(--rule-2)" />
            <Label x="60" y="45" tone={1} size={13} bold>The Invisible Tax: Tool Schemas in Context</Label>
            <Label x="60" y="65" tone={2} size={10.5} mono={false}>
              Every function definition adds ~150-250 tokens of JSON Schema boilerplate to the system prompt.
            </Label>

            {/* Visual Token Stream */}
            <Label x="60" y="95" tone={2} size={10} bold>Prompt Layout with {schemaCount} Tools:</Label>
            <Span x="60" y="105" w="120" h="28" kind="compute" label="System Prompt (250 T)" />
            <Span x="185" y="105" w={Math.min(330, 15 + schemaCount * 15)} h="28" kind="token" label={`Tools JSON (${totalSchemaTokens} T)`} />
            <Span x={205 + Math.min(330, 15 + schemaCount * 15)} y="105" w="80" h="28" kind="memory" label="Goal (120 T)" />

            {/* Code example of Schema boilerplate */}
            <rect x="60" y="145" width="520" height="140" rx="6" fill="var(--ground-3)" stroke="var(--rule)" />
            <Label x="75" y="165" tone={3} size={9.5}>// JSON Schema for one simple tool (db_query):</Label>
            <Label x="75" y="182" tone={1} size={9.5}>&#123;"name": "db_query", "description": "Execute read-only SQL SELECT",</Label>
            <Label x="75" y="199" tone={1} size={9.5}>  "parameters": &#123;"type": "object", "properties": &#123;</Label>
            <Label x="75" y="216" tone={1} size={9.5}>    "query": &#123;"type": "string", "description": "SQL SELECT query string"&#125;,</Label>
            <Label x="75" y="233" tone={1} size={9.5}>    "timeout_ms": &#123;"type": "integer", "default": 5000&#125;&#125;,</Label>
            <Label x="75" y="250" tone={1} size={9.5}>    "required": ["query"]&#125;&#125;  <tspan fill={COLOR.token}>← 185 tokens for 1 tool</tspan></Label>
            <Label x="75" y="272" tone={2} size={9.5} mono={false}>
              At 10 turns without prefix caching, {schemaCount} tools cost {schemaCount * 185 * 10} extra prefill tokens!
            </Label>
          </g>
        )}

        {/* Step 3: Constrained Decoding & Grammars */}
        {st === 3 && (
          <g>
            <rect x="40" y="20" width="560" height="280" rx="8" fill="var(--ground-2)" stroke="var(--rule-2)" />
            <Label x="60" y="45" tone={1} size={13} bold>Constrained Decoding: Eliminating Syntax Hallucinations</Label>
            <Label x="60" y="65" tone={2} size={10.5} mono={false}>
              An LLM emits logits over 128,000 vocabulary tokens. Without constraints, unescaped quotes break JSON parsers.
            </Label>

            {/* Unconstrained vs Constrained */}
            <rect x="60" y="90" width="250" height="190" rx="6" fill="var(--ground-3)" stroke="var(--error)" strokeWidth="1" />
            <Chip x="75" y="105" text="Standard Sampling (Fragile)" kind="error" mono={false} size={10.5} />
            <Label x="75" y="140" tone={2} size={10} mono={false}>All 128k vocab tokens allowed.</Label>
            <Label x="75" y="160" tone={1} size={9.5}>1. Generates: &#123;"query": "SELECT...</Label>
            <Label x="75" y="178" tone={1} size={9.5}>2. Decodes unescaped newline \n</Label>
            <Label x="75" y="196" tone={1} size={9.5}>3. json.loads() raises JSONDecodeError!</Label>
            <Label x="75" y="225" tone={3} size={9.5} mono={false}>Result: Agent crashes mid-flight.</Label>

            <rect x="330" y="90" width="250" height="190" rx="6" fill="var(--ground-3)" stroke="var(--token)" strokeWidth="1.5" />
            <Chip x="345" y="105" text="Grammar / Outlines (Guaranteed)" kind="token" mono={false} size={10.5} />
            <Label x="345" y="140" tone={2} size={10} mono={false}>Logit mask applied before sampling:</Label>
            <Label x="345" y="160" tone={1} size={9.5}>Logits[invalid_char] = -∞</Label>
            <Label x="345" y="180" tone={1} size={9.5}>Deterministic FSM state transitions.</Label>
            <Label x="345" y="200" tone={1} size={9.5}>Guaranteed 100% valid JSON syntax.</Label>
            <Label x="345" y="225" tone={3} size={9.5} mono={false}>Result: Zero parser syntax failures.</Label>
          </g>
        )}

        {/* Step 4: MCP Architecture */}
        {st === 4 && (
          <g>
            <rect x="40" y="20" width="560" height="280" rx="8" fill="var(--ground-2)" stroke="var(--rule-2)" />
            <Label x="60" y="45" tone={1} size={13} bold>Model Context Protocol (MCP): The Universal Tool Bus</Label>
            <Label x="60" y="65" tone={2} size={10.5} mono={false}>
              Anthropic’s open standard replaces custom API glue with a standard client-server protocol.
            </Label>

            {/* Host / Client / Server Architecture */}
            <rect x="60" y="100" width="130" height="120" rx="6" fill="var(--ground-3)" stroke="var(--rule-2)" />
            <Label x="125" y="125" anchor="middle" tone={1} size={12} bold>MCP Host</Label>
            <Label x="125" y="145" anchor="middle" tone={3} size={10} mono={false}>Claude Desktop</Label>
            <Label x="125" y="160" anchor="middle" tone={3} size={10} mono={false}>IDE / Custom App</Label>
            <Label x="125" y="190" anchor="middle" tone={2} size={9.5}>Manages UX & auth</Label>

            <Arrow x1="195" y1="160" x2="245" y2="160" />

            <rect x="250" y="100" width="140" height="120" rx="6" fill="var(--ground-3)" stroke="var(--token)" strokeWidth="1.5" />
            <Label x="320" y="125" anchor="middle" tone={1} size={12} bold>MCP Client</Label>
            <Label x="320" y="145" anchor="middle" tone={3} size={10} mono={false}>Agent Core Orchestrator</Label>
            <Label x="320" y="165" anchor="middle" tone={2} size={9.5}>Discovers tool schemas</Label>
            <Label x="320" y="185" anchor="middle" tone={2} size={9.5}>Routes JSON-RPC calls</Label>

            <Arrow x1="395" y1="160" x2="445" y2="160" />

            <rect x="450" y="90" width="130" height="140" rx="6" fill="var(--ground-3)" stroke="var(--rule-2)" />
            <Label x="515" y="115" anchor="middle" tone={1} size={12} bold>MCP Servers</Label>
            <Chip x="465" y="130" text="Postgres DB" kind="memory" mono={false} size={9} />
            <Chip x="465" y="158" text="GitHub API" kind="compute" mono={false} size={9} />
            <Chip x="465" y="186" text="Shell Exec" kind="error" mono={false} size={9} />

            <rect x="60" y="240" width="520" height="45" rx="6" fill="var(--ground-2)" stroke="var(--rule)" />
            <Label x="80" y="267" tone={2} size={10} mono={false}>
              Protocol primitive: Standardized JSON-RPC 2.0 messages over stdin/stdout or Server-Sent Events (SSE).
            </Label>
          </g>
        )}

        {/* Step 5: Human-In-The-Loop Approval Gate */}
        {st === 5 && (
          <g>
            <rect x="40" y="20" width="560" height="280" rx="8" fill="var(--ground-2)" stroke="var(--rule-2)" />
            <Label x="60" y="45" tone={1} size={13} bold>Human-In-The-Loop (HITL): The Gate for Write Actions</Label>
            <Label x="60" y="65" tone={2} size={10.5} mono={false}>
              Autonomous agents must pause before executing state mutations. Inspect the dry-run diff before approval.
            </Label>

            {/* Agent proposing action */}
            <rect x="60" y="95" width="200" height="120" rx="6" fill="var(--ground-3)" stroke="var(--rule-2)" />
            <Chip x="75" y="110" text="Agent Proposal" kind="token" mono={false} size={10} />
            <Label x="75" y="145" tone={1} size={9.5}>tool: delete_database_replica</Label>
            <Label x="75" y="165" tone={2} size={9.5}>cluster: "prod-us-east-1"</Label>
            <Label x="75" y="185" tone={3} size={9.5}>reason: "Recover from deadlock"</Label>

            <Arrow x1="265" y1="155" x2="315" y2="155" />

            {/* Checkpoint Gate */}
            <rect x="320" y="90" width="130" height="130" rx="6" fill="var(--ground-3)" stroke="var(--error)" strokeWidth="2" />
            <Label x="385" y="115" anchor="middle" tone={1} size={12} bold kind="error">HITL GATE</Label>
            <Label x="385" y="138" anchor="middle" tone={2} size={9.5} mono={false}>Blast Radius: HIGH</Label>
            <rect x="335" y="152" width="100" height="24" rx="4" fill="var(--error)" />
            <Label x="385" y="168" anchor="middle" tone={1} size={10} bold style={{ fill: '#0e1116' }}>Awaiting Human</Label>
            <Label x="385" y="195" anchor="middle" tone={3} size={9} mono={false}>Diff presented</Label>

            <Arrow x1="455" y1="155" x2="500" y2="155" />

            {/* Environment */}
            <rect x="505" y="95" width="80" height="120" rx="6" fill="var(--ground-3)" stroke="var(--rule-2)" />
            <Label x="545" y="145" anchor="middle" tone={2} size={11} bold>Target</Label>
            <Label x="545" y="165" anchor="middle" tone={3} size={9.5}>Prod DB</Label>

            <rect x="60" y="235" width="525" height="50" rx="6" fill="var(--ground-3)" stroke="var(--rule)" />
            <Label x="75" y="255" tone={2} size={10} mono={false}>
              Principle of Least Privilege: Agents should NEVER hold root/write credentials directly.
            </Label>
            <Label x="75" y="272" tone={2} size={10} mono={false}>
              The environment only issues scoped, short-lived mutation tokens upon cryptographic human sign-off.
            </Label>
          </g>
        )}
      </svg>
    </Stage>
  );
}
