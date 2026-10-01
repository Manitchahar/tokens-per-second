import { useEffect, useRef, useState } from 'react';
import { Stage, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade, Arrow, Span } from '../lib/svg';
import { ATTACK_SURFACE } from './orMetrics';

// An inference service is a very unusual target: it accepts arbitrary text, holds
// unbounded state on your GPU, and can be made to output your weights. This visual
// walks the request path and names the attack at each hop.

const STEPS = [
  'Walk the request path',
  'The edge: who is calling',
  'Amplification at the gateway',
  'The router and the cache',
  'The engine: quotas are security',
  'The KV cache is memory with a lifetime',
  'The weights are the asset',
  'Your logs are a copy of the weights',
  'Layered defences, in order',
];
const CAPTIONS = [
  'A request crosses seven hops. Each is a place where an attacker can take something, and each has a control that costs almost nothing if you put it in early.',
  'The edge accepts arbitrary bytes from anyone with a key. Without per-key quotas, one key can consume your entire GPU and you will find out on the invoice.',
  'A client call that your gateway expands into sixty backend calls is an amplification primitive. Nobody needs to attack your engine if they can attack the thing in front of it.',
  'The prefix cache is keyed by token blocks. If the key does not include the tenant, one tenant can read another tenant’s cached prefix — a cross-tenant confidentiality bug with no attacker skill required.',
  'Concurrency, context length and token budgets are security controls. A per-tenant cap means one tenant cannot consume the KV pool that the other forty-nine are paying for.',
  'KV cache entries outlive the request that created them. Reuse without zeroing, or scope without a tenant namespace, and you have handed a slot from one conversation to another.',
  'Sampling a model many times produces its weights — this is a known, demonstrated result. So the weights are exfiltratable by anyone who can ask enough well-chosen questions.',
  'Prompts, completions, traces and error bodies all contain the conversation. Redaction is not hygiene; for a data-protection regime it is the control that keeps you compliant.',
  'Authentication, then quotas, then admission control, then isolation, then encryption, then logging discipline. Cheap controls first, so the expensive ones are never the thing standing between you and an incident.',
];

const SEV_COLOR = { 1: 'var(--memory)', 2: 'var(--comm)', 3: 'var(--error)' } as const;

export default function TrustBoundary() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [focus, setFocus] = useState('edge');

  useEffect(() => {
    setFocus(ATTACK_SURFACE[Math.min(s.step, ATTACK_SURFACE.length - 1)].id);
  }, [s.step]);

  const hops = ATTACK_SURFACE;
  const active = hops.find((h) => h.id === focus) ?? hops[0];

  const Y0 = 62;
  const ROW = 36;
  const W = 268;

  return (
    <Stage
      rootRef={ref}
      title="Seven hops, seven places to lose something"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['error', 'loses money or data'], ['comm', 'moderate'], ['memory', 'low'], ['token', 'the control']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Hops with a named control', value: `${hops.length} / ${hops.length}`, kind: 'token', note: 'every one is fixable' },
          { label: 'Current focus', value: active.name, kind: active.severity === 3 ? 'error' : 'memory', note: `severity ${active.severity}/3` },
          { label: 'Cheapest missing control', value: focus === 'client' || focus === 'edge' ? 'per-key quota' : focus === 'router' ? 'tenant in cache key' : focus === 'kv' ? 'zero on reuse' : 'redact logs', kind: 'comm', note: 'minutes of work' },
        ]} />
      }
      controls={
        <Segmented
          label="Inspect a hop"
          value={focus}
          options={hops.map((h) => ({ value: h.id, label: h.id }))}
          onChange={(v) => setFocus(v)}
        />
      }
    >
      <svg viewBox="0 0 640 400" role="img" aria-label="A diagram of the request path through an inference service with seven hops, each labelled with what an attacker controls, what the threat is, and the defence.">
        {/* the path */}
        <Label x={16} y={18} tone={1} size={11} bold mono={false}>The request path, and what each hop is worth to an attacker</Label>
        <Label x={624} y={18} anchor="end" size={9.5} tone={3}>select a hop to read its threat and its control</Label>

        {hops.map((h, i) => {
          const y = Y0 + i * ROW;
          const on = h.id === focus;
          const col = SEV_COLOR[h.severity];
          return (
            <g key={h.id} className="v-fade">
              {i > 0 && <Arrow x1={44} y1={y - ROW + 14} x2={44} y2={y - 4} kind="text" muted />}
              <rect x={16} y={y} width={W} height={ROW - 6} rx={3}
                fill={on ? 'var(--ground-3)' : 'transparent'}
                stroke={on ? col : 'var(--rule)'} strokeWidth={on ? 1.4 : 1} className="v-cell" />
              <circle cx={32} cy={y + 13} r={5} fill={col} opacity={on ? 1 : 0.45} />
              <Label x={48} y={y + 17} size={10} mono={false}
                style={{ fill: on ? 'var(--text)' : 'var(--text-2)', fontWeight: on ? 700 : 400 }}>{h.name}</Label>
              <Label x={16 + W - 8} y={y + 17} anchor="end" size={8.5} kind={h.severity === 3 ? 'error' : 'text'}>{`sev ${h.severity}`}</Label>
            </g>
          );
        })}
        <line x1={16} y1={Y0 - 8} x2={16} y2={Y0 + hops.length * ROW - 8} stroke="var(--rule-2)" strokeWidth={1} />

        {/* the detail panel for the selected hop */}
        <line x1={300} y1={Y0 - 8} x2={300} y2={Y0 + hops.length * ROW - 8} className="v-rule faint" />

        <Label x={316} y={Y0 + 4} size={9} tone={3}>attacker controls</Label>
        <text x={316} y={Y0 + 20} className="v-label t1" style={{ fontSize: 10 }}>
          {active.attackerInput}
        </text>

        <Label x={316} y={Y0 + 44} size={9} tone={3}>threat</Label>
        <text x={316} y={Y0 + 60} className="v-label" style={{ fontSize: 10, fill: 'var(--error)' }}>
          {active.threat}
        </text>

        <Label x={316} y={Y0 + 84} size={9} kind="token">control</Label>
        <text x={316} y={Y0 + 100} className="v-label t1" style={{ fontSize: 10 }}>
          {active.control}
        </text>
        <Span x={316} y={Y0 + 110} w={308} h={3} kind="token" />

        {/* worked example, always visible */}
        <line x1={16} y1={322} x2={624} y2={322} className="v-rule faint" />
        <Label x={16} y={342} tone={2} size={10.5} bold mono={false}>Worked: one unauthenticated key against a 24-slot replica</Label>
        <Label x={16} y={360} size={9.5} tone={2}>{'capacity is 24 slots ÷ 1.89 s = 12.7 req/s. At $2.50 per GPU-hour that is $0.22 per hour.'}</Label>
        <Label x={16} y={376} size={9.5} tone={3}>{'A key with no per-key concurrency cap can hold all 24 slots with 8-second requests: 3 req/s, a quarter of it.'}</Label>
        <Label x={16} y={392} size={9.5} kind="error" bold>{'That is 10,800 requests per hour of your GPU, billing at $0.22/h, before anyone else arrives.'}</Label>
      </svg>
    </Stage>
  );
}