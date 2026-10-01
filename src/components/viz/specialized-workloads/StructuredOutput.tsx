import { useMemo, useRef, useState } from 'react';
import { Stage, Segmented, Readout, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Chip, Arrow, Fade, Dim, Bar, fmt } from '../lib/svg';
import {
  GRAMMAR_TRACE, GRAMMARS, TOOL_CALL, toolCallCost,
  type GrammarId,
} from './workloads';

const STEPS = [
  'Free text: the whole vocabulary is open',
  'A grammar turns sampling into walking a state machine',
  'The mask: one bit per token, recomputed each step',
  'Alternating narrow and wide',
  'Pinning a schema: the allowed set collapses',
  'Why constrained decoding is (almost) free',
  'A tool call, token by token',
  'What the caller must still validate',
];
const CAPTIONS = [
  'Ordinary generation: every step draws a token from the full vocabulary. Nothing constrains where the answer can go, so a stray token can end the JSON half-written — and it is the single most common way a tool-using model fails.',
  'Add a grammar — a JSON Schema, a regex, a grammar for a language — and the engine builds a second object alongside the model: a mask saying which tokens are legal right now. Sampling then intersects the two.',
  'The mask is one bit per vocabulary entry, computed from the parser state on the fly. Applying it is an elementwise AND with the logits: a few tens of microseconds on any modern GPU, against a decode step measured in milliseconds.',
  'A JSON grammar is not uniformly tight. Right after a character that ends a key or a value, almost nothing is legal — under 0.1% of a 128K vocabulary. Inside a string, most tokens are legal. The mask breathes.',
  'With a full schema the mask never opens up much: an enum field allows exactly its members. That is the point of pinning a schema — you trade freedom for a guarantee the model cannot violate.',
  'Constrained decoding removes failure modes, not tokens. Each emitted token is still a full decode step with the same weights read. It makes generation *correct*, not *faster*.',
  'A tool call is JSON with a name and an argument object. Walk it and watch which fields are pinned by the schema and which are free text — the free text is where the model actually spends its thinking.',
  'Everything the grammar cannot know: does the city exist, is the ID real, is the user allowed to do this. Validation is still your job, and it belongs in the same service that parses the call.',
];

export default function StructuredOutput() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;

  const [gid, setGid] = useState<GrammarId>('json');
  const [nArgs, setNArgs] = useState(2);

  const g = GRAMMARS.find((x) => x.id === gid)!;
  const cost = useMemo(() => toolCallCost(nArgs), [nArgs]);

  // The walk through the trace: a sliding window of the grammar trace.
  const trace = GRAMMAR_TRACE;
  const cur = trace[Math.min(trace.length - 1, st === 3 ? 4 : st >= 4 ? 5 : 1)];
  const allowedTokens = Math.round(128256 * cur.allowedFrac);

  // A fake vocabulary strip: which of 24 representative token groups are allowed.
  const VOCAB_GROUPS = 24;
  const groups = useMemo(() => {
    // Deterministic pseudo-random membership so server and client agree.
    const out: boolean[] = [];
    for (let i = 0; i < VOCAB_GROUPS; i++) {
      const x = Math.sin(i * 12.9898 + 78.233) * 43758.5453;
      out.push(x - Math.floor(x) < cur.allowedFrac);
    }
    return out;
  }, [cur]);

  const allowedShown = groups.filter(Boolean).length;

  return (
    <Stage
      rootRef={ref}
      title="Constrained decoding: the mask beside the logits"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['compute', 'legal token'],
        ['memory', 'masked out'],
        ['token', 'the chosen token'],
        ['error', 'a token the model wanted but could not emit'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Grammar', value: g.label, kind: 'text', note: g.blurb },
          { label: 'Allowed now', value: cur.allowedFrac >= 0.01 ? `${(cur.allowedFrac * 100).toFixed(0)}%` : `${(cur.allowedFrac * 100).toFixed(3)}%`, kind: cur.allowedFrac < 0.01 ? 'compute' : 'token', note: `${fmt.int(allowedTokens)} of 128,256 tokens` },
          { label: 'Expects', value: cur.expect, kind: 'text', note: cur.note },
          { label: 'Call cost', value: `${cost.total} tokens`, kind: 'memory', note: `${(cost.payloadFrac * 100).toFixed(0)}% is real payload` },
        ]} />
      }
      controls={
        <>
          <Segmented label="Grammar" value={gid} options={GRAMMARS.map((x) => ({ value: x.id, label: x.label }))} onChange={setGid} />
          <Slider label="Arguments in the tool call" kind="memory" min={1} max={5} step={1} value={nArgs} onChange={setNArgs} format={(v) => `${v} arguments`} />
        </>
      }
    >
      <svg viewBox="0 0 640 340" role="img" aria-label="A vocabulary strip in which most tokens are masked out by a JSON grammar at each step of parsing a tool call, with the allowed fraction, the expected next token, and the token cost of the call shown below.">
        {/* ── panel A: the vocabulary strip ── */}
        <Label x={12} y={20} tone={2} size={11} bold>The mask, one cell per group of vocabulary</Label>
        <Label x={628} y={20} anchor="end" tone={3} size={9}>128,256 tokens</Label>
        {groups.map((on, i) => {
          const cw = 24;
          const x = 12 + i * (cw + 1.6);
          return (
            <g key={i}>
              <rect x={x} y={30} width={cw} height={26} rx={2}
                fill={on ? 'var(--compute)' : 'var(--memory-dim)'} stroke={on ? 'var(--compute)' : 'var(--memory)'} strokeWidth={0.8}
                className="v-cell" />
            </g>
          );
        })}
        <Label x={12} y={72} size={9.5} kind="compute" mono={false}>{`${allowedShown} of ${VOCAB_GROUPS} groups legal — ${cur.allowedFrac >= 0.01 ? `${(cur.allowedFrac * 100).toFixed(0)}%` : `${(cur.allowedFrac * 100).toFixed(3)}%`} of the vocabulary`}</Label>

        {/* ── panel B: the state machine walk ── */}
        <Fade show={st >= 1}>
          <g>
            <Label x={12} y={98} tone={2} size={11} bold>Walking the grammar</Label>
            {trace.map((t, i) => {
              const active = t === cur;
              const x = 12 + i * 40;
              const w = 34;
              const h = Math.max(3, 40 * Math.sqrt(t.allowedFrac));
              return (
                <g key={i}>
                  <rect x={x} y={150 - h} width={w} height={h} rx={2}
                    fill={active ? 'var(--token)' : 'var(--memory-dim)'}
                    stroke={active ? 'var(--token)' : 'var(--memory)'} strokeWidth={active ? 1.8 : 0.8} className="v-cell" />
                  <Label x={x + w / 2} y={163} anchor="middle" tone={3} size={7.5}>{t.expect.slice(0, 6)}</Label>
                </g>
              );
            })}
            <Dim x1={12} x2={12 + trace.length * 40} y={172} label={`${trace.length} states in one tool call`} />
            <Label x={12} y={192} tone={3} size={9.5} mono={false}>{cur.note}</Label>
          </g>
        </Fade>

        {/* ── panel C: pinned schema ── */}
        <Fade show={st >= 4}>
          <g>
            <Label x={12} y={216} tone={2} size={11} bold>What the schema pins</Label>
            {TOOL_CALL.schema.map((f, i) => (
              <g key={f.field}>
                <Chip x={12} y={224 + i * 22} text={f.field} kind="text" h={18} w={54} size={9} />
                <Chip x={70} y={224 + i * 22} text={f.type} kind="compute" h={18} w={92} size={9} />
                <Label x={168} y={237 + i * 22} tone={2} size={9.5}>{f.example}</Label>
                <Label x={300} y={237 + i * 22} tone={3} size={9}>{f.required ? 'required' : 'optional'}</Label>
              </g>
            ))}
            <Label x={12} y={296} tone={3} size={9.5} mono={false}>Enum fields are pinned to their members. String fields stay open.</Label>
          </g>
        </Fade>

        {/* ── panel D: the cost of a call ── */}
        <Fade show={st >= 6}>
          <g>
            <Label x={380} y={216} tone={2} size={11} bold>Cost of {TOOL_CALL.name}()</Label>
            <Label x={380} y={236} tone={3} size={9.5} mono={false}>boilerplate (forced by grammar)</Label>
            <Bar x={380} y={242} w={200} h={12} frac={cost.boilerplate / cost.total} kind="memory" />
            <Label x={586} y={252} anchor="end" tone={2} size={9.5}>{cost.boilerplate}</Label>
            <Label x={380} y={266} tone={3} size={9.5} mono={false}>payload (real choice)</Label>
            <Bar x={380} y={272} w={200} h={12} frac={cost.payloadFrac} kind="token" />
            <Label x={586} y={282} anchor="end" tone={2} size={9.5}>{cost.payload}</Label>
            <Label x={380} y={304} tone={3} size={9.5}>{`${cost.total} tokens per call · ${(cost.payloadFrac * 100).toFixed(0)}% payload`}</Label>
          </g>
        </Fade>

        {/* ── panel E: the "still validate" note ── */}
        <Fade show={st >= 7}>
          <g>
            <Chip x={12} y={312} text="model" kind="compute" h={18} w={44} size={9} />
            <Arrow x1={60} y1={321} x2={92} y2={321} kind="text" />
            <Chip x={96} y={312} text="grammar" kind="memory" h={18} w={52} size={9} />
            <Arrow x1={152} y1={321} x2={184} y2={321} kind="text" />
            <Chip x={188} y={312} text="valid JSON" kind="token" h={18} w={62} size={9} />
            <Arrow x1={254} y1={321} x2={286} y2={321} kind="text" />
            <Chip x={290} y={312} text="validate" kind="error" h={18} w={56} size={9} />
            <Label x={354} y={325} tone={3} size={9.5}>syntax is guaranteed; meaning is not</Label>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}