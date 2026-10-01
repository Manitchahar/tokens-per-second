import { useEffect, useRef, useState } from 'react';
import { Stage, Readout, Segmented } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade, Chip, Move } from '../lib/svg';

/** Split a sentence into two balanced lines so it fits the note box. */
function twoLines(text: string): [string, string] {
  const words = text.split(' ');
  const mid = Math.ceil(words.length / 2);
  return [words.slice(0, mid).join(' '), words.slice(mid).join(' ')];
}

const STEPS = [
  'The title is a claim',
  'The abstract is a search query',
  'Find the roofline, not the throughput',
  'Check the setup is yours',
  'Read the threats to validity',
  'Trace the money: cost & code',
  'The one-page verdict',
];
const CAPTIONS = [
  'An inference systems paper title is a claim about which speed limit you are hitting. Read it as a hypothesis.',
  'The abstract is for people deciding whether to keep reading. Skip it first if you are skimming, return to it last if you are reproducing.',
  'The setup section tells you which of the two speed limits the work attacks: memory or compute.',
  'Reproduce only papers whose hardware and workload you actually have. Most failures to reproduce are mismatched setups.',
  'Every paper has a threats-to-validity section. It is a list of things that would break the result, written by the authors.',
  'Follow the artifacts: code, weights, and the cost of running it. A paper without code is a wish.',
  'Write the verdict before you do the work. One page: claim, setup, evidence, threats, cost, and the sentence you would say to a colleague.',
];

/** The reader's job in each step, shown as a checklist the reader fills in. */
type Lens = 'skim' | 'deep' | 'reproduce';

const VERDICT_LINES = [
  { k: 'claim', t: 'Claim in one sentence, with the number it beats.' },
  { k: 'setup', t: 'Setup match: same GPU? same model? same workload?' },
  { k: 'evidence', t: 'Evidence: is the win inside run-to-run noise?' },
  { k: 'threats', t: 'Threats: what would break this, per the authors.' },
  { k: 'cost', t: 'Cost to adopt, and whether the code exists.' },
];

const W = 640;
const PAGE_W = 300;

export default function PaperReader() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [lens, setLens] = useState<Lens>('skim');
  useEffect(() => {
    const m = ['skim', 'deep', 'reproduce'] as Lens[];
    setLens(m[Math.min(s.step, m.length - 1)]);
  }, [s.step]);

  // sections of a paper; each becomes "lit" at a given step
  const sections = [
    { name: 'Title & abstract', y: 0, height: 46, steps: [0, 1] },
    { name: 'Introduction', y: 52, height: 34, steps: [1] },
    { name: 'Background & problem', y: 92, height: 40, steps: [2] },
    { name: 'Method (the idea)', y: 138, height: 46, steps: [2, 3] },
    { name: 'Experimental setup', y: 190, height: 40, steps: [3] },
    { name: 'Results', y: 236, height: 46, steps: [4] },
    { name: 'Threats to validity', y: 288, height: 30, steps: [5] },
    { name: 'Related work & conclusion', y: 324, height: 26, steps: [6] },
  ];
  const active = new Set(sections.flatMap((x) => (x.steps.includes(s.step) ? [x.name] : [])));

  // how many verdict lines are filled at this step
  const filled = Math.min(VERDICT_LINES.length, s.step + 1);

  const lensNote =
    lens === 'skim'
      ? 'Skim: title, abstract, results table, conclusion. About ten minutes, and enough to know whether this is worth your week.'
      : lens === 'deep'
        ? 'Deep: method, setup, threats. Ask what the mechanism is and whether it would hold on your traffic.'
        : 'Reproduce: setup, artifacts, every number you must regenerate. Write the plan before you open a terminal.';
  const [lensNoteA, lensNoteB] = twoLines(lensNote);

  return (
    <Stage
      rootRef={ref}
      title="Reading an inference systems paper: what to look for, and when"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', ['token', 'what you extract'], ['error', 'noise / mismatch'], 'idle']}
      {...s}
      readout={
        <Readout items={[
          { label: 'Time budget', value: lens === 'skim' ? '~10 min' : lens === 'deep' ? '~1 hour' : '~1 day', kind: 'text', note: `${lens} lens` },
          { label: 'Verdict lines filled', value: `${filled} / ${VERDICT_LINES.length}`, kind: 'token', note: 'the artifact you keep' },
        ]} />
      }
      controls={<Segmented label="Reading lens" value={lens} onChange={setLens} options={[{ value: 'skim', label: 'Skim' }, { value: 'deep', label: 'Deep' }, { value: 'reproduce', label: 'Reproduce' }]} />}
    >
      <svg viewBox={`0 0 ${W} 372`} role="img" aria-label="A paper laid out as sections, with the sections relevant to the current reading step highlighted, next to a verdict sheet that fills in as you read.">
        {/* the paper page */}
        <Label x={PAGE_W / 2} y={-2} anchor="middle" tone={2} size={11} mono={false} bold>The paper, section by section</Label>
        {sections.map((sec) => {
          const on = active.has(sec.name);
          return (
            <g key={sec.name} className="v-fade" style={{ opacity: on ? 1 : 0.28 }}>
              <rect
                x={20}
                y={sec.y}
                width={PAGE_W}
                height={sec.height}
                rx={3}
                fill={on ? 'var(--token)' : 'var(--ground-3)'}
                stroke={on ? 'var(--token)' : 'var(--rule-2)'}
                strokeWidth={on ? 1.4 : 1}
              />
              {/* fake text lines */}
              {Array.from({ length: Math.floor(sec.height / 9) }, (_, i) => (
                <line
                  key={i}
                  x1={30}
                  x2={PAGE_W + 10 - ((i * 37) % 60)}
                  y1={sec.y + 12 + i * 9}
                  y2={sec.y + 12 + i * 9}
                  stroke={on ? 'rgba(14,17,22,0.5)' : 'var(--rule-2)'}
                  strokeWidth={1.4}
                />
              ))}
              <text x={28} y={sec.y + sec.height / 2 + 4} className="v-label bold" style={{ fontSize: 10.5, fill: on ? '#0e1116' : 'var(--text-2)' }}>
                {sec.name}
              </text>
            </g>
          );
        })}
        {/* the lens arrow showing reading mode */}
        <Move x={PAGE_W + 18} y={18}>
          <Chip x={0} y={0} text={lens.toUpperCase()} kind={lens === 'reproduce' ? 'error' : 'compute'} />
        </Move>

        {/* verdict sheet */}
        <Label x={PAGE_W + 34 + 130} y={-2} anchor="middle" tone={2} size={11} mono={false} bold>Your one-page verdict</Label>
        {VERDICT_LINES.map((v, i) => {
          const y = 14 + i * 40;
          const on = i < filled;
          return (
            <g key={v.k} className="v-fade" style={{ opacity: on ? 1 : 0.3 }}>
              <rect x={PAGE_W + 34} y={y} width={272} height={32} rx={3} fill="none" stroke={on ? 'var(--token)' : 'var(--rule-2)'} strokeDasharray={on ? undefined : '3 3'} />
              {/* checkbox */}
              <rect x={PAGE_W + 44} y={y + 10} width={12} height={12} rx={2} fill={on ? 'var(--token)' : 'none'} stroke={on ? 'var(--token)' : 'var(--rule-2)'} />
              {on && (
                <path
                  d={`M${PAGE_W + 46},${y + 16} L${PAGE_W + 49},${y + 19.5} L${PAGE_W + 55},${y + 12}`}
                  stroke="#0e1116"
                  strokeWidth={2}
                  fill="none"
                  strokeLinecap="round"
                />
              )}
              <text x={PAGE_W + 64} y={y + 20} className="v-label sans" style={{ fontSize: 10.5, fill: on ? 'var(--text)' : 'var(--text-3)' }}>
                {v.t}
              </text>
            </g>
          );
        })}

        {/* lens note under the sheet */}
        <Fade show={lens !== 'skim' || s.step >= 1}>
          <rect x={PAGE_W + 34} y={228} width={272} height={30} rx={3} fill="var(--compute-dim)" stroke="var(--compute)" />
          <text x={PAGE_W + 44} y={241} className="v-label sans" style={{ fontSize: 10, fill: 'var(--text)' }}>
            {lensNoteA}
          </text>
          <text x={PAGE_W + 44} y={252} className="v-label sans" style={{ fontSize: 10, fill: 'var(--text)' }}>
            {lensNoteB}
          </text>
        </Fade>

        {/* step counter strip */}
        <text x={W / 2} y={368} textAnchor="middle" className="v-label t3" style={{ fontSize: 9.5 }}>
          Fill the verdict sheet before you open a terminal. The artifact you keep is one page, not a PDF full of highlights.
        </text>
      </svg>
    </Stage>
  );
}