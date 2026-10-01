import { useRef, useState } from 'react';
import { Stage, Readout, Segmented, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Span, Chip, Fade, fmt } from '../lib/svg';
import { ENGINES, PROFILES, evaluate, type ProfileId } from './fairModel';

// The six items of the harness contract, in the order the lesson fixes them.
const CONTRACT = [
  'workload',
  'arrival pattern',
  'load point',
  'quality bar',
  'best settings',
  'version + warm',
];

const STEPS = [
  'Four wins, four different questions',
  'Pin the workload',
  'Pin the arrival pattern',
  'Pin the load point',
  'Pin the quality bar',
  'Let each engine keep its best settings',
  'Pin the version and measure warm',
];

const CAPTIONS = [
  'Four projects each ran a benchmark and each reported a win. But each used a different workload, a different arrival pattern, a different load, a different quality bar and a different amount of tuning, so the four numbers are measuring four different things. Until they share a contract they cannot be compared, and no conclusion drawn from them survives contact with your traffic.',
  'First item: the workload. A shared system prompt, a 24k-token document and a 128-request offline job ask completely different questions of a scheduler. Fix the request-length distribution and the ranking moves, because the engines do not share their strengths evenly across workload shapes. Nothing else about the experiment changed.',
  'Second: how requests arrive. A closed-loop generator keeps N requests in flight and issues the next one only when one finishes, so a slow engine is automatically given easier conditions. An open-loop generator issues on a fixed schedule, so a slow engine is caught by a growing queue. The same server, the same workload, opposite conclusions.',
  'Third: the load point. Load is a fraction rho of what the machine could serve with every slot permanently busy. The queue wait grows as rho/(1 - rho) times S/B, so it is nearly free at rho = 0.5 and enormous at rho = 0.9. Comparing engines at "as much as possible" compares them at different loads, and usually rewards whichever one degrades quietly.',
  'Fourth: the quality bar. Every engine must emit the same tokens. Fix the sampling parameters, pin the seed, and check the streams actually match. A "faster" engine that quietly changed the numerics is not faster — it is doing less work, and the comparison is void.',
  'Fifth: tuning. Giving every engine the same naive flags measures your flags, not the engines. Each engine must be allowed its own best settings, each one recorded. Note what that costs: it is the point at which a comparison stops being one experiment and becomes four maintenance jobs.',
  'Sixth: versions and warm state. Defaults, default backends and even default dtypes move between engine releases, so pin every version and write them down. Measure after compilation, allocation and clock ramp have finished, and report the cold start separately — never average the two together. Only now is the ranking below a fact about your traffic.',
];

export default function FairHarness() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [profileId, setProfileId] = useState<ProfileId>('chat');
  const [load, setLoad] = useState(0.6);

  const profile = PROFILES.find((p) => p.id === profileId)!;
  const rho = Math.min(Math.max(load, 0), 0.95);
  const rows = ENGINES.map((e) => evaluate(e, profile, load));
  const maxThr = Math.max(...rows.map((r) => r.throughput), 1);
  const anyFits = rows.some((r) => r.fits);

  // Step 0 shows four un-harnessed claims: the bars exist but are not comparable yet.
  const comparable = s.step >= 6;

  return (
    <Stage
      rootRef={ref}
      title="The harness contract: what must be identical before four numbers mean anything"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['token', 'meets the SLO'],
        ['error', 'misses the SLO / does not fit'],
        ['idle', 'not pinned yet'],
        ['text', 'output tokens/s'],
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Profile', value: profile.label, kind: 'text', note: `${profile.wantBatch} streams` },
          { label: 'Offered load', value: `ρ = ${rho.toFixed(2)}`, kind: rho > 0.85 ? 'error' : 'text', note: 'of measured capacity' },
          { label: 'SLO', value: `TTFT < ${profile.sloTTFTms} ms`, kind: 'token', note: `TPOT < ${profile.sloTPOTms} ms` },
          {
            label: 'Ranking valid?',
            value: comparable ? 'yes' : 'no',
            kind: comparable ? 'token' : 'error',
            note: comparable ? 'same harness on all four' : 'confounds still free',
          },
        ]} />
      }
      controls={
        <>
          <Segmented
            label="Traffic profile (item 1 of the contract)"
            value={profileId}
            options={PROFILES.map((p) => ({ value: p.id, label: p.label }))}
            onChange={setProfileId}
          />
          <Slider
            label="Offered load ρ (item 3)"
            kind={rho > 0.85 ? 'error' : 'compute'}
            min={0.3}
            max={0.95}
            step={0.01}
            value={load}
            onChange={setLoad}
            format={(v) => `ρ = ${v.toFixed(2)} of capacity`}
          />
        </>
      }
    >
      <svg viewBox="0 0 640 336" role="img" aria-label="A harness contract of six items sits above a scoreboard of four inference engines. As each item is pinned the scoreboard's bars become comparable and the ranking of engines changes with the traffic profile and the offered load. Until every item is pinned the ranking is marked not valid.">
        {/* ── the contract ─────────────────────────────────────── */}
        <Label x={40} y={18} tone={2} size={10.5} mono={false} bold>Harness contract — six things that must be identical</Label>
        {CONTRACT.map((c, i) => {
          const on = s.step >= i + 1;
          return (
            <g key={c}>
              <rect x={40 + i * 96} y={26} width={92} height={24} rx={3}
                fill={on ? 'var(--ground-3)' : 'none'}
                stroke={on ? 'var(--token)' : 'var(--idle)'} strokeWidth={1}
                strokeDasharray={on ? undefined : '3 3'} className="v-cell" />
              <text x={86 + i * 96} y={41} textAnchor="middle" className={`v-label sans ${on ? 'bold' : ''}`}
                style={{ fontSize: 9.5, fill: on ? 'var(--token)' : 'var(--text-3)' }}>
                {on ? '✓' : '·'} {c}
              </text>
            </g>
          );
        })}

        {/* ── scoreboard ───────────────────────────────────────── */}
        <Label x={40} y={72} tone={2} size={10.5} mono={false} bold>What each engine delivered, on this harness</Label>
        <Label x={600} y={72} anchor="end" tone={3} size={9}>
          {comparable ? 'comparable' : 'not comparable: every row ran its own experiment'}
        </Label>

        {rows.map((r, i) => {
          const y = 84 + i * 44;
          const lead = r.throughput === maxThr && anyFits;
          const kind = !r.fits ? 'error' : r.meets ? 'token' : 'error';
          const w = 214 * (r.throughput / maxThr);
          return (
            <g key={r.engine.id}>
              <Label x={40} y={y + 11} size={10.5} tone={1} bold>{r.engine.name}</Label>
              <Label x={40} y={y + 25} tone={3} size={8.5}>{r.engine.trade}</Label>

              {/* bar */}
              <rect x={166} y={y} width={214} height={22} rx={2} fill="none" stroke="var(--rule)" />
              {comparable ? (
                r.fits ? (
                  <rect x={166} y={y} width={Math.max(1, w)} height={22} rx={2}
                    fill={lead ? 'var(--token)' : 'var(--text)'} fillOpacity={lead ? 0.9 : 0.28}
                    stroke={lead ? 'var(--token)' : 'var(--text-2)'} className="v-cell" />
                ) : (
                  <rect x={166} y={y} width={214} height={22} rx={2} fill="url(#tps-hatch)"
                    stroke="var(--error)" className="v-cell" />
                )
              ) : (
                <rect x={166} y={y} width={214} height={22} rx={2} fill="url(#tps-hatch)"
                  stroke="var(--idle)" className="v-cell" />
              )}

              <Label x={390} y={y + 15} size={10.5} tone={1}>
                {comparable && r.fits ? `${fmt.int(r.throughput)} tok/s` : '?'}
              </Label>
              <Label x={468} y={y + 15} size={9.5} tone={3}>
                {comparable && r.fits ? `${r.perUser.toFixed(0)}/user` : ''}
              </Label>
              <Label x={530} y={y + 15} size={9.5} tone={3}>
                {comparable && r.fits ? `${r.ttftMs.toFixed(0)} ms` : ''}
              </Label>
              {comparable && (
                <Chip x={572} y={y + 1} w={62} h={20} size={9}
                  kind={kind}
                  text={!r.fits ? 'no fit' : r.meets ? 'meets SLO' : 'misses SLO'} />
              )}
            </g>
          );
        })}

        <Label x={390} y={262} tone={3} size={8.5} anchor="start">total tok/s</Label>
        <Label x={468} y={262} tone={3} size={8.5} anchor="start">per user</Label>
        <Label x={530} y={262} tone={3} size={8.5} anchor="start">TTFT</Label>

        {/* ── the one-line verdict ─────────────────────────────── */}
        <line x1={40} y1={276} x2={614} y2={276} className="v-rule faint" />
        <Fade show={s.step >= 6}>
          <Span x={40} y={288} w={200} h={22} kind="token" label="act on this ranking" labelTone="dark" />
          <Label x={252} y={303} size={10.5} mono={false} bold>
            {(() => {
              const ok = rows.filter((r) => r.fits && r.meets);
              if (!ok.length) return 'nothing meets the SLO at this load — back off ρ';
              const best = ok.reduce((a, b) => (b.throughput > a.throughput ? b : a));
              const c = rows.filter((r) => r.fits && r.meets).length;
              return `${c} of 4 qualify; ${best.engine.name} leads at ${fmt.int(best.throughput)} tok/s`;
            })()}
          </Label>
          <Label x={40} y={324} tone={3} size={9.5} mono={false}>
            {profile.blurb}
          </Label>
        </Fade>
        <Fade show={s.step < 6}>
          <Label x={40} y={300} tone={3} size={10} mono={false}>
            {6 - s.step} item{6 - s.step === 1 ? '' : 's'} of the contract still free. Change the profile and the bars move — but you do not yet know whether that is the engine or the experiment.
          </Label>
        </Fade>
      </svg>
    </Stage>
  );
}