import { useRef, useState } from 'react';
import { Stage, Readout, Segmented, Slider } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { useTween } from '../lib/tween';
import { Label, Span, Fade, fmt, COLOR } from '../lib/svg';
import { LINKS, type Link } from './parallel';

const STEPS = [
  'Two GPUs talking is not one thing',
  'Bandwidth first',
  'Latency is not zero',
  'The tiers, in order',
  'PCIe: 50x slower than NVLink',
  'NVLink: the island',
  'InfiniBand: the only way out',
  'The ring: what a collective actually costs',
];

const CAPTIONS = [
  'Two GPUs on one board are joined by one link. Two GPUs in different buildings are joined by a network card and a switch. "Multi-GPU" covers both, and the gap between them is about 70x.',
  'Bandwidth is bytes per second. A tensor-parallel all-reduce for a 4,096-wide hidden state is 8 KB at batch 1 — tiny, and it moves 128 times per forward pass on a 32-layer model. The rate is what decides whether that is free or ruinous.',
  'Every message pays a fixed cost before a single byte moves: about 1 µs inside a node with NVLink, about 2–3 µs across PCIe, and microseconds again when the message has to leave the machine.',
  'Six links worth knowing, from a PCIe 4.0 x16 slot at about 32 GB/s to NVLink 5 on a B200 at about 900 GB/s. The jump from the slowest to the fastest is 28x.',
  'A PCIe 4.0 x16 link moves 32 GB/s, and an H100 moves memory at 3,350 GB/s. A collective that reads as a pure function of bandwidth on a big payload is 100x slower on PCIe than the memory it is replacing.',
  'Inside an 8-GPU SXM node, NVLink gives every GPU 450–900 GB/s to every other. That is why tensor parallelism goes here and not one step further: the collective cost stops being the story.',
  'InfiniBand at about 50 GB/s per port is a third of NVLink, and an order of magnitude below a B200 link. It is the reason cross-node parallelism is coarse: pipelines and expert parallelism move whole layers, not vectors.',
  'A ring all-reduce is two passes around a ring of P GPUs. Each GPU sends 2(P-1)/P of the payload. The factor is 1.5 at P=4 and 1.75 at P=8 — communication grows with P, so scaling is not free.',
];

const ROW_Y = 44;
const ROW_H = 26;
const ROW_GAP = 8;
const X = 116;
const W = 404;

export default function InterconnectLadder() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [payloadKb, setPayloadKb] = useState(8);
  const [pick, setPick] = useState('nvlink4');

  const bytes = payloadKb * 1024;
  const link: Link = LINKS.find((l) => l.id === pick)!;
  const t = link.latency + bytes / (link.bw * link.eff);
  const tw = useTween(t, 450);

  // PCIe 4.0 x16 is the floor of the ladder, NVLink 4 the ceiling.
  const lg = (bw: number) => (Math.log10(bw / 1e9) - Math.log10(32)) / (Math.log10(900) - Math.log10(32));

  const pcie = LINKS[0];
  const nv = LINKS[3];
  const ratio = nv.bw / pcie.bw;

  // which group of rows each step is talking about
  const lit = (id: string) => {
    if (s.step < 4) return true;
    if (id.startsWith('pcie')) return s.step === 4;
    if (id === 'ib') return s.step === 6;
    return s.step === 5;
  };

  return (
    <Stage
      rootRef={ref}
      title="The interconnect ladder: bandwidth, latency, and what a collective costs"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[
        ['comm', 'the link'],
        ['compute', 'the GPU'],
        ['error', 'PCIe'],
        'text',
      ]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Message', value: `${fmt.int(payloadKb)} KB`, kind: 'text', note: 'one all-reduce payload' },
          { label: 'Link', value: link.name, kind: 'comm', note: `${(link.bw / 1e9).toFixed(0)} GB/s, ~${(link.latency * 1e6).toFixed(1)} µs` },
          { label: 'Transfer', value: `${(tw * 1e6).toFixed(1)} µs`, kind: 'comm', note: `${Math.round((1 - link.eff) * 100)}% below peak` },
          { label: 'vs PCIe 4.0', value: `${(pcie.bw / link.bw).toFixed(1)}x slower`, kind: 'comm', note: 'same bytes' },
        ]} />
      }
      controls={
        <>
          <Slider label="Collective payload" kind="comm" value={payloadKb} min={1} max={1024} log onChange={setPayloadKb} format={(v) => `${fmt.int(v)} KB`} />
          <Segmented
            label="Move it over"
            value={pick}
            options={LINKS.map((l) => ({ value: l.id, label: l.id === 'pcie4' ? 'PCIe 4' : l.id === 'pcie5' ? 'PCIe 5' : l.id === 'nvlink3' ? 'NV3' : l.id === 'nvlink4' ? 'NV4' : l.id === 'nvlink5' ? 'NV5' : 'IB' }))}
            onChange={setPick}
          />
        </>
      }
    >
      <svg viewBox="0 0 640 372" role="img" aria-label="A ladder of six interconnects ordered by bandwidth, from a PCIe 4.0 slot at 32 gigabytes per second up to NVLink 5 at 900 gigabytes per second. Below the ladder, a bar shows the time a chosen link takes to move a chosen payload, with latency dominating for small messages and bandwidth for large ones.">
        {/* ── the ladder ── */}
        <Label x={0} y={26} tone={2} size={11} bold>The ladder, slowest to fastest</Label>
        <Label x={604} y={26} anchor="end" tone={3} size={9.5}>{`${ratio.toFixed(0)}x from bottom to top`}</Label>

        {LINKS.map((l, i) => {
          const y = ROW_Y + i * (ROW_H + ROW_GAP);
          const isPick = l.id === pick;
          const dim = !isPick || !lit(l.id);
          const colour = l.id.startsWith('pcie') ? COLOR.error : l.id === 'ib' ? COLOR.comm : COLOR.token;
          const frac = 0.04 + lg(l.bw) * 0.96;
          return (
            <g key={l.id} className="v-fade" style={{ opacity: dim ? 0.3 : 1, transitionDuration: '500ms' }}>
              <rect x={X} y={y} width={W} height={ROW_H} rx={2} fill="none" stroke="var(--rule)" />
              <rect x={X} y={y} width={W * frac} height={ROW_H} rx={2} fill={colour} opacity={isPick ? 0.9 : 0.55} stroke={colour} />
              <Label x={X + 8} y={y + 17} size={10} style={{ fill: frac > 0.5 ? '#0e1116' : 'var(--text)' }}>{l.name}</Label>
              <Label x={X + W + 8} y={y + 17} size={10} tone={2}>{`${(l.bw / 1e9).toFixed(0)} GB/s`}</Label>
            </g>
          );
        })}

        {/* ── latency vs bandwidth ── */}
        <Fade show={s.step >= 2}>
          <Label x={0} y={278} tone={2} size={11} bold>What one message costs</Label>
          <Label x={0} y={294} tone={3} size={9}>{`${(link.latency * 1e6).toFixed(1)} µs before the first byte, then ${fmt.bytes(bytes)} at ${(link.bw * link.eff / 1e9).toFixed(0)} GB/s`}</Label>
          <Span x={0} y={302} w={Math.max(2, Math.min(W + 108, 4 + (t / 6e-5) * 250))} h={14} kind="comm" opacity={0.85} />
          <Label x={Math.min(W + 108, 4 + (t / 6e-5) * 250) + 8} y={313} size={10} kind="comm">{`${(t * 1e6).toFixed(1)} µs`}</Label>
          <Label x={0} y={332} size={9.5} tone={3}>
            {bytes / (link.bw * link.eff) < link.latency
              ? 'Latency dominates: the bytes would fit in the time it takes to open the channel.'
              : 'Bandwidth dominates: the channel is open and the bytes take as long as they take.'}
          </Label>
        </Fade>

        {/* ── the ring ── */}
        <Fade show={s.step >= 7}>
          <Label x={0} y={356} tone={2} size={11} bold>Ring all-reduce, P = 4</Label>
          <Label x={112} y={356} size={10} kind="comm">{`each GPU sends 2·(4-1)/4 = 1.50x the payload`}</Label>
        </Fade>
      </svg>
    </Stage>
  );
}