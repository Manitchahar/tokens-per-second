import { useRef } from 'react';
import { Stage, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Arrow, Fade } from '../lib/svg';

const STEPS = [
  'One line of your code',
  'The framework',
  'The runtime',
  'The driver',
  'The GPU',
  'You do not have the GPU',
  'Errors name the wrong layer',
];
const CAPTIONS = [
  'You write one line. The other four layers of software you have never met will carry it to silicon and bring back an answer.',
  'PyTorch looks at the tensor, its dtype and its device, picks a kernel from its library, and hands the work to the CUDA runtime.',
  'The CUDA runtime allocates device memory and launches kernels onto a stream. Everything after this point is asynchronous, which is a source of bugs worth knowing early.',
  'The driver is the only software permitted to talk to the GPU. It manages memory, sets up work, and hands each kernel to the scheduler.',
  'The GPU runs the kernels on its multiprocessors. It cannot report anything back by itself; it signals, and the driver decodes the signal.',
  'In Colab, on a rented H100, or on a cluster, the GPU is on a different machine. You reach it over a network, through a container, and your local files are not its files.',
  'A CUDA error is reported by the driver about four layers away from the mistake. Learning which layer names an error is a real debugging skill.',
];

const LAYERS = [
  { name: 'Your code', sub: 'model.generate(...)', note: 'PyTorch Python. Shapes, dtypes, devices, and the bugs you actually cause.' },
  { name: 'Framework', sub: 'PyTorch · ATen · TorchInductor', note: 'Picks a kernel, fuses what it can, moves data between host and device.' },
  { name: 'Runtime', sub: 'CUDA runtime', note: 'cudaMalloc, kernel launches, streams and events. Host-side, in your process.' },
  { name: 'Driver', sub: 'NVIDIA driver', note: 'The only layer allowed to touch the GPU. Translates calls into device commands.' },
  { name: 'Hardware', sub: 'GPU · 132 SMs', note: 'Kernels execute here. No error messages originate here; it raises a signal.' },
];

const BAND = { x: 56, w: 372, h: 32, y0: 54, gap: 5 };
const bandY = (i: number) => BAND.y0 + i * (BAND.h + BAND.gap);

const ERRORS = [
  'RuntimeError: mat1 and mat2 shapes cannot be multiplied',
  'RuntimeError: expected scalar type Half but found Float',
  'CUDA out of memory. Tried to allocate 2.00 GiB',
  'CUDA error: an illegal memory access was encountered',
  'nothing — the context is killed and the error is a signal',
];

const HOSTS = [
  { name: 'Your laptop', sub: 'no GPU in this machine', x: 56 },
  { name: 'Container', sub: 'cudnn, torch, vllm, your code', x: 240 },
  { name: 'GPU host', sub: 'driver + devices', x: 424 },
];
const HOST_W = 140;

export default function SoftwareStack() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;

  return (
    <Stage
      rootRef={ref}
      title="One call, five layers, and a network in between"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['compute', ['text', 'host process'], ['comm', 'over the network']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Layer in focus', value: LAYERS[Math.min(st, 4)].name, kind: 'text', note: st >= 5 ? 'and a machine far away' : LAYERS[Math.min(st, 4)].sub },
          { label: 'Where your code runs', value: st >= 5 ? 'their machine' : 'your machine', kind: st >= 5 ? 'comm' : 'text', note: 'process boundary' },
          { label: 'Error arrives at', value: st >= 6 ? 'the driver' : 'wherever raises it', kind: st >= 6 ? 'error' : 'text', note: st >= 6 ? 'four layers from the cause' : 'stacked traceback' },
        ]} />
      }
    >
      <svg viewBox="0 0 640 412" role="img" aria-label="A stack diagram of the software layers between your Python line and the GPU, plus a diagram of a laptop reaching a GPU on another machine over a network, and the error each layer reports.">
        <Label x={56} y={30} tone={1} size={11} bold mono={false}>What happens to one line of code</Label>
        <Label x={448} y={30} tone={2} size={10} mono={false}>what that layer is for</Label>

        {LAYERS.map((l, i) => {
          const on = i === st;
          const past = i < st;
          return (
            <g key={l.name}>
              <rect x={BAND.x} y={bandY(i)} width={BAND.w} height={BAND.h} rx={3}
                fill={on ? 'var(--compute-dim)' : past ? 'var(--ground-3)' : 'none'}
                stroke={on ? 'var(--compute)' : 'var(--rule-2)'} strokeWidth={on ? 1.3 : 1}
                strokeDasharray={past || on ? undefined : '3 3'} className="v-fade" style={{ opacity: past && !on ? 0.45 : 1 }} />
              <text x={BAND.x + 12} y={bandY(i) + 14} className="v-label" style={{ fontSize: 11, fill: on ? 'var(--text)' : 'var(--text-2)', fontWeight: on ? 700 : 400 }}>{l.name}</text>
              <text x={BAND.x + 12} y={bandY(i) + 26} className="v-label t3" style={{ fontSize: 9 }}>{l.sub}</text>
              {i < LAYERS.length - 1 && <Arrow x1={BAND.x + 90} y1={bandY(i) + BAND.h + 1} x2={BAND.x + 90} y2={bandY(i + 1) - 1} kind="compute" width={1.2} opacity={past ? 0.8 : 0.25} />}
            </g>
          );
        })}
        <Fade show={st < 5}>
          <Label x={BAND.x + 100} y={bandY(0) - 8} tone={3} size={9}>↓ your one line, then four layers you did not write</Label>
          <text x={448} y={bandY(Math.min(st, 4)) + 20} className="v-label" style={{ fontSize: 9.5, fill: 'var(--text-2)' }}>{LAYERS[Math.min(st, LAYERS.length - 1)].note}</text>        </Fade>

        {/* the machine, when it is not yours */}
        <Fade show={st === 5}>
          <line x1={56} y1={248} x2={584} y2={248} className="v-rule faint" />
          <Label x={56} y={268} tone={1} size={10.5} bold mono={false}>Where the GPU actually is</Label>
          {HOSTS.map((h, i) => (
            <g key={h.name}>
              <rect x={h.x} y={280} width={HOST_W} height={40} rx={3}
                fill={i === 2 ? 'var(--compute-dim)' : 'var(--ground-3)'}
                stroke={i === 2 ? 'var(--compute)' : 'var(--rule-2)'} strokeWidth={1} className="v-fade" />
              <text x={h.x + HOST_W / 2} y={296} textAnchor="middle" className="v-label" style={{ fontSize: 11, fill: i === 2 ? 'var(--text)' : 'var(--text-2)', fontWeight: 700 }}>{h.name}</text>
              <text x={h.x + HOST_W / 2} y={310} textAnchor="middle" className="v-label t3" style={{ fontSize: 9 }}>{h.sub}</text>
            </g>
          ))}
          {[0, 1].map((i) => (
            <g key={i}>
              <line x1={HOSTS[i].x + HOST_W} x2={HOSTS[i + 1].x - 2} y1={300} y2={300} stroke="var(--comm)" strokeWidth={1.4} strokeDasharray="4 3" className="v-fade" />
              <text x={(HOSTS[i].x + HOST_W + HOSTS[i + 1].x) / 2} y={340} textAnchor="middle" className="v-label" style={{ fontSize: 9, fill: 'var(--comm)' }}>
                {i === 0 ? 'SSH, 20–80 ms' : 'same fabric'}
              </text>
            </g>
          ))}
          <Label x={584} y={364} anchor="end" tone={3} size={9.5}>A 40 ms round trip is 5 decode steps. Shell commands feel slow because they are.</Label>
        </Fade>

        {/* the error column */}
        <Fade show={st >= 6}>
          <line x1={56} y1={248} x2={584} y2={248} className="v-rule faint" />
          <Label x={56} y={268} tone={1} size={10.5} bold mono={false}>The error each layer reports, for the same bug</Label>
          {LAYERS.map((l, i) => {
            const y = 288 + i * 17;
            return (
              <g key={l.name} className="v-fade">
                <text x={56} y={y} className="v-label t3" style={{ fontSize: 9.5 }}>{l.name}</text>
                <text x={148} y={y} className="v-label" style={{ fontSize: 9.5, fill: i >= 2 ? 'var(--error)' : 'var(--text-2)' }}>{ERRORS[i]}</text>
              </g>
            );
          })}
          <Label x={584} y={392} anchor="end" tone={3} size={9.5}>Fix the shape error at the top. You cannot fix the driver message at the bottom.</Label>
        </Fade>
      </svg>
    </Stage>
  );
}
