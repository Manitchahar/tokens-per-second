import { useEffect, useRef, useState } from 'react';
import { Stage, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Arrow, Fade } from '../lib/svg';

type Layer = 'kserve' | 'ray' | 'lgard' | 'k8s';

const STEPS = [
  'The five layers',
  'Kubernetes: the substrate',
  'A pod is not a replica',
  'KServe: inference as CRD',
  'Ray Serve: requests as objects',
  'llm-d: the scheduler aware one',
  'Choosing a layer',
];
const CAPTIONS = [
  'Every deployment of an LLM is a stack of five layers. The bottom three are shared infrastructure. The top two are where LLM inference differs from serving an ordinary web service, and that is where your engineering effort belongs.',
  'Kubernetes schedules pods, restarts crashed containers, discovers each other through a Service, and hands out DNS names. It knows nothing about tokens, GPUs or KV cache — it is a very good substrate that has never heard of a tensor.',
  'The trap: one pod looks like one server, but a GPU inference replica wants a whole node to itself, pinned devices, huge shared memory, and a shutdown grace period long enough to drain requests. Scheduling that needs node selectors, taints, and init containers.',
  'KServe turns a model into a Kubernetes resource. You declare a model and a runtime, and it reconciles to a deployment, a Service, an autoscaler and an ingress. The win is that everything you already know about Kubernetes applies.',
  'Ray Serve is built around the request itself. You write a Python @app.deployment class, and Ray handles placement, batching, replica scheduling and scale-to-zero. More control than KServe, more code to own.',
  'llm-d layers a scheduler-aware control plane over Kubernetes. Its premise is that inference has a scheduling problem GPUs do not natively understand — locality, cache state, prompt and decode balance — and that solving it should be a separate, pluggable layer.',
  'Pick by how much of the problem you want to own. KServe if you want Kubernetes to own it, Ray Serve if you want to program the serving logic, llm-d if the scheduling itself is the differentiator you are betting on.',
];

const PRESET: Record<number, Layer> = {
  0: 'kserve', 1: 'k8s', 2: 'k8s', 3: 'kserve', 4: 'ray', 5: 'lgard', 6: 'kserve',
};

const LAYERS: Record<Layer, { name: string; one: string; detail: string; accent: 'comm' | 'compute' | 'memory' | 'token' | 'text' }> = {
  k8s: {
    name: 'Kubernetes',
    one: 'Schedules pods, restarts them, wires a Service',
    detail: 'Nodes, pods, Deployments, Services, Ingress. You get scheduling, health, restarts, DNS and rolling updates — none of which know what a token is.',
    accent: 'text',
  },
  kserve: {
    name: 'KServe / InferenceService',
    one: 'A model declared as a Kubernetes resource',
    detail: 'Custom resource + controller. Declaring a model reconciles to a deployment, autoscaler, metrics adapter and ingress. Everything you know about k8s keeps working.',
    accent: 'comm',
  },
  ray: {
    name: 'Ray Serve',
    one: 'Requests as objects, deployment as a class',
    detail: 'You write @app.deployment. Ray owns placement, batching, replica routing and scale-to-zero. Maximum control; maximum code for you to maintain.',
    accent: 'compute',
  },
  lgard: {
    name: 'llm-d',
    one: 'A scheduler-aware control plane',
    detail: 'An inference gateway and a topology/placement layer that route on queue, cache locality and prefill/decode balance. Optimises placement, which k8s alone cannot.',
    accent: 'memory',
  },
};

export default function Orchestrator() {
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  const [layer, setLayer] = useState<Layer>('kserve');
  useEffect(() => setLayer(PRESET[s.step]), [s.step]);
  const L = LAYERS[layer];

  const rows = [
    { name: 'Your application', note: 'clients, streaming, auth', kind: 'text' as const },
    { name: layer === 'ray' ? 'Ray Serve router' : layer === 'lgard' ? 'llm-d inference gateway' : 'Ingress + KServe router', note: 'which replica?', kind: 'comm' as const },
    { name: 'Autoscaler', note: 'how many replicas?', kind: 'compute' as const },
    { name: 'Replicas (pods)', note: 'vLLM / SGLang engine, one per GPU', kind: 'memory' as const },
    { name: 'GPU', note: 'the actual work', kind: 'token' as const },
  ];
  const rowH = 52;
  const y0 = 46;

  return (
    <Stage
      rootRef={ref}
      title="The orchestration stack, top to bottom"
      steps={STEPS}
      captions={CAPTIONS}
      legend={['text', ['comm', 'routing'], ['compute', 'scaling'], ['memory', 'the engine'], ['token', 'the GPU']]}
      {...s}
      readout={
        <Readout items={[
          { label: 'Top layer', value: L.name, kind: L.accent === 'text' ? 'text' : L.accent },
          { label: 'Ownership', value: layer === 'k8s' ? 'you own the rest' : layer === 'kserve' ? 'shared' : layer === 'ray' ? 'you own the code' : 'shared + optional', note: L.one },
        ]} />
      }
      controls={
        <Segmented
          label="Focus layer"
          value={layer}
          options={[
            { value: 'k8s', label: 'Kubernetes' },
            { value: 'kserve', label: 'KServe' },
            { value: 'ray', label: 'Ray Serve' },
            { value: 'lgard', label: 'llm-d' },
          ]}
          onChange={setLayer}
        />
      }
    >
      <svg viewBox="0 0 640 380" role="img" aria-label="A layered diagram of an LLM serving stack. From the top: your application, a routing layer, an autoscaler, the inference replicas that run the engine, and the GPUs underneath. The highlighted layer changes to show what each orchestration technology adds.">
        <Label x={0} y={22} tone={2} size={11} bold>Everything between your code and the GPU</Label>

        {/* stack rows */}
        {rows.map((r, i) => {
          const y = y0 + i * rowH;
          const focused = (layer === 'ray' && i === 1) || (layer === 'lgard' && i === 1) || (layer === 'kserve' && i === 1) || (layer === 'k8s' && i >= 3);
          return (
            <g key={r.name}>
              <rect x={0} y={y} width={470} height={rowH - 10} rx={4}
                fill={focused ? 'var(--ground-3)' : 'transparent'}
                stroke={focused ? 'var(--rule-2)' : 'var(--rule)'} strokeWidth={1} />
              <rect x={0} y={y} width={5} height={rowH - 10} rx={2}
                fill={r.kind === 'text' ? 'var(--text-3)' : `var(--${r.kind})`} />
              <text x={20} y={y + 21} className="v-label" style={{ fontSize: 11.5, fill: 'var(--text)', fontWeight: focused ? 700 : 500 }}>{r.name}</text>
              <text x={20} y={y + 35} className="v-label t3" style={{ fontSize: 9.5 }}>{r.note}</text>
              {i < rows.length - 1 && <Arrow x1={235} y1={y + rowH - 10} x2={235} y2={y + rowH + 2} muted />}
            </g>
          );
        })}

        {/* the detail panel for the focused layer */}
        <g>
          <rect x={490} y={y0} width={150} height={rows.length * rowH - 10} rx={4} fill="var(--ground-2)" stroke="var(--rule-2)" />
          <Label x={502} y={y0 + 20} size={10} bold tone={1}>{L.name}</Label>
          <text x={502} y={y0 + 36} className="v-label" style={{ fontSize: 9, fill: 'var(--text-2)' }}>
            {wrap(L.detail, 24).map((ln, i) => (
              <tspan key={i} x={502} dy={i === 0 ? 0 : 11}>{ln}</tspan>
            ))}
          </text>
        </g>

        {/* the GPU reality underneath */}
        <Fade show={s.step >= 2}>
          <g>
            <rect x={490} y={y0 + 3 * rowH + 6} width={150} height={rowH - 10} rx={4} fill="url(#tps-hatch)" opacity={0.35} />
            <Label x={502} y={y0 + 3 * rowH + 24} size={10} bold kind="token">One GPU per pod</Label>
            <text x={502} y={y0 + 3 * rowH + 38} className="v-label t3" style={{ fontSize: 9 }}>exclusive, pinned</text>
          </g>
        </Fade>

        {/* llm-d's special claim */}
        <Fade show={s.step >= 5}>
          <g>
            <rect x={0} y={316} width={640} height={56} rx={4} fill="var(--ground-3)" opacity={0.5} />
            <Label x={12} y={332} size={10.5} bold tone={2}>{NOTE[s.step].title}</Label>
            <text x={12} y={348} className="v-label sans" style={{ fontSize: 10.5, fill: 'var(--text-2)' }}>{NOTE[s.step].body}</text>
          </g>
        </Fade>
      </svg>
    </Stage>
  );
}

const NOTE: { title: string; body: string }[] = [
  { title: '', body: '' },
  { title: 'Kubernetes is a substrate, not a solution', body: 'It gives you scheduling, restarts, DNS, rolling updates and HPA. It gives you nothing about tokens or KV cache.' },
  { title: 'A pod is not a replica', body: 'An inference replica wants a whole node, pinned devices, large shared memory, and time to drain on shutdown.' },
  { title: 'KServe: declare, do not build', body: 'You write a model resource; the controller reconciles the rest. Least code, least control.' },
  { title: 'Ray Serve: program the serving logic', body: 'Requests are first-class objects. You own batching, routing and scale-to-zero in Python.' },
  { title: 'llm-d: scheduling as its own layer', body: 'Locality, cache state and prefill/decode balance are placement problems k8s cannot express on its own.' },
  { title: 'Choose by how much you want to own', body: 'More control means more to build, watch, and page someone about at 3am.' },
];

function wrap(text: string, width: number): string[] {
  const words = text.split(' ');
  const lines: string[] = [];
  let line = '';
  for (const w of words) {
    if (line && line.length + 1 + w.length > width) { lines.push(line); line = w; }
    else line = line ? `${line} ${w}` : w;
  }
  if (line) lines.push(line);
  return lines;
}