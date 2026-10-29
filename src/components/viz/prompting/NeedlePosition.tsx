import { useEffect, useRef, useState, type PointerEvent } from 'react';
import { Stage, Slider, Segmented, Readout } from '../lib/Stage';
import { useStep } from '../lib/useStep';
import { Label, Fade, fmt } from '../lib/svg';
import {
  acc, promptTokens, expectedAcc, CLOSED_BOOK, ORACLE, ORDERINGS, NIAH_LENGTHS, NIAH_DEPTHS, niah,
  type K, type OrderId,
} from './position';

const STEPS = [
  'A long prompt with the answer in it',
  'Drag the answer into the middle',
  'Worse than no documents at all',
  'More documents, deeper valley',
  'Needle in a haystack',
  'The window you are sold and the window you get',
  'Put it where the model looks',
];
const CAPTIONS = [
  'Twenty retrieved documents, one of which holds the answer, then the question. With the right document first, the model answers correctly about 76% of the time (illustrative numbers).',
  'Drag the relevant document along the prompt. Accuracy falls as it moves toward the middle and partly recovers at the end: a U-shaped curve. Same tokens, same model, different answer.',
  'The dashed line is the model answering with no documents at all. In the middle of the prompt, the right document makes the model worse than having nothing.',
  'Ten, twenty or thirty documents: the valley deepens as the prompt grows, and every extra document costs 130 tokens of prefill.',
  'Needle-in-a-haystack tests hide one fact at every depth of contexts of every length and ask for it back. Short contexts are solid; long ones fail in the middle (toy heatmap).',
  'A model can advertise 128K tokens and use far fewer well. Benchmarks harder than a single needle find an effective length well below the advertised one.',
  'You control the order. Random order averages the whole U; ranking the most relevant document first puts it where the model looks. Same documents, about 12 points better here.',
];

const DEF_K: K[] = [20, 20, 20, 20, 20, 20, 20];
const DEF_POS = [1, 10, 10, 10, 1, 1, 1];
const DOC_X0 = 52;
const DOC_X1 = 560;
const BAR_Y = 30;
const BAR_H = 26;
const CY0 = 92; // accuracy 0.9
const CY1 = 262; // accuracy 0.4
const yA = (a: number) => CY1 - ((a - 0.4) / 0.5) * (CY1 - CY0);

export default function NeedlePosition() {
  const ref = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const s = useStep(STEPS.length, ref);
  const st = s.step;
  const [k, setK] = useState<K>(20);
  const [pos, setPos] = useState(1);
  const [order, setOrder] = useState<OrderId>('ranked');
  const [drag, setDrag] = useState(false);

  useEffect(() => { setK(DEF_K[st]); setPos(DEF_POS[st]); setOrder('ranked'); }, [st]);

  const slotW = (DOC_X1 - DOC_X0) / k;
  const cx = (p: number) => DOC_X0 + (p - 0.5) * slotW;
  const p = Math.min(pos, k);
  const a = acc(p, k);

  const posFromPointer = (e: PointerEvent<SVGElement>) => {
    const svg = svgRef.current;
    if (!svg) return;
    const r = svg.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * 640;
    setPos(Math.max(1, Math.min(k, Math.floor((x - DOC_X0) / slotW) + 1)));
  };

  const showCurve = st >= 1 && st <= 3;
  const curvePts = Array.from({ length: k }, (_, i) => `${cx(i + 1).toFixed(1)},${yA(acc(i + 1, k)).toFixed(1)}`).join(' ');
  const exp = expectedAcc(order, k);

  const notes: [string, string][] = [
    [`Relevant document at position ${p} of ${k}: accuracy ${(a * 100).toFixed(0)}%.`, `${k} documents × 130 tokens + instruction + question = ${fmt.int(promptTokens(k))} tokens. Far below any modern window.`],
    [`Position ${p} of ${k}: ${(a * 100).toFixed(0)}%. First: ${(acc(1, k) * 100).toFixed(0)}%. Last: ${(acc(k, k) * 100).toFixed(0)}%. Worst: ${(Math.min(...Array.from({ length: k }, (_, i) => acc(i + 1, k))) * 100).toFixed(0)}%.`, 'Primacy beats recency: the start of the prompt is used best, the end next, the middle worst.'],
    [`No documents: ${(CLOSED_BOOK * 100).toFixed(0)}%. Only the right document: ${(ORACLE * 100).toFixed(0)}%. Middle of 20: ${(acc(10, 20) * 100).toFixed(0)}%.`, 'Liu et al. (2023) saw this with GPT-3.5-Turbo: the middle fell below closed-book accuracy.'],
    [`${k} documents: ${fmt.int(promptTokens(k))} tokens, worst position ${(Math.min(...Array.from({ length: k }, (_, i) => acc(i + 1, k))) * 100).toFixed(0)}%, average over positions ${(expectedAcc('random', k) * 100).toFixed(0)}%.`, 'More context is more chances to bury the answer, more prefill, and more cost.'],
    ['Green: found. Red: missed. Short contexts are solid; at 64K and 128K the middle depths fail.', 'A single literal fact is the easiest possible test. Passing it is necessary, not sufficient.'],
    ['Advertised: what the position embeddings and training allow. Effective: where accuracy stays acceptable.', 'Hsieh et al. (2024, RULER): about half the models claiming 32K or more held up at 32K.'],
    [`${ORDERINGS[order].label}: ${(exp * 100).toFixed(1)}% expected. Random order: ${(expectedAcc('random', k) * 100).toFixed(1)}%.`, 'The retriever ranked the answer first half the time. Ordering costs nothing and uses that ranking.'],
  ];
  const [l1, l2] = notes[st];

  return (
    <Stage
      rootRef={ref}
      title="Where in the prompt the answer sits"
      steps={STEPS}
      captions={CAPTIONS}
      legend={[['memory', 'context documents'], ['token', 'the relevant document'], ['text', 'accuracy (illustrative)'], ['error', 'below no-documents']]}
      {...s}
      readout={
        <Readout items={st === 6 ? [
          { label: 'ordering', value: ORDERINGS[order].label, kind: 'text', note: 'same 20 documents, same tokens' },
          { label: 'expected accuracy', value: `${(exp * 100).toFixed(1)}%`, kind: 'text', note: `random order ${(expectedAcc('random', k) * 100).toFixed(1)}%` },
          { label: 'prompt tokens', value: fmt.int(promptTokens(k)), kind: 'memory', note: 'unchanged: placement is free' },
        ] : st === 4 || st === 5 ? [
          { label: 'advertised window', value: '128K', kind: 'memory', note: 'what the model card says' },
          { label: 'effective length', value: '≈ 32K', kind: 'text', note: 'toy example: where harder tasks still hold' },
        ] : [
          { label: 'relevant document at', value: `${p} of ${k}`, kind: 'token', note: p === 1 ? 'the very start' : p === k ? 'right before the question' : 'in the middle' },
          { label: 'accuracy', value: `${((st === 6 ? exp : a) * 100).toFixed(0)}%`, kind: (st === 6 ? exp : a) < CLOSED_BOOK ? 'error' : 'text', note: st === 6 ? `expected, ${ORDERINGS[order].label}` : `closed-book ${(CLOSED_BOOK * 100).toFixed(0)}%` },
          { label: 'prompt tokens', value: fmt.int(promptTokens(k)), kind: 'memory', note: `${k} docs × 130 + 100` },
        ]} />
      }
      controls={
        <>
          {st <= 3 && <Slider label="Relevant document (drag it)" value={p} min={1} max={k} onChange={setPos} format={(v) => `${v} of ${k}`} kind="token" />}
          {st === 3 && <Segmented label="Documents in the prompt" value={k} options={[{ value: 10 as K, label: '10' }, { value: 20 as K, label: '20' }, { value: 30 as K, label: '30' }]} onChange={setK} />}
          {st === 6 && <Segmented label="Order the retrieved documents" value={order} options={(Object.keys(ORDERINGS) as OrderId[]).map((o) => ({ value: o, label: ORDERINGS[o].label }))} onChange={setOrder} />}
        </>
      }
    >
      <svg ref={svgRef} viewBox="0 0 640 380" role="img" aria-label="A prompt made of retrieved documents with one relevant document that can be dragged to any position; below it, a U-shaped curve of answer accuracy against that position, with lines for no documents and for only the relevant document.">
        {/* ── the prompt ── */}
        <Label x={16} y={18} tone={2} size={10} bold>the prompt</Label>
        <Label x={624} y={18} anchor="end" tone={3} size={9}>{`instruction · ${k} documents · question`}</Label>
        <rect x={16} y={BAR_Y} width={32} height={BAR_H} rx={2} fill="var(--memory-dim)" stroke="var(--memory)" strokeWidth={0.8} />
        <text x={32} y={BAR_Y + 17} textAnchor="middle" className="v-label t1" style={{ fontSize: 8 }}>instr</text>
        <g style={{ cursor: st <= 3 ? 'ew-resize' : 'default', touchAction: 'none' }}
          onPointerDown={(e) => { if (st > 3) return; (e.target as Element).setPointerCapture?.(e.pointerId); setDrag(true); posFromPointer(e); }}
          onPointerMove={(e) => { if (drag) posFromPointer(e); }}
          onPointerUp={() => setDrag(false)}
          onPointerCancel={() => setDrag(false)}>
          <rect x={DOC_X0} y={BAR_Y - 8} width={DOC_X1 - DOC_X0} height={BAR_H + 16} fill="transparent" />
          {Array.from({ length: k }, (_, i) => {
            const rel = st <= 3 && i + 1 === p;
            return (
              <rect key={`${k}-${i}`} x={DOC_X0 + i * slotW + 0.8} y={BAR_Y} width={slotW - 1.6} height={BAR_H} rx={1.5}
                fill={rel ? 'var(--token)' : 'var(--memory-dim)'} stroke={rel ? 'var(--token)' : 'var(--memory)'} strokeWidth={0.7} className="v-cell" />
            );
          })}
          {st <= 3 && (
            <g style={{ transform: `translate(${cx(p)}px, 0)`, transition: drag ? 'none' : 'transform .45s var(--ease-out)' }}>
              <path d={`M-5,${BAR_Y - 9} L5,${BAR_Y - 9} L0,${BAR_Y - 2} Z`} fill="var(--token)" />
            </g>
          )}
        </g>
        <rect x={564} y={BAR_Y} width={60} height={BAR_H} rx={2} fill="var(--memory)" stroke="var(--memory)" strokeWidth={0.8} />
        <text x={594} y={BAR_Y + 17} textAnchor="middle" className="v-label" style={{ fontSize: 8.5, fontWeight: 700, fill: '#0e1116' }}>question</text>
        <text x={DOC_X0} y={BAR_Y + BAR_H + 14} className="v-label t3" style={{ fontSize: 8 }}>start</text>
        <text x={DOC_X1} y={BAR_Y + BAR_H + 14} textAnchor="end" className="v-label t3" style={{ fontSize: 8 }}>end, next to the question</text>

        {/* ── the U-curve ── */}
        <Fade show={showCurve || st === 0}>
          {[0.4, 0.5, 0.6, 0.7, 0.8, 0.9].map((v) => (
            <g key={v}>
              <line x1={DOC_X0} x2={DOC_X1} y1={yA(v)} y2={yA(v)} className="v-rule faint" />
              <text x={DOC_X0 - 6} y={yA(v) + 3} textAnchor="end" className="v-label t3" style={{ fontSize: 8 }}>{`${(v * 100).toFixed(0)}%`}</text>
            </g>
          ))}
          <text x={DOC_X0 - 6} y={CY0 - 10} textAnchor="end" className="v-label t3" style={{ fontSize: 8 }}>accuracy</text>
          <line x1={DOC_X0} x2={DOC_X1} y1={yA(ORACLE)} y2={yA(ORACLE)} stroke="var(--text-3)" strokeDasharray="5 3" />
          <text x={DOC_X1} y={yA(ORACLE) - 5} textAnchor="end" className="v-label t3" style={{ fontSize: 8 }}>{`only the right document: ${(ORACLE * 100).toFixed(0)}%`}</text>
          <g className="v-fade" style={{ opacity: st >= 2 ? 1 : 0 }}>
            <line x1={DOC_X0} x2={DOC_X1} y1={yA(CLOSED_BOOK)} y2={yA(CLOSED_BOOK)} stroke="var(--error)" strokeDasharray="5 3" />
            <text x={DOC_X1 + 6} y={yA(CLOSED_BOOK) + 3} className="v-label" style={{ fontSize: 8, fill: 'var(--error)' }}>{`no docs ${(CLOSED_BOOK * 100).toFixed(0)}%`}</text>
            {Array.from({ length: k }, (_, i) => acc(i + 1, k) < CLOSED_BOOK ? (
              <rect key={`b-${k}-${i}`} x={cx(i + 1) - slotW / 2 + 1} y={yA(CLOSED_BOOK)} width={slotW - 2} height={yA(acc(i + 1, k)) - yA(CLOSED_BOOK)} fill="var(--error-dim)" />
            ) : null)}
          </g>
          <polyline points={curvePts} fill="none" stroke="var(--text)" strokeWidth={1.8} className="v-fade" style={{ opacity: showCurve ? 1 : 0 }} />
          <line x1={cx(p)} x2={cx(p)} y1={BAR_Y + BAR_H + 2} y2={CY1} stroke="var(--token)" strokeDasharray="2 3" style={{ transition: drag ? 'none' : 'all .45s var(--ease-out)' }} />
          <circle cx={cx(p)} cy={yA(a)} r={5} fill={a < CLOSED_BOOK && st >= 2 ? 'var(--error)' : 'var(--text)'} stroke="var(--ground-2)" strokeWidth={1.5} style={{ transition: drag ? 'none' : 'cx .45s var(--ease-out), cy .45s var(--ease-out)' }} />
          <text x={cx(p) + 8} y={yA(a) - 8} className="v-label t1" style={{ fontSize: 9, fontWeight: 700 }}>{`${(a * 100).toFixed(0)}%`}</text>
          {[1, Math.round(k / 2), k].map((q) => (
            <text key={q} x={cx(q)} y={CY1 + 14} textAnchor="middle" className="v-label t3" style={{ fontSize: 8 }}>{q}</text>
          ))}
          <text x={(DOC_X0 + DOC_X1) / 2} y={CY1 + 28} textAnchor="middle" className="v-label t3" style={{ fontSize: 8.5 }}>position of the relevant document</text>
        </Fade>

        {/* ── needle in a haystack ── */}
        <Fade show={st === 4}>
          <Label x={16} y={86} tone={2} size={9.5}>needle recall by context length and depth (toy)</Label>
          {NIAH_LENGTHS.map((L, li) => (
            <g key={L}>
              <text x={60} y={110 + li * 26 + 13} textAnchor="end" className="v-label t2" style={{ fontSize: 8.5 }}>{L}</text>
              {NIAH_DEPTHS.map((d, di) => {
                const r = niah(li, d);
                return <rect key={d} x={68 + di * 44} y={110 + li * 26} width={41} height={22} rx={2} fill={r > 0.97 ? 'var(--token)' : 'var(--error)'} opacity={r > 0.97 ? 0.85 : 0.35 + 0.65 * (1 - r)} />;
              })}
            </g>
          ))}
          {NIAH_DEPTHS.map((d, di) => (
            <text key={d} x={68 + di * 44 + 20} y={276} textAnchor="middle" className="v-label t3" style={{ fontSize: 8 }}>{`${d}%`}</text>
          ))}
          <text x={68} y={292} className="v-label t3" style={{ fontSize: 8.5 }}>needle depth in the context →</text>
        </Fade>

        {/* ── advertised vs effective ── */}
        <Fade show={st === 5}>
          <Label x={16} y={92} tone={2} size={9.5}>context length a model is sold with, and the length it uses well (illustrative)</Label>
          {[
            { label: 'advertised window', v: 128, kind: 'memory' },
            { label: 'single-needle test passes', v: 128, kind: 'token' },
            { label: 'harder tasks hold up to', v: 32, kind: 'text' },
          ].map((b, i) => (
            <g key={b.label}>
              <text x={170} y={128 + i * 40} textAnchor="end" className="v-label t2" style={{ fontSize: 9 }}>{b.label}</text>
              <rect x={178} y={116 + i * 40} width={(b.v / 128) * 400} height={18} rx={2} fill={b.kind === 'memory' ? 'var(--memory)' : b.kind === 'token' ? 'var(--token)' : 'var(--text-2)'} />
              <text x={178 + (b.v / 128) * 400 + 6} y={129 + i * 40} className="v-label t1" style={{ fontSize: 9 }}>{`${b.v}K`}</text>
            </g>
          ))}
          <rect x={178 + (32 / 128) * 400} y={196} width={(96 / 128) * 400} height={18} rx={2} fill="var(--error-dim)" stroke="var(--error)" strokeDasharray="3 3" />
          <text x={178 + (80 / 128) * 400} y={209} textAnchor="middle" className="v-label" style={{ fontSize: 8.5, fill: 'var(--error)' }}>paid for in memory and prefill, used poorly</text>
          <text x={178} y={252} className="v-label t3" style={{ fontSize: 8.5 }}>Measure the length your task actually works at. Do not read it off the model card.</text>
        </Fade>

        {/* ── ordering strategies ── */}
        <Fade show={st === 6}>
          <Label x={16} y={92} tone={2} size={9.5}>expected accuracy over many questions, 20 documents (illustrative)</Label>
          {(Object.keys(ORDERINGS) as OrderId[]).map((o, i) => {
            const v = expectedAcc(o, 20);
            const on = o === order;
            return (
              <g key={o}>
                <text x={170} y={128 + i * 34} textAnchor="end" className="v-label" style={{ fontSize: 9, fill: on ? 'var(--text)' : 'var(--text-2)' }}>{ORDERINGS[o].label}</text>
                <rect x={178} y={116 + i * 34} width={((v - 0.4) / 0.5) * 380} height={18} rx={2} fill={v < CLOSED_BOOK ? 'var(--error)' : 'var(--text-2)'} stroke={on ? 'var(--text)' : 'none'} strokeWidth={1.4} />
                <text x={178 + ((v - 0.4) / 0.5) * 380 + 6} y={129 + i * 34} className="v-label t1" style={{ fontSize: 9 }}>{`${(v * 100).toFixed(1)}%`}</text>
              </g>
            );
          })}
          <line x1={178 + ((CLOSED_BOOK - 0.4) / 0.5) * 380} x2={178 + ((CLOSED_BOOK - 0.4) / 0.5) * 380} y1={108} y2={222} stroke="var(--error)" strokeDasharray="3 3" />
          <text x={178 + ((CLOSED_BOOK - 0.4) / 0.5) * 380 + 4} y={236} className="v-label" style={{ fontSize: 8, fill: 'var(--error)' }}>no documents</text>
          <text x={178} y={262} className="v-label t3" style={{ fontSize: 8.5 }}>bars start at 40%. Also restate the key instruction after the documents.</text>
        </Fade>

        {/* ── notes ── */}
        <line x1={16} x2={624} y1={308} y2={308} className="v-rule faint" />
        <Label x={16} y={328} tone={1} size={9.5}>{l1}</Label>
        <Label x={16} y={348} tone={2} size={9}>{l2}</Label>
      </svg>
    </Stage>
  );
}
