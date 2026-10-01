import { useEffect, useRef, useState } from 'react';
import { RequestTrace, END, OUTPUT, TTFT, T, arriveTime } from './RequestTrace';
import { prefersReducedMotion } from '../lib/useStep';
import '../lib/viz.css';

/** The landing page's live trace: one request replaying on a loop, slowed down so you can watch it. */
export default function HeroTrace() {
  // Start identical to the server render (t = 0, paused); browser-only preferences apply after hydration.
  const [t, setT] = useState(0);
  const [running, setRunning] = useState(false);
  useEffect(() => { if (prefersReducedMotion()) setT(END + 20); }, []);
  const box = useRef<HTMLDivElement>(null);
  const SLOW = 7; // 1 ms of trace = 7 ms on screen
  const HOLD = 2200;

  useEffect(() => {
    if (!running) return;
    let raf = 0;
    let start = performance.now() - t * SLOW;
    const tick = (now: number) => {
      let el = (now - start) / SLOW;
      if (el > END + 30 + HOLD / SLOW) { start = now; el = 0; }
      setT(Math.min(el, END + 30));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running]);

  // pause when off screen
  useEffect(() => {
    const el = box.current;
    if (!el || prefersReducedMotion()) return;
    const io = new IntersectionObserver(([e]) => setRunning(e.isIntersecting));
    io.observe(el);
    return () => io.disconnect();
  }, []);

  const shown = OUTPUT.filter((_, i) => t >= arriveTime(i)).length;
  const tps = t > TTFT ? Math.min(1000 / T.step, (shown - 1) / Math.max(0.001, (t - TTFT) / 1000)) : 0;

  return (
    <div className="hero-trace" ref={box}>
      <RequestTrace t={t} showBrackets={t >= END} />
      <div className="hero-readout" aria-live="off">
        <span><b className="num">{t < TTFT ? '—' : `${TTFT} ms`}</b> time to first token</span>
        <span className="tps"><b className="num">{tps ? tps.toFixed(0) : '0'}</b> tokens per second</span>
        <button type="button" className="hero-pause" onClick={() => setRunning(!running)} aria-label={running ? 'Pause the animation' : 'Play the animation'}>
          {running ? 'Pause' : 'Play'}
        </button>
      </div>
      <style>{`
        .hero-trace svg { width: 100%; height: auto; display: block; overflow: visible; }
        .hero-readout { display: flex; flex-wrap: wrap; align-items: baseline; gap: .5rem 2rem; padding: .4rem 0 0 92px; font-size: .8rem; color: var(--text-2); }
        .hero-readout b { font-size: 1.5rem; font-weight: 600; color: var(--text); margin-right: .35rem; font-variant-numeric: tabular-nums; }
        .hero-readout .tps b { color: var(--token); }
        .hero-pause { margin-left: auto; font: inherit; font-size: .75rem; color: var(--text-2); background: none; border: 1px solid var(--rule-2); border-radius: 3px; padding: .25rem .6rem; cursor: pointer; }
        .hero-pause:hover { color: var(--text); border-color: var(--text-2); }
        @media (max-width: 700px) { .hero-readout { padding-left: 0; } }
      `}</style>
    </div>
  );
}
