import { useEffect, useRef, useState } from 'react';
import { prefersReducedMotion } from './useStep';

export const easeOutExpo = (t: number) => (t >= 1 ? 1 : 1 - Math.pow(2, -10 * t));
export const easeInOutCubic = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

/** Smoothly animate a number toward `target`. */
export function useTween(target: number, ms = 700, ease = easeOutExpo): number {
  const [v, setV] = useState(target);
  const from = useRef(target);
  const cur = useRef(target);
  useEffect(() => {
    if (prefersReducedMotion() || ms <= 0) { cur.current = target; setV(target); return; }
    from.current = cur.current;
    const start = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / ms);
      const x = from.current + (target - from.current) * ease(t);
      cur.current = x;
      setV(x);
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);
  return v;
}

/** Animate several numbers together. Keys must stay stable between renders. */
export function useTweens<T extends Record<string, number>>(target: T, ms = 700, ease = easeOutExpo): T {
  const [v, setV] = useState<T>(target);
  const cur = useRef<T>(target);
  const key = JSON.stringify(target);
  useEffect(() => {
    if (prefersReducedMotion() || ms <= 0) { cur.current = target; setV(target); return; }
    const from = { ...cur.current };
    const start = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / ms);
      const e = ease(t);
      const next = {} as T;
      for (const k in target) (next as Record<string, number>)[k] = (from[k] ?? target[k]) + (target[k] - (from[k] ?? target[k])) * e;
      cur.current = next;
      setV(next);
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [key, ms]);
  return v;
}

/** A clock that runs while `running` is true: returns elapsed seconds. Used for continuous loops. */
export function useClock(running: boolean, speed = 1): number {
  const [t, setT] = useState(0);
  const acc = useRef(0);
  useEffect(() => {
    if (!running) return;
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      acc.current += ((now - last) / 1000) * speed;
      last = now;
      setT(acc.current);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [running, speed]);
  return t;
}

/** True once the element has scrolled into view (for starting ambient loops lazily). */
export function useInView<T extends Element>(ref: React.RefObject<T | null>, rootMargin = '0px'): boolean {
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(([e]) => setSeen(e.isIntersecting), { rootMargin });
    io.observe(el);
    return () => io.disconnect();
  }, [ref, rootMargin]);
  return seen;
}
