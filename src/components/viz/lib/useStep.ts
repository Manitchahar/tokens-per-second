import { useCallback, useEffect, useRef, useState, type RefObject } from 'react';

/**
 * Step state for a visual. Inside a <Scrolly> it follows the reader's scroll position;
 * on its own it keeps local state. Either way `go(i)` moves it, and `play` auto-advances.
 */
export function useStep(count: number, ref: RefObject<HTMLElement | null>, opts: { interval?: number; loop?: boolean } = {}) {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const containerRef = useRef<HTMLElement | null>(null);
  const [inScrolly, setInScrolly] = useState(false);
  const interval = opts.interval ?? 2600;

  useEffect(() => {
    const c = ref.current?.closest<HTMLElement>('[data-scrolly]') ?? null;
    containerRef.current = c;
    setInScrolly(!!c);
    if (!c) return;
    const read = () => setStep(Math.max(0, Math.min(count - 1, Number(c.dataset.index || 0))));
    read();
    c.addEventListener('scrolly:change', read);
    return () => c.removeEventListener('scrolly:change', read);
  }, [count, ref]);

  const go = useCallback((i: number) => {
    const n = Math.max(0, Math.min(count - 1, i));
    const c = containerRef.current;
    if (c) c.dispatchEvent(new CustomEvent('scrolly:go', { detail: n }));
    else setStep(n);
  }, [count]);

  // auto-play
  useEffect(() => {
    if (!playing) return;
    const t = window.setTimeout(() => {
      if (step >= count - 1) { if (opts.loop) go(0); else setPlaying(false); }
      else go(step + 1);
    }, interval);
    return () => window.clearTimeout(t);
  }, [playing, step, count, go, interval, opts.loop]);

  return { step, go, playing, setPlaying, inScrolly };
}

export const prefersReducedMotion = () =>
  typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
