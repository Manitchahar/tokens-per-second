import { useEffect, useState } from 'react';
import { Icon } from '../viz/lib/Icon';
import { inline, lessonKeyFromPath } from './inline';
import { setQuiz } from '../../scripts/progress';
import './quiz.css';

type Choice = { q: string; options: string[]; answer: number; why: string };
type Numeric = { q: string; numeric: true; answer: number; tolerance?: number; unit?: string; why: string };
export type Question = Choice | Numeric;

function isNumeric(q: Question): q is Numeric {
  return (q as Numeric).numeric === true;
}

export default function Quiz({ questions, title = 'Check yourself' }: { questions: Question[]; title?: string }) {
  const [picked, setPicked] = useState<(number | null)[]>(() => questions.map(() => null));
  const [typed, setTyped] = useState<string[]>(() => questions.map(() => ''));
  const [checked, setChecked] = useState<boolean[]>(() => questions.map(() => false));
  const [key, setKey] = useState('');
  useEffect(() => setKey(lessonKeyFromPath()), []);

  const correct = (i: number) => {
    const q = questions[i];
    if (isNumeric(q)) {
      const v = parseFloat(typed[i].replace(/,/g, ''));
      const tol = q.tolerance ?? Math.abs(q.answer) * 0.05;
      return Number.isFinite(v) && Math.abs(v - q.answer) <= tol;
    }
    return picked[i] === q.answer;
  };
  const answered = checked.filter(Boolean).length;
  const score = questions.reduce((s, _, i) => s + (checked[i] && correct(i) ? 1 : 0), 0);

  useEffect(() => {
    if (key && answered === questions.length) setQuiz(key, score, questions.length);
  }, [answered, score, key, questions.length]);

  const reset = () => {
    setPicked(questions.map(() => null));
    setTyped(questions.map(() => ''));
    setChecked(questions.map(() => false));
  };

  return (
    <section className="quiz" aria-label={title}>
      <header className="qz-head">
        <h2>{title}</h2>
        <span className="qz-score" aria-live="polite">
          {answered === questions.length ? `${score} of ${questions.length} right` : `${answered} of ${questions.length} answered`}
        </span>
      </header>
      <ol className="qz-list">
        {questions.map((q, i) => {
          const done = checked[i];
          const ok = done && correct(i);
          return (
            <li key={i} className={`qz-item ${done ? (ok ? 'is-right' : 'is-wrong') : ''}`}>
              <p className="qz-q">{inline(q.q)}</p>
              {isNumeric(q) ? (
                <form
                  className="qz-num"
                  onSubmit={(e) => {
                    e.preventDefault();
                    if (typed[i].trim()) setChecked((c) => c.map((v, j) => (j === i ? true : v)));
                  }}
                >
                  <input
                    inputMode="decimal"
                    aria-label="Your answer"
                    placeholder="your answer"
                    value={typed[i]}
                    disabled={done}
                    onChange={(e) => setTyped((t) => t.map((v, j) => (j === i ? e.target.value : v)))}
                  />
                  {q.unit && <span className="qz-unit">{q.unit}</span>}
                  {!done && (
                    <button type="submit" className="btn ghost">
                      Check
                    </button>
                  )}
                  {done && (
                    <span className={`qz-verdict ${ok ? 'ok' : 'no'}`}>
                      <Icon name={ok ? 'check' : 'close'} size={16} />{' '}
                      {ok ? 'Right' : `Answer: ${q.answer.toLocaleString('en-US')}${q.unit ? ' ' + q.unit : ''}`}
                    </span>
                  )}
                </form>
              ) : (
                <div className="qz-opts" role="radiogroup">
                  {q.options.map((o, j) => {
                    const state = !done ? (picked[i] === j ? 'picked' : '') : j === q.answer ? 'answer' : picked[i] === j ? 'wrong' : 'ghost';
                    return (
                      <button
                        type="button"
                        role="radio"
                        aria-checked={picked[i] === j}
                        key={j}
                        className={`qz-opt ${state}`}
                        disabled={done}
                        onClick={() => {
                          setPicked((p) => p.map((v, k) => (k === i ? j : v)));
                          setChecked((c) => c.map((v, k) => (k === i ? true : v)));
                        }}
                      >
                        <span className="qz-mark" aria-hidden="true">
                          {state === 'answer' ? <Icon name="check" size={14} /> : state === 'wrong' ? <Icon name="close" size={14} /> : String.fromCharCode(65 + j)}
                        </span>
                        <span>{inline(o)}</span>
                      </button>
                    );
                  })}
                </div>
              )}
              {done && <p className="qz-why">{inline(q.why)}</p>}
            </li>
          );
        })}
      </ol>
      {answered > 0 && (
        <button type="button" className="btn ghost qz-reset" onClick={reset}>
          <Icon name="restart" size={15} /> Try again
        </button>
      )}
    </section>
  );
}
