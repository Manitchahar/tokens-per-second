# Authoring guide: Tokens per Second

You are writing one module of **Tokens per Second**, a free, visual, first-principles course that takes a learner from zero to strong inference engineer. Read this whole file before writing anything. Then study the finished reference modules and match their quality, depth and voice.

- `src/content/lessons/big-picture/*.mdx`, `numbers-and-math/*.mdx`, `gpu-first-principles/*.mdx` (finished lessons)
- `src/components/viz/trace/*.tsx`, `viz/math/*.tsx`, `viz/gpu/*.tsx` (finished visuals)
- `src/components/viz/lib/*` (the shared visual toolkit; read `Stage.tsx`, `svg.tsx`, `useStep.ts`, `tween.ts`, `perfModel.ts`)
- `src/content/cheatsheets/*.mdx`, `notebooks-src/*.py` (finished cheat sheets and labs)
- `src/data/course.ts` (the syllabus: every module and lesson slug, title, kind and minutes; **do not change it**)
- `PRODUCT.md` (who the learner is and the product principles)

## Who the learner is

They can write a little Python (loops, functions). They know nothing else: no ML, no GPUs, no Linux. They are a **visual learner** who wants everything **spoon-fed from first principles**: no hand-waving, no jargon before it is explained, every rule of thumb shown with the arithmetic that produces it. Their goal is to become a genuinely strong ("cracked") inference engineer. Do not dumb things down. Go deep, but build every step on the previous one. Assume they finished every earlier module in `course.ts` order, and refer back to those lessons by name when you reuse an idea.

Voice: a patient mentor. Plain words first, then the precise term. Short paragraphs. Concrete numbers (real models such as Llama-3.1-8B/70B and Qwen2.5, and real GPUs such as T4, A100, H100, H200, B200). Public spec-sheet numbers only, marked approximate where they vary. Never invent benchmarks, quotes, customers or citations. When describing a real engine's behaviour (vLLM, SGLang, TensorRT-LLM, llama.cpp), describe what is well established and say "at the time of writing" for anything version-specific.

## What you must produce for your module

For a module with slug `<m>` (from `course.ts`):

1. **One MDX file per lesson**: `src/content/lessons/<m>/<lesson-slug>.mdx`, with the slugs exactly as in `course.ts`, every lesson, lab and project included.
2. **Visuals**: React components in `src/components/viz/<m>/` (create the folder). At least **one bespoke interactive or step-through visual per `lesson`**, plus a simulator for in-browser labs where it makes sense.
3. **Cheat sheet**: `src/content/cheatsheets/<m>.mdx` with frontmatter `title: 'Module NN · Title'` and `module: <m>`. Include every formula, number and rule of thumb from the module.
4. **Colab notebook(s)**: for every lab/project that needs a real GPU, write `notebooks-src/NN-<short-name>.py` in percent format (`# %% [markdown]` text cells, `# %%` code cells; see `notebooks-src/01-tensors.py`). They must run top to bottom on a free Colab **T4 (16 GB, FP16, no BF16 tensor cores, no FP8)**. Use small models (Qwen2.5-0.5B/1.5B-Instruct, SmolLM2, TinyLlama) so they fit. Mention the lab's notebook in its frontmatter as `colab: NN-<short-name>.ipynb` and add `<Colab nb="NN-<short-name>.ipynb" />` in the lab page. Labs that are pure in-browser simulators need no notebook.
5. **Glossary terms**: new terms go in `src/data/glossary/<m>.ts`:
   ```ts
   import type { GlossaryEntry } from '../glossary';
   export const terms: Record<string, GlossaryEntry> = {
     'prefix-cache': { term: 'Prefix cache', def: 'Plain-words definition.', lesson: '<m>/<lesson-slug>' },
   };
   ```
   Use `<Term t="key">text</Term>` in prose the first time a term appears. Existing keys are in `src/data/glossary.ts`.

**Only create or edit files inside the five locations above.** Do not edit shared files: `src/components/viz/lib/*`, layouts, pages, `course.ts`, `glossary.ts`, other modules, global CSS or package.json. If you need a new helper, put it inside `src/components/viz/<m>/`.

## Lesson anatomy (every `lesson`)

```mdx
---
title: 'Exact title from course.ts'
summary: 'One or two sentences: what they will understand and why it matters.'
objectives:
  - Verb-led, checkable objective
  - 3 to 5 of them
---
import MyViz from '@/components/viz/<m>/MyViz.tsx';
import Quiz from '@/components/learn/Quiz.tsx';

Hook: 2-3 short paragraphs. The concrete problem, in plain words, and why an inference engineer cares.

## Section heading

<Scrolly>
  <MyViz client:visible slot="stage" />

  <Step title="1. Short step title">
    2-5 sentences. Each step changes ONE thing in the visual. Use **bold** for the key idea.
  </Step>
  ... 5 to 8 steps ...
</Scrolly>

## Then derive it properly
First-principles derivation with KaTeX math ($inline$ and $$display$$), worked numeric examples,
tables, Python code blocks (```python), and Callouts:
<Callout type="intuition|mentor|pitfall|math|real|try|recap" title="optional">...</Callout>

<KeyNotes>
  1. 5-8 numbered takeaways, each a full sentence.
</KeyNotes>

<Quiz client:visible questions={[
  { q: 'Multiple choice?', options: ['A', 'B', 'C', 'D'], answer: 1, why: 'Explanation.' },
  { q: 'Calculation?', numeric: true, answer: 42, tolerance: 1, unit: 'GB', why: 'Show the arithmetic.' },
]} />
```

Targets per lesson: about **1,800–3,000 words** of prose, a 5–8 step Scrolly, 2–4 Callouts, at least one worked numeric example, code where relevant, 5–8 KeyNotes, and a **5–6 question quiz with at least 2 numeric questions**. Labs and projects: a clear goal, setup, step-by-step tasks with the expected output and how to read it, "check yourself" questions, and KeyNotes. A Quiz is optional for labs.

Projects should be substantial: a deliverable (a report, a harness, a matrix, a design doc), acceptance criteria and a rubric.

## Visual rules (the "profiler trace" world)

- Build on `Stage` + `useStep` from `src/components/viz/lib`. The pattern is in every reference visual:
  ```tsx
  const ref = useRef<HTMLDivElement>(null);
  const s = useStep(STEPS.length, ref);
  return <Stage rootRef={ref} title="..." steps={STEPS} captions={CAPTIONS} legend={[...]} {...s}>
    <svg viewBox="0 0 640 320" role="img" aria-label="...">...</svg>
  </Stage>;
  ```
  The number of `STEPS` must equal the number of `<Step>`s in the Scrolly. `captions` are used when the visual is standalone.
- **Colour means the kind of work, always:** `compute` = amber, `memory` = blue (memory traffic and KV cache), `token` = green (tokens produced), `comm` = violet (network/interconnect), `error` = red (OOM, failure, waste), `idle` = hatched grey (waiting, bubbles). Neutral data uses `text`/greys. Never use colour decoratively. Use the `Span`, `Label`, `Arrow`, `Ruler`, `Track`, `Dim`, `Chip`, `Move`, `Fade`, `Matrix`, `Bar` primitives and the `COLOR`/`DIM` maps from `svg.tsx`.
- Animate with `Move`/`Fade` (CSS transitions) and `useTween`/`useTweens`/`useClock`. Motion should be smooth and calm (3Blue1Brown style), never flashy.
- Use `Slider`, `Segmented` and `Readout` from `Stage.tsx` for interactivity. Use `perfModel.ts` for speed, memory and cost math instead of inventing new constants.
- Keep **all SVG content inside the viewBox** (width 640, height 280–380). Pinned stages must fit the screen. Leave label room; avoid overlapping text. Use `fmt` from `svg.tsx` for numbers.
- `Fade` children are always evaluated even when hidden, so clamp array indexes used inside them.
- Accessibility: every `<svg>` needs `role="img"` and an `aria-label` describing what it shows.

## Hard syntax rules (these break the build or the page)

1. **Quote** frontmatter `title`/`summary` with single quotes if they contain `: ` or start with a special character. Escape an apostrophe inside as `''`.
2. Prose currency must be `\$5` (backslash-dollar), because single `$` starts math. Inline math must not start with a digit; write `$\frac{1}{x}$` or plain text "3 × 5" instead of `$3 \times 5$`. After writing, run `node scripts/fix-dollars.mjs src/content/lessons/<m>` and re-check.
3. In MDX prose, `{` and `}` start JavaScript. Only use braces inside math, code blocks or JSX props. Write `<` and `>` in prose as `&lt;`/`&gt;` or reword. Never use HTML comments `<!-- -->` in MDX; use `{/* */}`.
4. In JSX/TSX text nodes never write `→`-style escapes. Use the real character (→, ×, ≈, —) or `{'→'}`.
5. Quiz props are JavaScript strings: escape `'` as `’` or use typographic ’ quotes. `answer` is a 0-based index for options.
6. `<Step>` and `<Callout>` contents are Markdown: leave blank lines between paragraphs and do not indent by 4+ spaces.
7. **Hydration-safe visuals:** the first render must be identical on the server and in the browser. Never read `window`, `matchMedia`/`prefersReducedMotion()`, `localStorage`, `Math.random()` or `Date.now()` in `useState` initializers or in the render body. Do that inside `useEffect`. For pseudo-random layouts use a deterministic seeded function (see `rand` in `src/components/viz/trace/StackZoom.tsx`).
8. Every lesson imports `Quiz` and its visuals with the `@/` alias. `Scrolly`, `Step`, `Callout`, `KeyNotes`, `Term`, `Colab`, `Figure` and `Wide` need no import.

## Definition of done (check every item before you stop)

- [ ] Every lesson, lab and project slug of your module in `course.ts` has its MDX file with real, complete content. No TODOs, placeholders or "coming soon".
- [ ] Each `lesson` has a bespoke visual in `src/components/viz/<m>/` driven by a Scrolly whose step count matches.
- [ ] The cheat sheet `src/content/cheatsheets/<m>.mdx` exists and is complete.
- [ ] Notebooks for GPU labs exist in `notebooks-src/` and are referenced from the lab's frontmatter and `<Colab>`.
- [ ] New glossary terms are in `src/data/glossary/<m>.ts`.
- [ ] `npm run build` passes with **zero errors**. Run it yourself and fix every error, then run it again.
- [ ] Re-read each lesson once as the learner: is every term explained before it is used? Is every number derived?
