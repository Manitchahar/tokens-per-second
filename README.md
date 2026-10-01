# Tokens per Second

A free, visual, first-principles course in **LLM inference engineering**, from zero to hero.

**Live site:** https://manitchahar.github.io/tokens-per-second/

29 modules and 142 lessons take you from "how a number lives in memory" to GPUs, transformers, the KV cache, batching and PagedAttention, vLLM, quantization, profiling, Triton kernels, FlashAttention, speculative decoding, multi-GPU and MoE serving, production operations and economics, and finally building your own mini inference engine.

Every lesson has:

- a scrollytelling animation drawn as a GPU profiler trace (amber = compute, blue = memory, green = tokens, violet = communication)
- a first-principles derivation with numbers you can check
- code, a quiz, key notes, and a per-module cheat sheet
- hands-on labs, either as in-browser simulators or as Colab notebooks that run on the free T4 GPU

Progress, quiz scores and personal notes are saved in your browser. No account needed.

## Run it locally

```bash
npm install
npm run dev
```

Then open http://localhost:4321.

## Build

```bash
npm run build      # compiles notebooks-src/*.py to public/notebooks/*.ipynb, then builds dist/
```

For a sub-path deploy (e.g. GitHub Pages) set `SITE_URL` and `BASE_PATH`:

```bash
SITE_URL=https://manitchahar.github.io BASE_PATH=/tokens-per-second npm run build
```

Pushing to `main` deploys automatically through `.github/workflows/deploy.yml`.

## Project layout

| Path | What |
|---|---|
| `src/data/course.ts` | The syllabus: parts, modules and lessons |
| `src/content/lessons/<module>/*.mdx` | Lessons |
| `src/components/viz/` | Interactive visuals (`lib/` is the shared toolkit) |
| `src/content/cheatsheets/*.mdx` | Cheat sheets |
| `notebooks-src/*.py` | Colab labs in percent format |
| `src/data/glossary*` | Glossary terms |
| `AUTHORING.md` | The guide for writing new lessons |
