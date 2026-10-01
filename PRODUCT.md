# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Astro static site (user choice). Lessons authored in MDX; interactive visuals are client-side islands that hydrate only where used. Output is plain static HTML so it can be hosted free (GitHub Pages, Netlify, Vercel, Cloudflare Pages). No accounts or server: progress, notes and quiz state live in the learner's browser.

## Users

A motivated learner who can write a little Python (loops, functions) and knows nothing about ML systems, GPUs or LLM serving. They are a visual learner: they understand an idea when they can watch it move and poke at it, and they want every concept derived from first principles ("spoon-fed", no hand-waving). Their goal is to become a genuinely strong ("cracked") inference engineer, able to reason end to end from a single matrix multiply up to a production multi-GPU fleet and its cost.

## Product Purpose

**Tokens per Second** is one complete, free-standing course in inference engineering. It takes a learner from zero to hero: how a token is produced, why the KV cache exists, how serving engines batch and page memory, how to benchmark, quantize, profile, write kernels, speculate, scale across GPUs, run production and price it. Success = someone who started knowing nothing can finish the course, explain every layer of the stack from first principles, and do the labs on real hardware.

## Positioning

Reference syllabus: fanout.sh/inference-eng (31 modules, 226 lessons). The user liked only its syllabus and found that site unintuitive and not engaging. Tokens per Second keeps the full topic coverage but restructures it the way a mentor would teach, and makes every idea visual: each lesson is scrollytelling, where a pinned animation advances as the learner reads, and can be stepped or played. The simulators run in the browser with no GPU; real-hardware labs run on free Google Colab.

## Operating Context

- Learners study on a laptop (primary) and review on a phone (cheat sheets, notes, quizzes).
- Hands-on GPU work happens in Jupyter notebooks opened in Google Colab (free T4 GPU). The author has no NVIDIA GPU locally.
- Supporting material the learner returns to: per-module cheat sheets, a master formula sheet, a glossary, calculators (model memory, KV cache, roofline, cost), personal notes and interview drills.

## Capabilities and Constraints

- Lesson anatomy: hook → intuition/analogy → scrollytelling animation → first-principles math → code → pitfalls/mentor notes → quiz → key notes.
- Progress tracking, quiz state and personal notes are stored locally (localStorage); no login. Accounts and payments are explicitly out of scope for now.
- Colab notebooks ship with the site; "Open in Colab" requires the repo to be public on GitHub (configurable), with a download fallback.
- Must work without JavaScript for reading text; animations and simulators progressively enhance.
- Undecided: final hosting provider and domain; whether to add search.

## Brand Commitments

- Name: **Tokens per Second**.
- The user asked for a 3Blue1Brown-style feel: dark canvas, calm focus, smooth animated diagrams. This is binding.
- Voice: a patient mentor. Plain words first, then the precise term. Never assumes knowledge that was not taught earlier in the course.

## Evidence on Hand

- No testimonials, learner counts, benchmarks of our own, or partner logos exist. Do not invent any.
- Hardware numbers quoted in lessons (GPU bandwidth, FLOPs, prices) must be public spec-sheet figures, labelled approximate where they vary.

## Product Principles

1. Nothing is used before it is taught. Every term is introduced from first principles the first time it appears.
2. Show it moving. If an idea can be animated or simulated, it is.
3. Numbers you can compute yourself. Every rule of thumb comes with the arithmetic that produces it.
4. Theory is paired with hands-on work. Every module ends in a lab: a browser simulator, a Colab notebook, or both.
5. Complete, not shallow. Advanced topics (kernels, MoE, disaggregation, building an engine) are kept and taught, not cut.

## Accessibility & Inclusion

- Animations respect `prefers-reduced-motion` (step-through controls replace autoplay).
- Every animation has step controls, so pace is always in the learner's hands, and its key state is also stated in text.
- Keyboard operable controls; WCAG AA contrast on the dark canvas.
