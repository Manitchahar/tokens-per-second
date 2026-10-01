import { defineCollection } from 'astro:content';
import { z } from 'astro/zod';
import { glob } from 'astro/loaders';

const lessons = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/lessons' }),
  schema: z.object({
    title: z.string(),
    summary: z.string(),
    objectives: z.array(z.string()).default([]),
    colab: z.string().optional(),
  }),
});

const cheatsheets = defineCollection({
  loader: glob({ pattern: '*.mdx', base: './src/content/cheatsheets' }),
  schema: z.object({
    title: z.string(),
    module: z.string(),
  }),
});

export const collections = { lessons, cheatsheets };
