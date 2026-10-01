import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import react from '@astrojs/react';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// Set SITE_URL (and BASE_PATH for GitHub Pages project sites, e.g. "/tokens-per-second")
// when you deploy. Both are optional for local use.
const BASE = (process.env.BASE_PATH || '/').replace(/\/?$/, '/');

/** Prefix root-relative links and images in lesson Markdown ("/learn/...") with the deploy base path. */
function rehypeBaseLinks() {
  const fix = (v) => (typeof v === 'string' && v.startsWith('/') && !v.startsWith('//') && !v.startsWith(BASE) ? BASE + v.slice(1) : v);
  const walk = (node) => {
    if (node.type === 'element' && node.properties) {
      if ('href' in node.properties) node.properties.href = fix(node.properties.href);
      if ('src' in node.properties) node.properties.src = fix(node.properties.src);
    }
    if (node.children) node.children.forEach(walk);
  };
  return (tree) => { if (BASE !== '/') walk(tree); };
}

export default defineConfig({
  site: process.env.SITE_URL || 'https://tokens-per-second.example',
  base: BASE,
  trailingSlash: 'always',
  // Parallel build workers share node_modules, so each can point its caches elsewhere.
  ...(process.env.ASTRO_CACHE_DIR ? { cacheDir: process.env.ASTRO_CACHE_DIR } : {}),
  vite: process.env.VITE_CACHE_DIR ? { cacheDir: process.env.VITE_CACHE_DIR } : {},
  integrations: [react(), mdx()],
  markdown: {
    processor: unified({ remarkPlugins: [remarkMath], rehypePlugins: [[rehypeKatex, { strict: false }], rehypeBaseLinks] }),
    syntaxHighlight: 'shiki',
    shikiConfig: { theme: 'css-variables', wrap: false },
  },
});
