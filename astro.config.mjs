// @ts-check
import { unified } from '@astrojs/markdown-remark';
import { defineConfig } from 'astro/config';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';

// https://astro.build/config
export default defineConfig({
  site: 'https://david-recio.com',
  output: 'static',
  markdown: {
    // Two Shiki themes rather than one. With `defaultColor: false` every token
    // carries both colours as `--shiki-light` / `--shiki-dark` custom
    // properties and the stylesheet picks one, so code follows
    // `prefers-color-scheme` like the rest of the page. A single theme would
    // leave code blocks the one element that ignores dark mode.
    shikiConfig: {
      themes: { light: 'github-light', dark: 'github-dark' },
      defaultColor: false,
    },
    // Math is rendered once, here, at build time. The Jekyll site loaded KaTeX,
    // MathJax and a kramdown math engine to do the same job in the browser;
    // this ships the resulting HTML plus `katex.min.css` and no JavaScript.
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex],
    }),
  },
  // The published URLs mix two shapes — posts are `/2022/03/19/slug.html`,
  // pages are `/about/` — and both are load-bearing (tests/expected-urls.json).
  // `preserve` writes each route where its source file says, so a page at
  // `pages/about/index.astro` becomes `about/index.html` while the post route
  // emits a bare `.html` file. `directory` would wrap the posts in
  // `slug.html/index.html`; `file` would flatten the pages to `about.html`.
  build: { format: 'preserve' },
});
