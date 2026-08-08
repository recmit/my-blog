// @ts-check
import { unified } from '@astrojs/markdown-remark';
import sitemap from '@astrojs/sitemap';
import { defineConfig } from 'astro/config';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';

// https://astro.build/config
export default defineConfig({
  site: 'https://david-recio.com',
  output: 'static',
  markdown: {
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
  // Publishes `sitemap-index.xml` plus numbered chunks, not a single
  // `sitemap.xml` — the one frozen URL the owner chose to give up, so that the
  // page list is discovered rather than hand-maintained. `robots.txt` points at
  // the index. `/404.html` is excluded: it is not a page to crawl.
  //
  // `serialize` repairs the URLs. The integration does not understand
  // `build.format: 'preserve'`: it strips the `.html` from post paths (which
  // then 404) and the trailing slash from page paths. Both shapes are restored
  // here, matching `postPath()` in src/lib/posts.ts. `tests/urls.test.ts`
  // checks the emitted URLs against the same list, because getting this wrong
  // is invisible in a build log.
  integrations: [
    sitemap({
      filter: (page) => !page.endsWith('/404.html'),
      serialize(item) {
        const { pathname } = new URL(item.url);
        if (/^\/\d{4}\/\d{2}\/\d{2}\/[^/]+$/.test(pathname)) {
          item.url = `${item.url}.html`;
        } else if (pathname !== '/' && !pathname.endsWith('/')) {
          item.url = `${item.url}/`;
        }
        return item;
      },
    }),
  ],
});
