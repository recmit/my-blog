// @ts-check
import { defineConfig } from 'astro/config';

// https://astro.build/config
export default defineConfig({
  site: 'https://david-recio.com',
  output: 'static',
  // The published URLs mix two shapes — posts are `/2022/03/19/slug.html`,
  // pages are `/about/` — and both are load-bearing (tests/expected-urls.json).
  // `preserve` writes each route where its source file says, so a page at
  // `pages/about/index.astro` becomes `about/index.html` while the post route
  // emits a bare `.html` file. `directory` would wrap the posts in
  // `slug.html/index.html`; `file` would flatten the pages to `about.html`.
  build: { format: 'preserve' },
});
