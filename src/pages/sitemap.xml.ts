/**
 * `/sitemap.xml` — hand-rolled rather than `@astrojs/sitemap`.
 *
 * The integration asks the config what shape URLs are instead of looking at
 * what was built, so under `build.format: 'preserve'` it strips the `.html`
 * off posts and the trailing slash off pages and publishes URLs that 404. It
 * also cannot serve a single `sitemap.xml`; it publishes `sitemap-index.xml`
 * plus numbered chunks, which would have moved a frozen URL.
 *
 * Here both URL shapes are derived, not guessed: posts from `postPath()` — the
 * one function that owns post URLs, shared with the routes and the feed — and
 * pages from the files themselves.
 */
import type { APIRoute } from 'astro';
import { getPublishedPosts, postPath } from '../lib/posts';

/**
 * Every page route, found at build time. Vite replaces this with an object
 * keyed by matching file path; nothing scans the disk at request time.
 *
 * The pattern encodes this repo's page convention: a page is
 * `src/pages/<name>/index.astro`, so that `build.format: 'preserve'` publishes
 * it at `/<name>/`. A bare `<name>.astro` would publish at `/<name>.html`
 * instead — and would not be found here. `404.astro` is such a file, which is
 * why it needs no explicit exclusion: it is not a page to crawl.
 */
const pageFiles = import.meta.glob('./**/index.astro');

/** `./about/index.astro` → `/about/`, `./index.astro` → `/`. */
function pagePath(file: string): string {
  return file.replace(/^\./, '').replace(/index\.astro$/, '');
}

export const GET: APIRoute = async ({ site }) => {
  const base = site!.href.replace(/\/$/, '');

  const posts = await getPublishedPosts();
  const postEntries = posts.map((post) => ({
    loc: base + postPath(post),
    lastmod: post.data.date.toISOString(),
  }));

  const pageEntries = Object.keys(pageFiles)
    .map(pagePath)
    .sort()
    .map((path) => ({ loc: base + path, lastmod: undefined }));

  const body = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${[...postEntries, ...pageEntries]
  .map(
    ({ loc, lastmod }) =>
      `  <url>\n    <loc>${loc}</loc>${lastmod ? `\n    <lastmod>${lastmod}</lastmod>` : ''}\n  </url>`,
  )
  .join('\n')}
</urlset>
`;

  return new Response(body, {
    headers: { 'Content-Type': 'application/xml' },
  });
};
