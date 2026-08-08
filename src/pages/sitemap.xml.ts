/**
 * `/sitemap.xml` — hand-rolled rather than `@astrojs/sitemap`, which publishes
 * a `sitemap-index.xml` pointing at numbered chunks. The live site has a single
 * `/sitemap.xml` and `robots.txt` names that exact path, so the integration
 * would have moved a published URL to buy pagination this site cannot use.
 *
 * Pages are listed by hand: there are five of them, and the alternative is
 * globbing routes to rediscover something a reader can see in one screen.
 */
import type { APIRoute } from 'astro';
import { getPublishedPosts, postPath } from '../lib/posts';

const PAGES = ['/', '/about/', '/contact/', '/search/'];

export const GET: APIRoute = async ({ site }) => {
  const base = site!.href.replace(/\/$/, '');
  const posts = await getPublishedPosts();

  const entries = [
    ...posts.map((post) => ({
      loc: base + postPath(post),
      lastmod: post.data.date.toISOString(),
    })),
    ...PAGES.map((page) => ({ loc: base + page, lastmod: undefined })),
  ];

  const body = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${entries
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
