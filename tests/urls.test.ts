/**
 * URL parity: every path the live site publishes must still exist after the
 * build. The list in `expected-urls.json` was taken from the deployed
 * `origin/gh-pages` branch, so it is what is actually live, not an inference.
 *
 * If this test disagrees with the routes, the routes are wrong. Do not edit
 * the expected list to make it pass — inbound links depend on these paths.
 *
 * One path is not from the deployment: the owner traded `/sitemap.xml` for
 * `/sitemap-index.xml` in Phase 4 to get page discovery from
 * `@astrojs/sitemap` (see `docs/decisions.md`). Nothing links to a sitemap.
 * That is the only such exception, and it took an explicit decision to make.
 */

import { readFileSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const testsDir = dirname(fileURLToPath(import.meta.url));
const distDir = resolve(testsDir, '..', 'dist');

const expectedUrls: string[] = JSON.parse(
  readFileSync(join(testsDir, 'expected-urls.json'), 'utf8'),
);

/**
 * A published URL maps to a file in `dist/`. Directory-style URLs are served
 * by their `index.html`; everything else is the file itself. The site uses
 * both forms — posts end in `.html`, pages use trailing slashes — and that
 * distinction is exactly what this test exists to protect.
 */
function distPathFor(url: string): string {
  const relative = url.replace(/^\//, '');
  return url.endsWith('/') ? join(distDir, relative, 'index.html') : join(distDir, relative);
}

function statOrNull(path: string) {
  try {
    return statSync(path);
  } catch {
    return null;
  }
}

describe('expected-urls.json', () => {
  it('lists the published URL surface', () => {
    expect(expectedUrls.length).toBeGreaterThan(0);
    expect(new Set(expectedUrls).size).toBe(expectedUrls.length);
    for (const url of expectedUrls) {
      expect(url.startsWith('/'), `${url} must be a site-root-relative path`).toBe(true);
    }
  });
});

describe('URL parity against dist/', () => {
  it('has a build to check', () => {
    expect(
      statOrNull(distDir)?.isDirectory() ?? false,
      `${distDir} does not exist — run \`npm run build\` first`,
    ).toBe(true);
  });

  it.each(expectedUrls)('serves %s', (url) => {
    const path = distPathFor(url);
    const stat = statOrNull(path);

    expect(stat !== null, `${url} is not published — expected ${path}`).toBe(true);
    expect(stat!.isFile(), `${url} resolved to ${path}, which is not a file`).toBe(true);
    expect(stat!.size, `${url} is published but empty`).toBeGreaterThan(0);
  });
});

/**
 * The sitemap advertises URLs to crawlers, so a wrong path there is the same
 * bug as a missing route — just invisible, because the file still builds.
 *
 * `@astrojs/sitemap` does not understand `build.format: 'preserve'` and emits
 * post paths without their `.html` and page paths without their trailing
 * slash; `serialize` in astro.config.mjs repairs both. This is the check on
 * that repair.
 */
describe('sitemap contents', () => {
  /** Everything a crawler should see: pages and posts, not 404 or the feeds. */
  const crawlable = expectedUrls.filter(
    (url) => !['/404.html', '/feed.xml', '/sitemap-index.xml', '/robots.txt'].includes(url),
  );

  function sitemapUrls(): string[] {
    const xml = readFileSync(join(distDir, 'sitemap-0.xml'), 'utf8');
    return [...xml.matchAll(/<loc>(.*?)<\/loc>/g)].map((m) =>
      m[1].replace('https://david-recio.com', ''),
    );
  }

  it('lists every crawlable URL, and only those', () => {
    expect(sitemapUrls().sort()).toEqual(crawlable.sort());
  });
});
