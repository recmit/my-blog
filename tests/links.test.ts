/**
 * Internal link check: every site-relative `href`/`src` in the built HTML must
 * resolve to a file in `dist/`.
 *
 * `tests/urls.test.ts` protects the URLs the outside world links to. This one
 * protects the links the site makes to itself — the ones the notebook
 * conversion rewrote by hand, where a stale `../images/` path builds cleanly
 * and 404s only when someone clicks it.
 */

import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, posix, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const testsDir = dirname(fileURLToPath(import.meta.url));
const distDir = resolve(testsDir, '..', 'dist');

/** Every `.html` under `dist/`, as site-root-relative paths. */
function htmlUnder(dir: string, prefix = '/'): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const child = posix.join(prefix, entry.name);
    if (entry.isDirectory()) return htmlUnder(join(dir, entry.name), child);
    return entry.name.endsWith('.html') ? [child] : [];
  });
}

const htmlFiles = existsSync(distDir) ? htmlUnder(distDir).sort() : [];

/** `href`/`src` on any element, single or double quoted. */
const ATTR = /(?:href|src)\s*=\s*["']([^"']*)["']/gi;

/** Anything with a scheme, a protocol-relative `//host`, or a bare fragment. */
function isExternal(url: string): boolean {
  return /^[a-z][a-z0-9+.-]*:/i.test(url) || url.startsWith('//') || url.startsWith('#');
}

/** The file `dist/` would serve for a link found in `fromFile`. */
function targetFor(url: string, fromFile: string): string {
  const path = url.split(/[?#]/)[0];
  const absolute = path.startsWith('/')
    ? posix.normalize(path)
    : posix.resolve(posix.dirname(fromFile), path);
  // A directory-style URL is served by its index.html — including the bare
  // `/`, which leaves a path with no filename at all.
  const file = absolute.endsWith('/') ? posix.join(absolute, 'index.html') : absolute;
  return join(distDir, file);
}

function linksIn(file: string): string[] {
  const html = readFileSync(join(distDir, file), 'utf8');
  return [...html.matchAll(ATTR)]
    .map((m) => m[1].trim())
    .filter((url) => url !== '' && !isExternal(url));
}

describe('internal links in dist/', () => {
  it('has a build to check', () => {
    expect(
      htmlFiles.length,
      `no HTML in ${distDir} — run \`npm run build\` first`,
    ).toBeGreaterThan(0);
  });

  it.each(htmlFiles)('%s links only to files that exist', (file) => {
    const broken = linksIn(file)
      .map((url) => ({ url, target: targetFor(url, file) }))
      .filter(({ target }) => !existsSync(target) || !statSync(target).isFile())
      .map(({ url, target }) => `${url} → ${relative(distDir, target)}`);

    expect(broken, `${file} links to files that are not published`).toEqual([]);
  });
});
