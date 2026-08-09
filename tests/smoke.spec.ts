/**
 * Smoke tests: every page loads, is titled, and does not throw in the browser.
 *
 * The other checks read the built files. This one is the only place the site
 * is actually rendered, so it catches what static inspection cannot — a
 * stylesheet that 404s, a script that throws, a page that builds to an empty
 * shell.
 *
 * Runs against `npm run preview`, not `astro dev`: the built output is what
 * ships, and Pagefind's index only exists after a build.
 */

import { expect, test } from '@playwright/test';
import expectedUrls from './expected-urls.json' with { type: 'json' };

/** The HTML pages. `feed.xml`, `sitemap.xml` and `robots.txt` are not pages. */
const pages = (expectedUrls as string[]).filter(
  (url) => url.endsWith('/') || url.endsWith('.html'),
);

/**
 * Third-party scripts we load on purpose and do not control. Both are blocked
 * or noisy on a network without egress, and neither failing says anything
 * about this site's own code, so console noise from these hosts is ignored —
 * anything else is a failure.
 */
const THIRD_PARTY = [/googletagmanager\.com/, /google\.com\/recaptcha/, /gstatic\.com/];

const isOurs = (...where: string[]) =>
  !THIRD_PARTY.some((host) => where.some((text) => host.test(text)));

for (const url of pages) {
  test(`${url} loads cleanly`, async ({ page }) => {
    const problems: string[] = [];

    page.on('console', (message) => {
      // A failed subresource logs "Failed to load resource: …" with no URL in
      // the text — the host is only in `location()`, so both are checked.
      const from = message.location().url;
      if (message.type() === 'error' && isOurs(message.text(), from)) {
        problems.push(`console: ${message.text()} (${from})`);
      }
    });
    page.on('pageerror', (error) => {
      if (isOurs(error.message)) problems.push(`uncaught: ${error.message}`);
    });
    page.on('requestfailed', (request) => {
      if (isOurs(request.url())) problems.push(`failed request: ${request.url()}`);
    });

    const response = await page.goto(url);

    expect(response?.status(), `${url} did not return 200`).toBe(200);
    await expect(page.locator('h1')).toHaveCount(1);
    await expect(page.locator('h1')).not.toBeEmpty();
    expect(problems, `${url} reported browser errors`).toEqual([]);
  });
}
