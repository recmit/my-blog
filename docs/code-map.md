# Code map

Where things live and why.

## Content

| Path | What |
|---|---|
| `notebooks/*.ipynb` | Source of truth for the five 2022 posts. Kept for provenance; read by the conversion script, never executed |
| `src/content/posts/<slug>/` | One directory per post: `index.md` and its images |
| `src/content.config.ts` | Front matter schema. A typo in a post's front matter fails the build here, loudly, instead of rendering blank |

## Site

| Path | What |
|---|---|
| `src/pages/` | One file per route. Pages are `<name>/index.astro` (they publish as `/<name>/`); `[...slug].astro` generates the dated `.html` post URLs; `feed.xml.ts` and `sitemap.xml.ts` are endpoints rather than pages |
| `src/layouts/` | Page shells — `<head>`, nav, footer |
| `src/components/` | Reusable pieces. Build-time unless marked `client:*` |
| `src/lib/` | Non-rendering helpers. `posts.ts` owns the date→URL mapping, so the published paths are derived in exactly one place |
| `src/styles/` | Plain CSS |
| `public/` | Copied verbatim to the site root: `images/favicon.ico`, `CNAME`, `robots.txt`. Pagefind writes its index into `dist/pagefind/` after the build, so it never appears here |

## Build and checks

| Path | What |
|---|---|
| `astro.config.mjs` | Integrations, Markdown pipeline, URL format |
| `scripts/convert-notebooks.sh` | One-time `.ipynb` → Markdown. **Not run in CI.** It ran once; its output is committed. Installs its two dependencies, then runs the `.py` |
| `scripts/convert-notebooks.py` | The conversion itself: drops fastpages `#hide` cells, folds `#collapse-*` into `<details>`, strips Colab's dataframe widgets. Took front matter from Jekyll's `_posts/`, deleted at the cutover — so it is now a record of how the posts were made, not a runnable tool |
| `scripts/nbtemplate/` | nbconvert template that emits those `<details>` wrappers |
| `tests/expected-urls.json` | The published URL surface, taken from the live deployment |
| `tests/*.test.ts` | vitest, run against `dist/`: URL parity and the internal link check |
| `tests/smoke.spec.ts` | Playwright. The only check that renders the site — needs a build and `astro preview`, both of which its config starts |
| `vitest.config.ts` / `playwright.config.ts` | Which runner claims which files: `.test.ts` is vitest's, `.spec.ts` is Playwright's |
| `.github/workflows/checks.yml` | The four checks, on every push, every branch |
| `.github/workflows/deploy.yml` | Build → GitHub Pages, on pushes to `master`. Uploads `dist/` as a Pages artifact; there is no deploy branch |

## Not in this repo

DNS for `david-recio.com`, the Formspree endpoint, the Google Analytics
property, and the Pages source setting (**GitHub Actions**, not a branch — the
deploy workflow depends on it) are configured outside it. Changing them is not
a code change.

The `gh-pages` branch still holds the last Jekyll deployment, as the record of
what the site was before the migration. Nothing writes to it.
