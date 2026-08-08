# Code map

Where things live and why. This describes the target structure — the Astro tree
arrives with the migration ([`migration-plan.md`](migration-plan.md)).

## Content

| Path | What |
|---|---|
| `notebooks/*.ipynb` | Source of truth for the five 2022 posts. Kept for provenance; read by the conversion script, never executed |
| `src/content/posts/<slug>/` | One directory per post: `index.md` and its images |
| `src/content.config.ts` | Front matter schema. A typo in a post's front matter fails the build here, loudly, instead of rendering blank |

## Site

| Path | What |
|---|---|
| `src/pages/` | One file per route. Pages are `<name>/index.astro` (they publish as `/<name>/`); `[...slug].astro` generates the dated `.html` post URLs; `feed.xml.ts` is an endpoint rather than a page (the sitemap comes from `@astrojs/sitemap`) |
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
| `scripts/convert-notebooks.py` | The conversion itself: drops fastpages `#hide` cells, folds `#collapse-*` into `<details>`, takes front matter from `_posts/` |
| `scripts/nbtemplate/` | nbconvert template that emits those `<details>` wrappers |
| `tests/expected-urls.json` | The published URL surface, taken from the live deployment |
| `tests/` | URL parity, Playwright smoke tests, internal link check |
| `.github/workflows/deploy.yml` | Build → GitHub Pages |
| `.claude/skills/migrate/` | The `/migrate` command. Temporary — delete once the migration lands |

## Still present, due for deletion

The Jekyll site fastpages built. Removed in the final migration phase, not
before: `_config.yml`, `Gemfile*`, `_layouts/`, `_includes/`, `_sass/`,
`_plugins/`, `_posts/`, `_pages/`, `_action_files/`, `_fastpages_docs/`,
`_word/`, `assets/`, `images/`, `Makefile`, `settings.ini`,
`docker-compose.yml`, `index.html`, `.devcontainer.json`.

Note that `_posts/*.md` is generated output, not authored content — the
notebooks are the originals.

## Not in this repo

DNS for `david-recio.com`, the Formspree endpoint, and the Google Analytics
property are configured outside it. Changing them is not a code change.
