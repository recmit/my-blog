# AGENTS.md

How to work in this repo. Keep it this short.

An Astro site on GitHub Pages, migrated from fastpages/Jekyll in 2026. Pushing
to `master` builds and deploys it; there is no deploy branch. The old site
remains on the `gh-pages` branch as a record — nothing writes to it now.

## Commands

| Command | What |
|---|---|
| `npm run dev` | Local preview at `localhost:4321`, hot reload |
| `npm run build` | Static build into `dist/` |
| `npx astro check` | Types and content schema. Run before every commit |
| `npm test` | Builds, then runs URL parity, the internal link check and the smoke tests |

Those four are the whole test suite, and all four run in CI on every push. They
check the built site, not the source — so `npm test` builds first, and a stale
`dist/` is never what you are testing.

If you are an agent, run `ASTRO_PREVIEW_BACKGROUND=1 npm test`. Astro 7's
`astro preview` detects an agentic environment and daemonises itself, which
Playwright reads as the server exiting early — the smoke tests then fail before
running. That variable suppresses the detection, leaving preview in the
foreground. CI is unaffected, so this never shows up on a push.

## Layout

Where things live: [`docs/code-map.md`](docs/code-map.md).
Why they're that way: [`docs/decisions.md`](docs/decisions.md).

## Conventions

- Posts are Markdown in `src/content/posts/<slug>/`, images alongside them.
  Use `.mdx` only when a post genuinely needs a component.
- A page is `src/pages/<name>/index.astro`, never `<name>.astro` — that is what
  publishes it at `/<name>/`, and it is how the sitemap finds it. Nothing lists
  pages by hand; both follow from the file's location.
- Front matter is validated by `src/content.config.ts`. Add a field there before
  using it anywhere.
- Plain CSS with custom properties, in `src/styles/`. No CSS framework.
- Components render at build time by default. A `client:*` directive ships
  JavaScript to visitors — reach for it deliberately, not by habit.

## Invariants

1. **Published URLs never change.** The list is `tests/expected-urls.json`,
   enforced by the URL parity test. Inbound links depend on these.
2. **Notebooks are never executed.** `notebooks/*.ipynb` carry their saved
   outputs. The conversion script reads them; it does not run them. No ML
   dependencies belong in this repo.
3. **The five 2022 posts are archives.** Add TL;DRs, tags, and formatting fixes.
   Do not rewrite the prose or redo the analysis.
4. **Static only.** No SSR adapter, no server endpoints — GitHub Pages serves
   files and nothing else. The contact form uses Formspree for this reason.

## Ask first

Before adding a dependency, a client-side island, or a new top-level directory.
