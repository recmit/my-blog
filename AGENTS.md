# AGENTS.md

How to work in this repo. Keep it this short.

**Status:** migrating from fastpages/Jekyll to Astro. The Jekyll tree (`_*`
directories, `Gemfile`) still publishes the live site; the Astro tree (`src/`,
`astro.config.mjs`) is being built alongside it. Phase order and gates:
[`docs/migration-plan.md`](docs/migration-plan.md). The Jekyll tree is deleted
last, and only with the owner's approval.

Run a single phase with `/migrate`. **If that isn't recognized** — project
skill discovery can be lazy in cloud sessions, registering only once something
reads the directory — try, in order:

1. Ask Claude to check the repo for custom skills. This alone has fixed it.
2. Paste directly: "Follow `.claude/skills/migrate/SKILL.md` to run the next
   migration phase."

## Commands

| Command | What |
|---|---|
| `npm run dev` | Local preview at `localhost:4321`, hot reload |
| `npm run build` | Static build into `dist/` |
| `npx astro check` | Types and content schema. Run before every commit |
| `npm test` | URL parity, smoke tests, internal link check |

## Layout

Where things live: [`docs/code-map.md`](docs/code-map.md).
Why they're that way: [`docs/decisions.md`](docs/decisions.md).

## Conventions

- Posts are Markdown in `src/content/posts/<slug>/`, images alongside them.
  Use `.mdx` only when a post genuinely needs a component.
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

Before adding a dependency, a client-side island, a new top-level directory, or
anything outside the current phase of the migration plan.
