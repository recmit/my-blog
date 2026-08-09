# Migration plan: fastpages/Jekyll → Astro

**Status:** Phase 5 complete. Next: Phase 6.
**Audience:** the implementer (human or agent). Start with `/migrate`.

The Status line above is the handoff signal — keep it in the form
`Phase N complete. Next: Phase N+1.` so the next agent can read it.

This file is a *work order*, not permanent documentation. It is prescriptive on
purpose. Delete it once the migration lands. The docs it tells you to write
(`AGENTS.md`, `docs/code-map.md`) are the opposite: short and permanent.

---

## Why

The site is built by [fastpages](https://github.com/fastai/fastpages), deprecated
since 2022. The build runs inside `docker://fastai/fastpages-jekyll@sha256:6db4173…`,
an unmaintained image, on Ruby 2.7-era gems. If that image disappears, the site
cannot be rebuilt. Astro replaces it: content stays portable, interactive pages
become possible, and nothing depends on an abandoned toolchain.

## Invariants — breaking any of these fails the migration

1. **Every published URL keeps working.** Exact list in [Appendix A](#appendix-a).
2. **`david-recio.com` stays live on GitHub Pages.** No DNS changes, no host change.
3. **The five 2022 posts keep their content.** Do not rewrite, reword, re-order,
   or "improve" the prose or the analysis. Additive changes only (see Phase 5).
4. **Notebooks are never re-executed.** Outputs stored in the `.ipynb` files are
   the source. No 2022 Python environment is needed and none should be built.
5. **`master` is not touched** until the user explicitly approves the cutover.

## Pre-made decisions — do not relitigate these

| Decision | Choice |
|---|---|
| Framework | Astro 6, `output: 'static'` |
| Host | GitHub Pages (unchanged) |
| Notebook handling | Convert once to Markdown, commit the result, freeze |
| Post format | `.md` (use `.mdx` only if a post needs a component) |
| Islands | None yet. No React/Svelte/Vue until there's a reason |
| Styling | Hand-written CSS + a few custom properties. No Tailwind, no Bootstrap |
| Search | Pagefind (replaces lunr) |
| Math | Rendered at build time (`remark-math` + `rehype-katex`), CSS only |
| Analytics | Keep GA4 `G-J4ZRL7L1T1` |
| Contact form | Keep Formspree. No backend |
| Package manager | npm, lockfile committed |

If something in this table turns out to be wrong or impossible, **stop and ask** —
do not silently substitute an alternative.

## Do not

- Do not re-execute notebooks or install pandas/torch/transformers.
- Do not change any URL in Appendix A.
- Do not delete Jekyll files before Phase 7.
- Do not merge anything, and do not push to `master`. End your phase with a
  pull request and leave it open. Do not open PRs mid-phase, and do not include
  work from a phase you were not assigned.
- Do not add an SSR adapter, server endpoints, or Cloudflare config. Static only.
- Do not redesign wholesale. Improvements in Phase 5 are the agreed scope.
- Do not write docs longer than the budgets in Phase 6.

---

## Target structure

```
CLAUDE.md               → one line, points at AGENTS.md
AGENTS.md               → how to work in this repo
README.md               → what this is, how to run it
docs/
  code-map.md           → where things live and why
  decisions.md          → short log of choices and their reasons
notebooks/              → *.ipynb, source of truth, renamed from _notebooks/
scripts/
  convert-notebooks.sh  → one-time conversion, not run in CI
src/
  content/posts/<slug>/index.md + images
  layouts/  components/  pages/  styles/
public/                 → CNAME, robots.txt, static images
tests/                  → urls.test.ts, smoke.spec.ts
astro.config.mjs  package.json  .nvmrc
.github/workflows/deploy.yml
```

---

## Phases

Each phase ends with a **gate**. Do not start the next phase until the gate
passes. Commit at every gate.

**One agent per group, in sequence — never two at once.** The groups below are
context boundaries, not parallel tracks: the dependency graph is close to
linear, and a fresh agent starting from these docs beats one carrying 600 KB of
notebook JSON in its context. Each group ends with a pull request; the owner
reviews and merges before the next begins.

| Agent | Phases |
|---|---|
| A | 0–1 — baseline, parity test, scaffold |
| B | 2 — notebook conversion. Isolated: the notebooks are 1.8 MB, ~95% of it stored outputs and base64 |
| C | 3–4 — routes, layouts, feature parity |
| D | 5 — improvements. Reads all five posts to draft TL;DRs, so it gets its own agent |
| E | 6 — tests and docs |
| F | 7 — cutover, only once approved |

Agent D should work post by post — draft, commit, move on — rather than
reading all five before writing anything.

Do not split a phase across agents. In particular, do not convert the five
notebooks with five agents: the conversion script is written once and run five
times, or you get five inconsistent conversions.

### Phase 0 — Baseline and the parity test

- Fetch `origin/gh-pages` (the currently deployed output) into a scratch dir.
  It is the ground truth for both URLs and rendered appearance.
- Save the URL list from Appendix A as `tests/expected-urls.json`.
- Save a copy of one rendered post (e.g. `2022/03/19/grades-analysis.html`) for
  visual comparison later.
- **Write `tests/urls.test.ts` now**, asserting that every path in
  `expected-urls.json` exists in `dist/`. It will fail — there is no `dist/`
  yet. That is correct; commit it red.

The test is written here, before any routing exists, and by a different agent
than the one that builds the routes. If the agent building routes also writes
the test that checks them, a misreading of the URL scheme lands in both and the
test passes while every inbound link breaks.

**Gate:** `tests/expected-urls.json` contains the 13 paths from Appendix A, and
`tests/urls.test.ts` exists.

### Phase 1 — Scaffold Astro alongside Jekyll

Astro and Jekyll can coexist: Astro owns `src/`, `public/`, `astro.config.mjs`,
`package.json`; Jekyll owns the `_*` directories. They do not collide.

- `npm create astro@latest` — minimal template, TypeScript strict.
- Add `src/`, `public/`, `node_modules`, `dist`, `package.json`, `package-lock.json`
  to `exclude:` in `_config.yml` so the Jekyll build ignores them. (`tests` is
  already excluded — Phase 0 creates that directory, so it excludes it too.)
- Pin Node in `.nvmrc` and use the same version in CI.
- Define the content collection schema in `src/content.config.ts`:

  ```ts
  {
    title: string
    description: string
    date: Date
    tldr: string[]          // hand-written summary bullets, Phase 5
    notebook: string | undefined  // path to source .ipynb
    archived: boolean       // true for all five 2022 posts
    tags: string[]          // default []
    draft: boolean          // default false
  }
  ```

**Gate:** `npm run build` produces a `dist/`, the Jekyll build still succeeds,
and the parity test runs and fails for the right reason — missing routes, not a
syntax error.

### Phase 2 — Notebook conversion (highest risk — do it early)

This is where surprises live. If something is going to derail the migration, it
will be here, so it happens before any layout work.

- `git mv _notebooks notebooks`. Losing the leading underscore makes Jekyll
  *publish* the directory, so add `notebooks` (and `scripts`) to `exclude:` in
  `_config.yml`, and repoint `_action_files/nb2post.py`, `tsconfig.json` and
  `.gitignore` at the new name.
- Write `scripts/convert-notebooks.sh`: `jupyter nbconvert --to markdown`, one
  output directory per post, images extracted to real files alongside the `.md`.
- Handle the fastpages cell directives, which nbconvert knows nothing about:
  `#hide` drops the cell entirely (prose cells too, not only code), and
  `#collapse-hide` / `#collapse-output` fold the input or the output into a
  `<details>`. The deployed pages are the check on all three.
- Take front matter (`title`, `description`, date) from the **existing**
  `_posts/*.md` files — it is already correct there. Do not re-derive it from
  the notebooks.
- Post-process: fix image paths, strip nbconvert artifacts, drop the leftover
  `keywords: fastai` field.
- Run once, commit the output. The script is kept for reference; **it does not
  run in CI**.

Check each of the five posts for: LaTeX math, syntax-highlighted code, matplotlib
plots, pandas HTML tables, and long output blocks.

**Gate:** all five posts exist as Markdown, no base64 remains in
`src/content/posts/`, and the total size of the five posts is well under the
~1.9 MB the current `_posts/` occupies.

### Phase 3 — Routes and layouts

- Base layout: `<head>`, nav, footer, GA4.
- Post layout, page layout.
- Post route generating the `.html` URLs from Appendix A. The site mixes
  formats — posts are `/…/slug.html`, pages are `/about/` — so the build uses
  `build.format: 'preserve'`: each route lands where its source file sits, and
  every page route is a directory with an `index.astro` in it. The parity test
  is what proves it correct.
- Pages: `/` (home), `/about/`, `/contact/`, `/search/`, `/404.html`.
- Port About and Contact prose verbatim from `_pages/`.

**Gate:** every *page* in the parity test passes — the five posts, `/`,
`/about/`, `/contact/`, `/search/`, `/404.html`. Its three remaining paths
(`feed.xml`, `sitemap.xml`, `robots.txt`) are Phase 4's, and stay red until
then. Do not edit the test to make it pass — if it disagrees with your routes,
your routes are wrong.

### Phase 4 — Feature parity

- `feed.xml` (`@astrojs/rss`), `sitemap.xml`, `robots.txt`, `CNAME` in `public/`.
  Do not reach for `@astrojs/sitemap`: it publishes `sitemap-index.xml` rather
  than the frozen `/sitemap.xml`, and it derives URLs from the config instead of
  the built files, so under `build.format: 'preserve'` every URL it emits is
  wrong. A ~40-line endpoint does it correctly.
- SEO meta + Open Graph (replaces `jekyll-seo-tag`).
- Pagefind search on `/search/`.
- Math CSS.
- Styles: readable measure, code blocks, tables, figures with captions.
  Self-host what you need — **drop the Primer and FontAwesome CDN links**
  currently in `_includes/custom-head.html`.

**Gate:** feed, sitemap and search all work in a local preview build, and the
parity test now passes in full — this is the phase that turns its last three
paths green.

### Phase 5 — Improvements (agreed scope — nothing beyond this)

- **Home page**: short intro + post list, replacing the current bare `# Posts`.
- **TL;DR block** on each archived post: 3–5 bullets in the `tldr` front matter
  field, rendered above the content. Draft these; the owner reviews and edits.
  Three rules, because this is the only *authored* content in the migration and
  it carries the owner's name:

  - **Anchor on the post's existing `description` field.** The owner wrote those
    in 2022 and they are accurate. Expand them into bullets — do not invent a
    fresh framing or re-characterise the work.
  - **Say what the post concluded**, not only what it did. The descriptions
    already cover the method; the findings sit at the bottom of the notebook and
    are the reason a reader keeps going.
  - **Flag every bullet you are less than certain about** in the PR description.
    A fluent, confident, wrong summary of someone else's statistics is the worst
    output this migration can produce. Uncertainty is cheap to check; a wrong
    claim published under the owner's name is not.

  Put all five drafts in the PR description so they can be read together without
  opening five files.
- **Archive note** on each 2022 post: one line, e.g. *"Written in 2022, when I
  was moving from mathematics into industry. Left as it was."*
- **Reading time** (currently commented out at `_layouts/post.html:39`).
- **Dark mode**, respecting `prefers-color-scheme`.
- **Image optimization** via Astro's `<Image>`.
- **Tags**: add real tags to the five posts. Drop the dead category machinery.

Ask before adding anything not on this list.

**Gate:** `astro check` clean, the parity test still green in full, and every
item above visible in a preview build under both colour schemes.

### Phase 6 — Tests and docs

Testing, proportionate to a five-post blog — these four, no more:

1. `astro check` — types and content schema. Cheapest, catches the most.
2. **URL parity** — already written in Phase 0 and passing since Phase 3. Wire
   it into CI here. It is the most important test in the repo: the thing between
   you and silently breaking every inbound link.
3. **Smoke tests** (Playwright) — each page returns 200, has an `<h1>`, and logs
   no console errors. Chromium is already available; do not run `playwright install`.
   GA4 and the contact page's reCAPTCHA load from CDNs, so the console-error
   assertion needs egress or an allowance for those two hosts.
4. **Internal link check** over `dist/` — catches bad paths from Phase 2.

All four run in CI on every push to the branch.

Docs — **hard budgets, high level, no edge-case specs**:

| File | Budget | Content |
|---|---|---|
| `CLAUDE.md` | 3 lines | Points at `AGENTS.md`. Nothing else |
| `AGENTS.md` | ≤ 60 lines | Commands, conventions, the invariants above, what not to touch |
| `docs/code-map.md` | ≤ 60 lines | Directory-by-directory: what lives there and why. A reader should locate any file in under a minute |
| `docs/decisions.md` | ≤ 40 lines | One short entry per decision: what, why, what it rules out |
| `README.md` | ≤ 40 lines | What the site is, how to run it, how to add a post. Replaces the fastpages boilerplate |

If a doc wants to exceed its budget, cut it instead.

**Gate:** all four checks green; every doc within budget.

### Phase 7 — Cutover (only with explicit user approval)

- Replace `.github/workflows/ci.yaml` with an Astro build →
  `actions/upload-pages-artifact` → `actions/deploy-pages` workflow.
- Delete in one commit: `Gemfile`, `Gemfile.lock`, `_config.yml`, `_layouts/`,
  `_includes/`, `_sass/`, `_plugins/`, `_posts/`, `_action_files/`,
  `_fastpages_docs/`, `_word/`, `_pages/`, `docker-compose.yml`, `Makefile`,
  `settings.ini`, `index.html`, `assets/`, `.devcontainer.json`, and the
  `check_cdns`, `check_config`, `docker-nbdev`, `upgrade`, `gh-page` workflows.
- Keep: `notebooks/`, `images/` (as needed), `CNAME`, `LICENSE`, `.gitattributes`.
- Also delete, now that their jobs are done: `tests/baseline/` (the deployed
  `origin/gh-pages` branch remains the permanent record of the old site, so the
  baseline is only a working convenience for Phase 2) and
  `.claude/skills/migrate/` along with this plan.

**Gate:** URL parity passes against the deleted-Jekyll build. Then hand back to
the user for review and merge.

### Phase 8 — Optional, after merge

Connect the repo to Cloudflare Pages for per-branch preview URLs. Production
stays on GitHub Pages. Purely additive; do not migrate hosting.

---

## Appendix A — URLs that must not break

Extracted from the deployed `origin/gh-pages` branch, so this is what is
actually live, not an inference.

```
/
/2022/03/19/grades-analysis.html
/2022/03/19/podcasts-recommender.html
/2022/04/11/titanic-leak.html
/2022/10/21/bert-fine-tune-podcast-reviews.html
/2022/10/21/vader-bert-podcast-reviews.html
/about/
/contact/
/search/
/404.html
/feed.xml
/sitemap.xml
/robots.txt
```

Note the two formats: posts end in `.html`, pages use trailing slashes. Both
must be preserved exactly. There are no pagination or category pages — the
`categories` layout exists in the Jekyll source but was never published.

## Appendix B — Known quirks in the current site

Things you will encounter; all are fixed by the phases above.

- `_includes/custom-head.html` loads **both** KaTeX *and* MathJax 2.7.5, plus
  `kramdown: math_engine: katex` in `_config.yml`. Three math paths for one job.
  Phase 4 replaces all of it with build-time rendering.
- Primer CSS and FontAwesome load from CDN; a whole workflow
  (`check_cdns.yaml`) exists to watch them. Self-hosting removes both.
- Client-side JS wraps images in `<figure>` and injects anchor icons. Do this at
  build time instead.
- `settings.ini` is unmodified nbdev boilerplate — `lib_name = nbdev`,
  `author = Sylvain Gugger and Jeremy Howard`, `baseurl = /my-blog`. Never used
  by anything that matters. Delete it.
- `_layouts/default.html` does not exist locally; it comes from
  `remote_theme: jekyll/minima@69664442…`, pinned to a 2020 commit and fetched
  at build time. Astro replaces it with a local layout.
- `_posts/*.md` are nbconvert output, not hand-written. Nothing there is
  authored content — the `.ipynb` files are.

---

## Handoff log

**This log is the handoff between agents**, not the pull request descriptions —
it arrives in `master` when a phase's PR merges, so the next agent finds it by
reading the repo. PR descriptions are written for the owner.

One entry per phase, appended at the end. **Ten lines maximum each.** Only
three things: what differed from the plan, what the next agent needs to know,
and anything left broken. If a step in the plan was wrong, fix the step — do
not describe the discrepancy here.

<!-- Phase 0: -->

**Phase 0** — Appendix A verified against `origin/gh-pages` @ `d447eaf`: all 13
paths resolve to non-empty files, list unchanged.
`tests/urls.test.ts` targets **vitest**; Phase 1 must add it plus an `npm test`
script. Until then it is red for want of vitest, not routing. Its URL→file
mapping (trailing slash → `index.html`, else verbatim) is validated against the
deployed output, so a Phase 3 failure means the routes are wrong.
`tests` was excluded in `_config.yml` here, not Phase 1 — Phase 0 creates it,
and otherwise Jekyll publishes the 450 KB baseline live on merge. Phase 1's
step is edited to match.
`.gitignore` has a bare `*.xml` that would swallow XML committed outside `dist/`.

<!-- Phase 1: -->

**Phase 1** — `npm create astro@latest` can't fetch templates here: egress policy
403s `codeload.github.com` and `api.github.com`, which it reports as "Template
minimal could not be found". **`raw.githubusercontent.com` works**, and PyPI and
npm are exempt (so Phase 2's `pip install nbconvert` is fine). The scaffold was
hand-written, then diffed against `examples/minimal` at tag `astro@6.4.8`; it
matches but for what this repo needs (`site`/`output` set, and `tsconfig`
excludes the Jekyll dirs, else `astro check` typechecks `assets/js/search.js`,
which is Liquid). `content.config.ts` imports `z` from `astro/zod`; the
`astro:content` re-export goes away in Astro 7. No docker daemon and no runnable
Ruby here, so CI on the PR is the only proof the Jekyll build still works.

<!-- Phase 2: -->

**Phase 2** — nbconvert was uneventful; the fastpages directives were the work,
and the step above now documents them. Dropping `#hide` cells also removed the
only ANSI escapes, so nothing needed unmangling. Check fidelity by asserting
every retained cell's source appears verbatim in the output — not by diffing the
deployed HTML, whose KaTeX and Liquid make text comparison useless.
Math is intact as LaTeX but **unrendered until Phase 4** adds `remark-math`;
until then remark reads `_` in `$y_{i,j}$` as emphasis — install the plugin,
do not "escape" the math. Images are `![](./output_<cell>_<n>.png)` with empty
alt text (nbconvert's `![png]` names the format, not the figure); Phase 5 owns
real alt text. `notebooks/{README.md,ghtop_images/,my_icons/}` go in Phase 7.

<!-- Phase 3: -->

**Phase 3** — The two URL shapes forced `build.format: 'preserve'`; the step and
`docs/decisions.md` now record it. Consequence for later phases: **a new page is
`src/pages/<name>/index.astro`**, never `<name>.astro`, or it publishes as
`<name>.html`. Post URLs in `astro dev` have no `.html` — only the built files
do, so check routing against `astro preview`, not `dev`.
Date→URL conversion reads UTC (`src/lib/posts.ts`); a local-time build would
move a post a day back. Markdown images are already optimized to webp by
default, so Phase 5's `<Image>` item is about alt text and figures, not formats.
Left dangling on purpose: the footer's `/feed.xml` link, and `/search/`, which
has the input markup but no Pagefind. Both are Phase 4.
`docs/decisions.md` is now over its Phase 6 budget — cut there.

<!-- Phase 4: -->

**Phase 4** — All 13 parity paths green. **`@astrojs/sitemap` was tried and
reverted**: it derives URLs from the config, not the built files, so under
`build.format: 'preserve'` it emits posts without `.html` and pages without their
slash — every URL a 404. `src/pages/sitemap.xml.ts` derives both shapes instead,
finding pages with `import.meta.glob`, and `tests/urls.test.ts` checks the result.
Math plugins go through `processor: unified({...})` from `@astrojs/markdown-remark`;
`markdown.remarkPlugins` is deprecated in Astro 6. nbconvert left four `$$…$$` on
single lines in grades-analysis, which remark-math renders inline, not display — the
delimiters were rewrapped (no math or prose changed). Pagefind (Component UI) indexes
after `astro build`, so **search is empty under `astro dev`**; use `npm run preview`.

<!-- Phase 5: -->

**Phase 5** — This phase had no gate; one is added above, matching the others.
Colab leftovers survive in the two 2022-10-21 posts (a **Phase 2** miss): the
`<style>` blocks are live CSS, and the `<script>` blocks are indented, so they
render as a visible plaintext code block instead of executing. That means no
console error — **the Phase 6 smoke test will not catch them.**
Astro's `<Image>` was not needed: markdown images already build to sized, lazy
webp, so the work was the 31 alt texts. Tags are labels, not links — no tag
pages, so no new URLs. Colours now live only in `:root` and one
`prefers-color-scheme` block; a literal hex anywhere else breaks one scheme.
`docs/decisions.md` is 79 lines against its 40-line budget — cut in Phase 6.
