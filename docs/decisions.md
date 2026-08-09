# Decisions

One entry each: what, why, what it rules out.

**Astro, replacing fastpages.** fastpages was deprecated in 2022 and built in an
unmaintained Docker image on Ruby 2.7-era gems, so the site was becoming
unbuildable. Beat Quarto (notebook-first), Hugo (no component model), Next.js
(framework overhead on five posts). Rules out little — content stays Markdown.

**Notebooks frozen, out of the build.** Converted to Markdown once and
committed, never re-executed; fastpages never did either, so the saved outputs
have always been what shipped, and Python stays out of the build for good. It
reproduces what fastpages published, not the raw notebook — `#hide` cells
dropped, `#collapse-*` folded, Colab widgets stripped.

**GitHub Pages, deployed as an artifact.** Static hosting is enough, the domain
already points there, and an adapter can move it later. Rules out a backend —
hence Formspree. `deploy.yml` uploads `dist/` rather than committing to
`gh-pages` as fastpages did, leaving that branch as the record of the old site.
Costs a setting only GitHub holds: Pages source must be **GitHub Actions**.

**URLs frozen, `build.format: 'preserve'`.** Pages has no server-side redirects,
so a changed path is a hard 404. The URLs come in two shapes — posts
`/2022/03/19/slug.html`, pages `/about/` — and `preserve` is the only format
keeping both. Pinned in `tests/expected-urls.json`, enforced by a test. Costs:
every page is a directory with an `index.astro`, and no `.html` in `astro dev`.

**Sitemap hand-rolled.** `@astrojs/sitemap` derives URLs from the config, not the
built files, so under `preserve` every URL it emitted was a 404 — and it publishes
`sitemap-index.xml`, not the frozen `/sitemap.xml`. `src/pages/sitemap.xml.ts`
derives both shapes by convention; a page placed elsewhere is missed, and
`tests/urls.test.ts` catches that.

**Dark mode by `prefers-color-scheme`.** No toggle: that needs client-side JS and
a stored preference, and the OS already holds one. Shiki emits both themes per
token so code follows. Costs: colours live only in `:root` and that media block.

**Four checks, no more.** `astro check`, URL parity, smoke tests, internal link
check — proportionate to a five-post blog. No unit tests: barely any logic to
test, and the build either produces the right files or it does not.
