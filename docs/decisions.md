# Decisions

One entry each: what, why, what it rules out.

**Astro, replacing fastpages.** fastpages was deprecated in 2022 and built in an
unmaintained Docker image on Ruby 2.7-era gems; the site was becoming
unbuildable. Astro beat Quarto (notebook-first), Hugo (no component model) and
Next.js (framework overhead on five posts). Rules out little — content stays Markdown.

**Notebooks frozen, out of the build.** Converted to Markdown once and
committed, never re-executed; fastpages never did either, so the saved outputs
have always been what shipped, and Python stays out of the build for good. The
conversion reproduces what fastpages published, not the raw notebook — `#hide`
cells dropped, `#collapse-*` folded into `<details>`, Colab widgets stripped —
so it costs a converter that knows a dead tool's directives.

**GitHub Pages, for now.** Static hosting is enough and the domain already points
there; an adapter can move it later. Rules out a backend — hence Formspree.

**URLs frozen, `build.format: 'preserve'`.** Pages has no server-side redirects,
so a changed path is a hard 404. The published URLs come in two shapes — posts
`/2022/03/19/slug.html`, pages `/about/` — and `preserve` is the only format
that keeps both. The list is pinned in `tests/expected-urls.json` and enforced
by a test, not by convention. Costs: every page is a directory with an
`index.astro` in it, and post URLs lack `.html` in `astro dev`.

**Sitemap hand-rolled.** `@astrojs/sitemap` derives URLs from the config rather
than the built files, so under `preserve` every URL it emitted was a 404, and it
publishes `sitemap-index.xml`, not the frozen `/sitemap.xml`.
`src/pages/sitemap.xml.ts` derives both shapes, finding pages by convention — one
placed outside it is missed, which `tests/urls.test.ts` catches.

**Dark mode by `prefers-color-scheme`.** No toggle: that needs client-side
JavaScript and a stored preference, and the OS already holds one. Shiki emits
both themes per token so code follows too. Costs: colours may only be defined in
`:root` and that one media block.

**Four checks, no more.** `astro check`, URL parity, smoke tests, internal link
check — proportionate to a five-post blog. No unit tests: there is barely any
logic to test, and the build either produces the right files or it does not.
