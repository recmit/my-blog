# Decisions

Why the repo is the way it is. One entry each: what, why, and what it rules out.

## Astro, replacing fastpages — 2026-07

fastpages was deprecated in 2022 and built inside an unmaintained Docker image
on Ruby 2.7-era gems; the site was on track to become unbuildable.

Astro beat Quarto (notebook-first, and this blog is moving away from notebooks),
Hugo (no component model, so interactive pages mean hand-rolling), and Next.js
(application-framework overhead on a five-post site). It keeps interactive
elements and a future backend available without charging for them up front.

Rules out: little. Content stays portable Markdown, so a later move is cheap.

## Notebooks frozen, out of the build

The five 2022 posts were converted to Markdown once and committed, and the
notebooks were not re-executed — fastpages never did either, so the saved
outputs have always been what shipped. This keeps Python and a 2022 ML
dependency tree out of the build permanently.

The conversion reproduces what fastpages published, not the raw notebook:
`#hide` cells are dropped, `#collapse-*` cells become `<details>`. Costs: a
converter that knows a dead tool's directives, and republishing by hand.

## GitHub Pages, for now

Static hosting is sufficient and the domain already points there. Astro can move
to a host with server support later by adding an adapter, so this is not a
one-way door.

Rules out: a backend. The contact form goes through Formspree as a result.

## `build.format: 'preserve'`

The frozen URLs come in two shapes: posts are `/2022/03/19/slug.html`, pages are
`/about/`. Astro's `directory` format would publish the posts as
`slug.html/index.html`; `file` would flatten the pages to `about.html`.
`preserve` writes each route where its source file sits, so both shapes coexist.

Costs: every page route is a directory with an `index.astro` inside, and post
URLs in `astro dev` lack the `.html` the built files carry.

## Sitemap hand-rolled, not `@astrojs/sitemap`

The integration was tried and reverted: it derives URLs from the config rather
than the built files, so under `build.format: 'preserve'` it stripped the
`.html` off every post and the slash off every page, and it publishes
`sitemap-index.xml` rather than the frozen `/sitemap.xml`. Patching its output
would have restated this site's URL rules as guesses about someone else's.

`src/pages/sitemap.xml.ts` derives both shapes: posts from `postPath()`, pages
from `import.meta.glob`. Nothing is listed by hand.

Costs: a page outside the `<name>/index.astro` convention is not found —
`tests/urls.test.ts` turns that into a failure rather than a silent omission.

## URLs frozen

The published URL surface predates this repo's current shape, and GitHub Pages
serves files with no server-side redirects — so a changed path is a hard 404,
not a redirect. Keeping the paths costs one build setting; replacing them would
cost a stub page per old URL. It is pinned in `tests/expected-urls.json` and
enforced by a test, rather than left to convention and good intentions.
