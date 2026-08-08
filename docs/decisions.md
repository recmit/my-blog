# Decisions

Why the repo is the way it is. One entry each: what, why, and what it rules out.

## Astro, replacing fastpages — 2026-07

fastpages was deprecated in 2022, and the build ran inside an unmaintained
Docker image on Ruby 2.7-era gems; the site was on track to become unbuildable.

Astro was chosen over Quarto (notebook-first, and this blog is moving away from
notebooks), Hugo (no component model, so interactive pages mean hand-rolling),
and Next.js (application-framework overhead on a five-post content site). Astro
keeps interactive elements and a future backend available without charging for
them up front.

Rules out: little. Content stays portable Markdown, so a later move is cheap.

## Notebooks frozen, out of the build

The five 2022 posts were converted to Markdown once and committed. The
notebooks were not re-executed — fastpages never did either, so the saved
outputs have always been what shipped. This keeps Python and a 2022 ML
dependency tree out of the build permanently.

Costs: republishing a notebook means re-running the conversion script by hand.
Acceptable, because these posts are archives.

The conversion reproduces what fastpages published rather than the raw notebook:
`#hide` cells are dropped and `#collapse-*` cells become `<details>`, so the
posts read as they always have. That keeps the archives honest, at the price of
a converter that has to know a dead tool's directives.

## GitHub Pages, for now

Static hosting is sufficient and the domain already points there. Astro can move
to a host with server support later by adding an adapter, so this is not a
one-way door.

Rules out: a backend. The contact form goes through Formspree as a result.

## URLs frozen

The published URL surface predates the current shape of this repo and is linked
from job applications. It is pinned in `tests/expected-urls.json` and enforced
by a test, rather than left to convention and good intentions.
