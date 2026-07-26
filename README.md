# david-recio.com

Personal blog. Posts on statistics and machine learning, written while moving
from pure mathematics research into software engineering.

Live at **<https://david-recio.com>**.

## Status

Migrating from [fastpages](https://github.com/fastai/fastpages), deprecated
since 2022, to [Astro](https://astro.build). The Jekyll build still publishes
the site; the Astro replacement is being built alongside it. Plan and phase
order: [`docs/migration-plan.md`](docs/migration-plan.md).

## Running it locally

```sh
npm install
npm run dev      # http://localhost:4321
```

## Adding a post

Create `src/content/posts/<slug>/index.md` with front matter matching
`src/content.config.ts`, and put its images in the same directory.

The five 2022 posts came from Jupyter notebooks, kept in `notebooks/` for
provenance. They are archives and are not rebuilt from source.

## Docs

- [`AGENTS.md`](AGENTS.md) — working conventions and invariants
- [`docs/code-map.md`](docs/code-map.md) — where things live
- [`docs/decisions.md`](docs/decisions.md) — why they're that way

## License

[Apache 2.0](LICENSE)
