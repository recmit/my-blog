# david-recio.com

Personal blog. Posts on statistics and machine learning, written while moving
from pure mathematics research into software engineering.

Live at **<https://david-recio.com>**.

Built with [Astro](https://astro.build) and published to GitHub Pages by
`.github/workflows/deploy.yml` on every push to `master`.

## Running it locally

```sh
npm install
npm run dev      # http://localhost:4321
npm run preview  # builds first — the only way to see search and the final URLs
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
