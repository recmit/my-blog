#!/usr/bin/env python3
"""One-time conversion of the fastpages notebooks into Astro content.

Run through `scripts/convert-notebooks.sh`, which installs the two
dependencies first. Not run in CI: the output is committed, and re-running
this is only useful if the conversion itself needs fixing.

Notebooks are never executed. The outputs saved in the `.ipynb` files are the
source, so no 2022 Python environment is needed.

For each `notebooks/YYYY-MM-DD-<slug>.ipynb` this writes
`src/content/posts/<slug>/index.md` plus the images that post references.
Front matter comes from the matching `_posts/*.md`, which is where the
correct title and description already live.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import textwrap
from pathlib import Path

import nbformat
import yaml
from nbconvert import MarkdownExporter

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"
JEKYLL_POSTS = ROOT / "_posts"
OUT = ROOT / "src" / "content" / "posts"
TEMPLATE_BASEDIR = Path(__file__).resolve().parent / "nbtemplate"

# `2022-03-19-grades-analysis.ipynb` -> date `2022-03-19`, slug `grades-analysis`
NAME = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})-(?P<slug>.+)$")

# fastpages cell directives. `#hide` drops the cell from the published post;
# the two `#collapse-*` forms keep it but fold it away.
DIRECTIVE = re.compile(r"^\s*#\s*(hide|hide_input|collapse[-_](?:hide|output|show))\s*$")
# One cell carries a stray `#collapse-output` at the end of a code line
# (podcasts-recommender, cell 58) — a typo that fastpages ignored.
TRAILING_DIRECTIVE = re.compile(r"\s*#\s*(?:hide|collapse[-_](?:hide|output|show))\s*$")

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# pandas emits `<style scoped>` with every `to_html`. `scoped` was dropped from
# the HTML spec, so these would apply site-wide rather than to their table.
PANDAS_STYLE = re.compile(r"^<style scoped>.*?^</style>\n", re.DOTALL | re.MULTILINE)
# The two 2022-10-21 notebooks were run in Google Colab, which wraps every
# dataframe in a "convert to interactive table" widget: a container div, a
# button, an unscoped `<style>` and a `<script>` calling `google.colab`. None of
# it works outside Colab. Keep the table, drop the widget.
COLAB_TABLE = re.compile(
    r"^[ ]*<div id=\"df-[0-9a-f-]+\">\n"
    r"[ ]*<div class=\"colab-df-container\">\n"
    r"(?P<table>[ ]*<div>\n<table class=\"dataframe\">.*?</table>\n</div>)\n"
    r".*?^[ ]*</script>\n"
    r"[ ]*</div>\n"
    r"[ ]*</div>",
    re.DOTALL | re.MULTILINE,
)
IMAGE_REF = re.compile(r"!\[[^\]]*\]\((?P<path>[^)\s]+_files/[^)\s]+)\)")


def die(message: str) -> None:
    sys.exit(f"convert-notebooks: {message}")


def read_jekyll_front_matter(name: str) -> dict:
    """Title and description as published, from the generated Jekyll post."""
    path = JEKYLL_POSTS / f"{name}.md"
    if not path.exists():
        die(f"no Jekyll post at {path} to take front matter from")

    text = path.read_text(encoding="utf8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        die(f"{path} has no front matter block")

    data = yaml.safe_load(match.group(1))
    for field in ("title", "description"):
        if not data.get(field):
            die(f"{path} has no {field}")
    # `keywords: fastai` is fastpages boilerplate; `toc`, `nb_path` and `layout`
    # are Jekyll's. None of them carry over.
    return {"title": data["title"], "description": data["description"]}


def prepare(notebook: nbformat.NotebookNode) -> nbformat.NotebookNode:
    """Drop what the live site does not publish; record the collapse intent."""
    cells = []
    for index, cell in enumerate(notebook.cells):
        if index == 0 and is_fastpages_front_matter(cell):
            continue

        if cell.cell_type != "code":
            # `#hide` works on prose cells too (titanic-leak has one), and the
            # deployed page confirms fastpages drops them.
            first = cell.source.split("\n")[0] if cell.source else ""
            match = DIRECTIVE.match(first)
            if match and match.group(1) in ("hide", "hide_input"):
                continue
            cells.append(cell)
            continue

        lines = cell.source.split("\n")
        directive = None
        while lines:
            match = DIRECTIVE.match(lines[0])
            if not match:
                break
            directive = match.group(1)
            lines.pop(0)

        if directive in ("hide", "hide_input"):
            # `#hide` removes input and output both — verified against the
            # deployed pages, where these cells do not appear at all.
            continue

        source = "\n".join(lines)
        source = "\n".join(TRAILING_DIRECTIVE.sub("", line) for line in source.split("\n"))
        cell.source = source.strip("\n")

        if directive in ("collapse-hide", "collapse_hide"):
            cell.metadata["collapse"] = "input"
        elif directive in ("collapse-output", "collapse_output"):
            cell.metadata["collapse"] = "output"

        cells.append(cell)

    notebook.cells = cells
    return notebook


def is_fastpages_front_matter(cell: nbformat.NotebookNode) -> bool:
    """The first cell carries the post's title/description as fastpages markup.

    Astro takes those from real front matter instead, so the cell is dropped.
    Recognised by shape (`# title`, `> description`, `- toc:`) rather than by
    position alone, so a post that lacks one is not silently beheaded.
    """
    if cell.cell_type != "markdown":
        return False
    lines = [line.strip() for line in cell.source.strip().split("\n") if line.strip()]
    return (
        len(lines) >= 2
        and lines[0].startswith("# ")
        and any(line.startswith("> ") for line in lines)
    )


def rewrite_images(body: str, resources: dict, post_dir: Path) -> str:
    """Write extracted images beside index.md and point the Markdown at them.

    nbconvert names them `index_files/index_<cell>_<output>.png`; the nesting
    and the doubled `index` carry no information, so they land as
    `output_<cell>_<output>.png` in the post directory itself.
    """
    outputs = resources.get("outputs", {})
    written: dict[str, str] = {}

    for original, data in outputs.items():
        name = Path(original).name
        name = re.sub(r"^index_", "output_", name)
        (post_dir / name).write_bytes(data)
        written[original] = name

    def replace(match: re.Match) -> str:
        path = match.group("path")
        name = written.get(path) or written.get(Path(path).name)
        if name is None:
            die(f"image {path} referenced but not extracted")
        # Empty alt text: nbconvert's `![png]` describes the file format, not
        # the figure. Real alt text is a judgement call for a human.
        return f"![](./{name})"

    body, count = IMAGE_REF.subn(replace, body)
    if count != len(written):
        die(f"{count} image references for {len(written)} extracted images")
    return body


def clean(body: str) -> str:
    body = ANSI.sub("", body)
    body = PANDAS_STYLE.sub("", body)
    body = COLAB_TABLE.sub(
        lambda m: textwrap.dedent(m.group("table").replace("      <div>", "<div>", 1)),
        body,
    )
    # `border="1"` is presentation, and it overrides the stylesheet.
    body = body.replace('<table border="1" class="dataframe">', '<table class="dataframe">')
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip() + "\n"


def front_matter(meta: dict) -> str:
    """Serialise to YAML by hand so field order is stable and readable.

    JSON strings are valid YAML double-quoted scalars, which sidesteps the
    quoting in the source titles.
    """
    lines = ["---"]
    for key, value in meta.items():
        if isinstance(value, str) and key != "date":
            lines.append(f"{key}: {json.dumps(value)}")
        elif isinstance(value, bool):
            lines.append(f"{key}: {str(value).lower()}")
        elif isinstance(value, list):
            lines.append(f"{key}: []")
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def convert(path: Path, exporter: MarkdownExporter) -> tuple[str, int]:
    match = NAME.match(path.stem)
    if not match:
        die(f"{path.name} is not named YYYY-MM-DD-<slug>.ipynb")
    date, slug = match.group("date"), match.group("slug")

    notebook = prepare(nbformat.read(path, as_version=4))
    body, resources = exporter.from_notebook_node(
        notebook,
        resources={"unique_key": "index", "output_files_dir": "index_files"},
    )

    post_dir = OUT / slug
    if post_dir.exists():
        shutil.rmtree(post_dir)
    post_dir.mkdir(parents=True)

    body = rewrite_images(body, resources, post_dir)
    body = clean(body)

    meta = read_jekyll_front_matter(path.stem)
    meta["date"] = date
    meta["notebook"] = f"notebooks/{path.name}"
    meta["archived"] = True
    # `tldr` and `tags` are authored in Phase 5, not derived from the notebook.

    index = post_dir / "index.md"
    index.write_text(front_matter(meta) + body, encoding="utf8")

    images = len(list(post_dir.glob("*.png")))
    return slug, images


def main() -> None:
    if not NOTEBOOKS.is_dir():
        die(f"{NOTEBOOKS} does not exist")

    exporter = MarkdownExporter(
        template_name="fastpages-collapse",
        extra_template_basedirs=[str(TEMPLATE_BASEDIR)],
    )

    notebooks = sorted(NOTEBOOKS.glob("*.ipynb"))
    if not notebooks:
        die(f"no notebooks in {NOTEBOOKS}")

    for path in notebooks:
        slug, images = convert(path, exporter)
        size = (OUT / slug / "index.md").stat().st_size
        print(f"  {slug:<38} {size // 1024:>4} KB markdown, {images} images")

    print(f"\n{len(notebooks)} posts written to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
