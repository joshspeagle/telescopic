"""Template context and the hero directive of the telescopic docs theme.

* The sections of the top bar are the captioned toctrees of the root document; a page's
  section is the one whose toctree lists it, or lists the page whose toctree lists it, as
  the API reference lists its pages. The sidebar lists that section's pages and marks the
  one that holds the page.
* A page with two or more sections, or two or more entries below its one object, as a
  class's page lists its methods, gets the "On this page" outline.
* A page whose front matter says ``layout: home`` uses ``home.html``.
* ``{hero}`` inlines the landing page's coupled pair (``_themes/telescopic/hero.svg``,
  written by ``_scripts/brand.py``) as a figure, with its one paragraph of content as
  caption. A page has at most one hero, since the drawing carries fixed ids.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from docutils import nodes
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective

logger = logging.getLogger(__name__)
HERO = Path(__file__).parents[1] / "_themes" / "telescopic" / "hero.svg"


class Hero(SphinxDirective):
    has_content = True

    def run(self) -> list[nodes.Node]:
        self.env.note_dependency(str(HERO))
        caption = nodes.Element()
        self.state.nested_parse(self.content, self.content_offset, caption)
        if len(caption) != 1 or not isinstance(caption[0], nodes.paragraph):
            raise self.error("the hero directive expects one paragraph, its caption, as content")
        figure = nodes.figure(classes=["hero"])
        figure += nodes.raw("", HERO.read_text(encoding="utf-8"), format="html")
        figure += nodes.caption("", "", *caption[0].children)
        return [figure]


def _sections(app: Sphinx, env: BuildEnvironment) -> set[str]:
    root = env.get_doctree(app.config.root_doc)
    sections = []
    for toctree in root.findall(addnodes.toctree):
        pages = [
            {"docname": doc, "title": title or env.titles[doc].astext()}
            for title, doc in toctree["entries"]
            if doc in env.titles
        ]
        if toctree.get("caption") and pages:
            sections.append({"caption": toctree["caption"], "pages": pages})
    if not sections:
        logger.warning("the root document has no captioned toctree, so the top bar is empty")
    # Every page carries the navigation, so a change to it rewrites every page, not only
    # those that were read again.
    changed = sections != getattr(env, "telescopic_sections", None)
    env.telescopic_sections = sections
    return set(env.found_docs) if changed else set()


def _section_of(sections: list[dict[str, Any]], docname: str) -> int | None:
    return next(
        (
            i
            for i, section in enumerate(sections)
            if any(page["docname"] == docname for page in section["pages"])
        ),
        None,
    )


def _page_context(
    app: Sphinx, pagename: str, templatename: str, context: dict[str, Any], doctree: Any
) -> str | None:
    sections = context["sections"] = app.env.telescopic_sections
    section_page = pagename
    parents = {
        child: parent
        for parent, children in app.env.toctree_includes.items()
        for child in children
    }
    while section_page in parents and _section_of(sections, section_page) is None:
        section_page = parents[section_page]
    context["section_page"] = section_page
    context["section_index"] = _section_of(sections, section_page)
    toc = app.env.tocs.get(pagename)
    headings = toc.next_node(nodes.bullet_list) if toc else None
    if headings is not None and len(headings) == 1:
        headings = headings[0].next_node(nodes.bullet_list)
    context["outline"] = headings is not None and len(headings) > 1
    if (context.get("meta") or {}).get("layout") == "home":
        return "home.html"
    return None


def setup(app: Sphinx) -> dict[str, bool]:
    app.add_directive("hero", Hero)
    # Last, so that the theme wins over the extensions' style sheets at equal specificity.
    app.add_css_file("telescopic.css", priority=900)
    app.add_js_file("telescopic.js", loading_method="defer")
    app.connect("env-updated", _sections)
    app.connect("html-page-context", _page_context)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
