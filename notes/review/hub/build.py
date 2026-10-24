"""Build review-data.json for the review hub from the repository at HEAD.

Run from the repository root: python notes/review/hub/build.py. Links point at the
commit HEAD, which must be pushed for them to resolve.
"""

import ast
import json
import re
import runpy
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HUB = Path(__file__).resolve().parent
STEPS = runpy.run_path(str(HUB / "steps.py"))
DEPTHS, LEVELS, PHASES = STEPS["DEPTHS"], STEPS["LEVELS"], STEPS["PHASES"]

REPO = "https://github.com/joshspeagle/telescopic"
EXCERPT_LINES = 70


def git(*args):
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


SHA = git("rev-parse", "HEAD")


def link(path, line=None, end=None):
    url = f"{REPO}/blob/{SHA}/{path}"
    if line:
        url += f"#L{line}" + (f"-L{end}" if end and end != line else "")
    return url


def symbol(ref):
    # "path:Class.method" -> location and source excerpt.
    path, _, name = ref.partition(":")
    source = (ROOT / path).read_text()
    if not name:
        lines = source.splitlines()
        return {
            "ref": ref,
            "path": path,
            "name": "",
            "line": 1,
            "end": len(lines),
            "url": link(path),
            "source": "\n".join(lines[:40]),
            "truncated": len(lines) > 40,
        }
    tree = ast.parse(source)
    node = None
    scope = tree.body
    for part in name.split("."):
        node = next(
            n for n in scope if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part
        )
        scope = getattr(node, "body", [])
    start = node.lineno - len(node.decorator_list)
    lines = source.splitlines()[start - 1 : node.end_lineno]
    return {
        "ref": ref,
        "path": path,
        "name": name,
        "line": start,
        "end": node.end_lineno,
        "url": link(path, start, node.end_lineno),
        "source": "\n".join(lines[:EXCERPT_LINES]),
        "truncated": len(lines) > EXCERPT_LINES,
    }


def sections(path, pattern):
    # Headings matching pattern (a regex with groups number and title) and their text.
    out, current = [], None
    for raw in (ROOT / path).read_text().splitlines():
        m = re.match(pattern, raw)
        if m:
            current = {"num": m.group(1), "title": m.group(2).strip(), "lines": []}
            out.append(current)
        elif current is not None:
            current["lines"].append(raw)
    for item in out:
        item["text"] = "\n".join(item.pop("lines")).strip()
    return out


def table_rows(text, first):
    # Rows of a markdown table whose first cell matches `first`, split on unescaped pipes.
    rows = []
    for line in text.splitlines():
        if not re.match(rf"^\| {first} \|", line):
            continue
        cells = [
            c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))
        ]
        rows.append(cells)
    return rows


def decisions():
    rows = table_rows((ROOT / "notes/DECISIONS.md").read_text(), r"\d+")
    out = []
    for cells in rows:
        number, date, topic, decision, reason, alternatives, outcome = cells[:7]
        out.append(
            {
                "id": int(number),
                "date": date,
                "topic": topic,
                "decision": decision,
                "reason": reason,
                "alternatives": alternatives,
                "outcome": outcome,
            }
        )
    return out


def spec_decisions():
    rows = table_rows((ROOT / "docs/design/spec.md").read_text(), r"D-\d+")
    return [
        {"id": c[0], "decision": c[1], "reason": c[2], "alternatives": c[3] if len(c) > 3 else ""}
        for c in rows
    ]


def questions():
    text = (ROOT / "notes/QUESTIONS.md").read_text()
    items = re.split(r"\n(?=- \[[ x]\] )", text)
    out = []
    for item in items:
        m = re.match(r"- \[([ x])\] \*\*(.+?)\*\*\s*(.*)", item, re.DOTALL)
        if m:
            out.append(
                {
                    "done": m.group(1) == "x",
                    "title": m.group(2).rstrip("."),
                    "text": re.sub(r"\s*\n\s*", " ", m.group(3)).strip(),
                }
            )
    return out


def experiments(decision_rows, phases):
    folder = ROOT / "notes/experiments"
    out = []
    outs = {p.stem: p for p in folder.glob("*.out")}
    for script in sorted(folder.glob("*.py")):
        tree = ast.parse(script.read_text())
        doc = ast.get_docstring(tree) or ""
        outputs = []
        for stem, path in sorted(outs.items()):
            if stem == script.stem or (
                stem not in {p.stem for p in folder.glob("*.py")} and f"`{path.name}`" in doc
            ):
                text = path.read_text()
                outputs.append(
                    {
                        "name": path.name,
                        "url": link(f"notes/experiments/{path.name}"),
                        "text": text[:6000],
                        "truncated": len(text) > 6000,
                    }
                )
        names = {script.stem} | {o["name"] for o in outputs}
        cited = [
            d["id"]
            for d in decision_rows
            if any(n in " ".join((d["reason"], d["decision"], d["alternatives"])) for n in names)
        ]
        steps = [
            s["id"] for p in phases for s in p["steps"] if script.stem in s.get("experiments", [])
        ]
        out.append(
            {
                "name": script.stem,
                "url": link(f"notes/experiments/{script.name}"),
                "doc": doc,
                "outputs": outputs,
                "decisions": cited,
                "steps": steps,
            }
        )
    return out


def reviews():
    out = []
    for path in sorted((ROOT / "notes/reviews").glob("*.md")):
        text = path.read_text()
        title = next(
            (line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("#")),
            path.stem,
        )
        out.append(
            {
                "name": path.name,
                "title": title,
                "url": link(f"notes/reviews/{path.name}"),
                "lines": len(text.splitlines()),
            }
        )
    return out


def modules():
    out = []
    for path in sorted((ROOT / "src/telescopic").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        source = path.read_text()
        tree = ast.parse(source)
        public = [
            n.name
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not n.name.startswith("_")
        ]
        test = ROOT / "tests" / f"test_{path.stem.lstrip('_')}.py"
        out.append(
            {
                "path": rel,
                "url": link(rel),
                "lines": len(source.splitlines()),
                "doc": (ast.get_docstring(tree) or "").split("\n")[0],
                "public": public,
                "requirements": sorted(set(re.findall(r"R-[A-Z]+(?:-[A-Z0-9]+)?", source))),
                "test": test.relative_to(ROOT).as_posix() if test.exists() else None,
            }
        )
    return out


def counts():
    tests = sum(
        len(re.findall(r"^\s*def test_", p.read_text(), re.MULTILINE))
        for p in (ROOT / "tests").glob("*.py")
    )
    src = sum(len(p.read_text().splitlines()) for p in (ROOT / "src/telescopic").rglob("*.py"))
    return {"src_lines": src, "test_functions": tests}


def main():
    decision_rows = decisions()
    spec = sections(
        "docs/design/spec.md", r"^#{2,3} ((?:\d+(?:\.\d+)?)|Appendix [A-Z])\.?:? (.+)$"
    )
    for item in spec:
        item["num"] = item["num"].replace("Appendix ", "")
        item["requirements"] = sorted(
            set(re.findall(r"R-[A-Z]+(?:-[A-Z0-9]+)?", item["title"] + item["text"]))
        )
        item["url"] = link("docs/design/spec.md")
    theory = sections("docs/theory.md", r"^## ()(.+)$")
    phases = []
    for phase in PHASES:
        steps = []
        for step in phase["steps"]:
            step = dict(step)
            step["code"] = [symbol(ref) for ref in step.get("code", [])]
            step["docs"] = [{"path": p, "url": link(p)} for p in step.get("docs", [])]
            step["tests"] = [{"path": p, "url": link(p)} for p in step.get("tests", [])]
            steps.append(step)
        phases.append({**phase, "steps": steps})
    audit = json.loads((HUB / "audit2.json").read_text()) if (HUB / "audit2.json").exists() else []
    data = {
        "meta": {
            "sha": SHA,
            "short": SHA[:7],
            "repo": REPO,
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "date": git("log", "-1", "--format=%cs"),
            **counts(),
        },
        "depths": DEPTHS,
        "levels": LEVELS,
        "phases": phases,
        "spec": spec,
        "theory": theory,
        "decisions": decision_rows,
        "specDecisions": spec_decisions(),
        "questions": questions(),
        "experiments": experiments(decision_rows, PHASES),
        "reviews": reviews(),
        "modules": modules(),
        "audit": audit,
    }
    (HUB / "review-data.json").write_text(json.dumps(data, ensure_ascii=False))
    print(
        f"review-data.json: {len(json.dumps(data)) / 1e3:.0f} kB, {len(phases)} phases, "
        f"{sum(len(p['steps']) for p in phases)} steps, {len(spec)} spec sections, "
        f"{len(decision_rows)} decisions, {len(data['experiments'])} experiments"
    )


if __name__ == "__main__":
    main()
