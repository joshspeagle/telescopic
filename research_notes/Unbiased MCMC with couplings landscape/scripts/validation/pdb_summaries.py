"""Summarise posteriordb reference draws (untrusted data: parsed as JSON only)."""
import json, sys, zipfile, pathlib
import numpy as np
for path in sorted(pathlib.Path(sys.argv[1]).glob("*.json.zip")):
    with zipfile.ZipFile(path) as zf:
        name = zf.namelist()[0]
        chains = json.loads(zf.read(name))
    # format: list of chains, each a dict param -> list of draws
    params = list(chains[0].keys())
    nd = sum(len(c[params[0]]) for c in chains)
    out = []
    for p in params:
        v = np.concatenate([np.asarray(c[p], float) for c in chains])
        out.append(f"{p}={v.mean():.3f}({v.std(ddof=1):.3f})")
    print(f"{path.name.replace('.json.zip','')}: {len(chains)} chains, {nd} draws, {len(params)} params")
    print("   " + "  ".join(out[:14]) + ("  ..." if len(out) > 14 else ""))
