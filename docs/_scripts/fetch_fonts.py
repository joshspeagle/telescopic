"""Download the site's fonts and their licenses from the pinned Fontsource packages on npm.

Run when ``brand.FONTS`` changes: ``python docs/_scripts/fetch_fonts.py``. Each package's
tarball must have the SHA-256 recorded in ``brand.FONTS``. The Latin woff2 files of the
faces listed there, and each family's SIL Open Font License, are written to the theme's
``static/fonts/`` and committed, so the site loads no font from another host.
``brand.py`` writes the ``@font-face`` rules that serve them.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
import urllib.request

from brand import FONT_DIR, FONTS, face_file

REGISTRY = "https://registry.npmjs.org/@fontsource/{0}/-/{0}-{1}.tgz"


def main() -> None:
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    for old in FONT_DIR.glob("*.woff2"):  # faces no longer in FONTS go too
        old.unlink()
    for family, package, version, sha256, faces, _ in FONTS.values():
        url = REGISTRY.format(package, version)
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != sha256:
            raise SystemExit(f"{url} does not have the SHA-256 recorded for {family}.")
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for weight, style in faces:
                name = face_file(package, weight, style)
                (FONT_DIR / name).write_bytes(tar.extractfile(f"package/files/{name}").read())
            lines = tar.extractfile("package/LICENSE").read().decode("utf-8").splitlines()
        license_text = "\n".join(line.rstrip() for line in lines) + "\n"
        (FONT_DIR / f"OFL-{package}.txt").write_text(license_text, encoding="utf-8")
    print("\n".join(sorted(p.name for p in FONT_DIR.iterdir())))


if __name__ == "__main__":
    main()
