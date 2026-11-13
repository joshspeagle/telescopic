"""How wide does the summary print, against the column of the documentation (issue #85)?

The summary of `sample()` on `examples.gaussian(d)` with its defaults and `rng=0`, at
d = 2 and d = 10: the length of its longest line, and that line. A code output of the
documentation holds 83 characters at 1440 px: the article is 46rem = 736 px wide, an
output's text box 736 - 2 x 16 px of padding - 2 x 1 px of border = 702 px, and Red Hat
Mono at 14 px advances 0.6 em = 8.4 px per character, so 702 / 8.4 = 83.6. Chromium,
on the built Getting started page, gives the same 702 px and 8.4 px.

Result: 130 and 131 characters at d = 2 and d = 10 at 5313efd, where the third line of
the footer held the cost, the relative cost and the late fraction
(`i85_summary_width_before.out`); with the summary of DECISIONS #134, 75 and 76
(`i85_summary_width.out`): the cost line at d = 2 and the table's header at d = 10. The
cost line prints the seconds of the run, so its length moves by a character between
runs.

Run: python notes/experiments/i85_summary_width.py
"""

import subprocess
from pathlib import Path

import telescopic

commit = subprocess.run(
    ["git", "describe", "--always", "--dirty"],
    capture_output=True,
    text=True,
    check=True,
    cwd=Path(telescopic.__file__).parent,
).stdout.strip()
print(f"commit {commit} of the package")
for d in (2, 10):
    example = telescopic.examples.gaussian(d)
    result = telescopic.sample(example.log_prob, example.initial, vectorize=True, rng=0)
    lines = repr(result.summary()).splitlines()
    longest = max(lines, key=len)
    print(f"d = {d}: {len(longest)} characters in the longest line:\n  {longest}")
