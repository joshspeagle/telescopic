"""Review of x123: is the registered table of predictions the output of the prediction model?

Question: does every rate of the table of PREREGISTRATION.md, section 5, equal the rate
that predictions.out prints for the same family?

Design: reads the two files of the folder above (it runs nothing and reads no seed). A
family of the model "point" has only check rates in predictions.out, and its line rates
are "-" in the table; for every other family the four rates are compared as printed.

Run: python notes/experiments/x123/review/predtable.py > predtable.out

Result: see predtable.out and REVIEW.md, the checks that found nothing: 62 rows, no
mismatch.
"""

import re
from pathlib import Path

X = Path(__file__).resolve().parent.parent
table = re.findall(
    r"^\| ([BCF]) \| (.*?) \| (\w+) \| (\S+) \| (\S+) \| (\S+) \| (\S+) \|$",
    (X / "PREREGISTRATION.md").read_text(),
    re.MULTILINE,
)
printed = {}
for line in (X / "predictions.out").read_text().splitlines():
    m = re.match(r"\s*\d+ ([BCF])\s+(.*?)\s+(\w+)\s+\|(.*)\|(.*)$", line)
    if m:
        printed[(m[1], m[2])] = (m[3], m[4].split(), m[5].split())
mismatches = 0
for group, name, model, n_line, n_check, h_line, h_check in table:
    got_model, n, h = printed[(group, name)]
    if model == "point":
        same = (n_line, h_line) == ("-", "-") and (n_check, h_check) == (n[0], h[0])
    else:
        same = (n_line, n_check, h_line, h_check) == (n[0], n[1], h[0], h[1])
    if not same or model != got_model:
        mismatches += 1
        print("MISMATCH", group, name, (n_line, n_check, h_line, h_check), n, h)
print(f"rows of the table {len(table)}, families in predictions.out {len(printed)}")
print(f"mismatches {mismatches}")
