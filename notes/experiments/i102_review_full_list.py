"""The adversarial review: default runs for listed jobs of i102_review_runner.py.

Used to rerun with their default runs the jobs on which the rule fired in a "rule"
invocation, to see whether their estimates were right. Arguments are "<cell>@<seed>,<seed>".

Outputs:
  i102_review_pumps_full1.out   "ex pumps default@2002,2015"
  i102_review_pumps_full2.out   "ex pumps default@2206,2491,2625,2863,2980,2981"
  i102_review_g50b_full.out     "gauss d=50 cond=10000 tight@2000,2001,...,2009"
Every run in them has a right estimate, with |z| at most 3.06. CAP was raised for these
runs, which took up to 239 s.

Ran against commit c961089.

Run: OUT=<folder for the rows> CAP=<seconds> \
     python notes/experiments/i102_review_full_list.py "<cell>@<seed>,<seed>"
"""

import sys
from multiprocessing import Pool

import i102_review_runner as runner

if __name__ == "__main__":
    jobs = []
    for spec in sys.argv[1:]:
        name, seeds = spec.split("@")
        jobs += [(name, int(s)) for s in seeds.split(",")]
    with Pool(2, maxtasksperchild=4) as pool:
        for r in pool.imap(runner.job, jobs, chunksize=1):
            print(runner.show(r), "| warn:", r["n_warn"], r["warn_text"], flush=True)
