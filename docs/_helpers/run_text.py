"""Sentences about a run, written from its own outputs, for the gallery pages to glue.

A gallery page's run can differ in its last bits from one machine to another, and an
accept decision that the rounding moves changes the rest of a pair. So what a page says
about the warnings of its run is built here from the run, not written by hand.
"""

import numpy as np

from telescopic import Result, diagnostics

# The heading of each warning on the page Warnings and errors.
ANCHORS = {
    "censored": "censored-pairs-censored",
    "late": "late-pairs-late",
    "variance": "one-replicate-dominating-the-variance-variance",
    "tail": "heavy-tailed-meeting-times-tail",
    "tuning": "disagreeing-tuning-chains-tuning",
    "slow": "slow-pairs-slow",
}


def warning_links(kinds: list[str]) -> str:
    """The kinds of warnings as links to their entries, joined as a list in prose."""
    # Glued Markdown is not resolved as a cross-reference, so the links name the built
    # page.
    links = [f"[`{kind}`](../reference/warnings.html#{ANCHORS[kind]})" for kind in kinds]
    if len(links) <= 2:
        return " and ".join(links)
    return ", ".join(links[:-1]) + " and " + links[-1]


def warnings_text(result: Result) -> str:
    """The warnings the run emits, and the replicate with the largest share in a variance.

    The share's replicate is named with the time its pair met and whether that time
    leaves a bias correction in its estimate.
    """
    kinds = [issue.kind for issue in diagnostics.issues(result)]
    checks = result.diagnostics()
    if not kinds:
        text = "The run emits no warning."
    elif len(kinds) == 1:
        text = f"The run emits one warning, {warning_links(kinds)}."
    else:
        text = f"The run emits {len(kinds)} warnings: {warning_links(kinds)}."
    replicate = checks["variance_share_replicate"]
    if replicate is None:
        return text
    position = int(np.flatnonzero(result.index == replicate)[0])
    tau = result.meeting_times[position]
    later = int(np.sum(result.meeting_times > tau))
    # A censored pair has the meeting time inf, so it counts as later.
    rank = (
        "the latest of the run"
        if later == 0
        else f"with {later} of the {len(result.meeting_times)} pairs meeting later or not at all"
    )
    k_lag = result.k + result.lag
    correction = (
        f"after $k + L$ = {k_lag:,}, so its estimate carries a bias correction"
        if tau > k_lag
        else f"by $k + L$ = {k_lag:,}, so its estimate carries no bias correction"
    )
    text += (
        f" The largest share of one replicate in the variance of an estimate is"
        f" {checks['variance_share']:.2f}, for {checks['variance_share_of']}, held by"
        f" replicate {replicate}, whose pair met at $\\tau$ = {tau:,.0f}, {rank}, {correction}."
    )
    return text
