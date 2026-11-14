import itertools
import re
import runpy
import tempfile
import warnings
from importlib import resources
from pathlib import Path

import numpy as np
import pytest

import telescopic
from references import Counted, gaussian_log_prob, synthetic
from references import Normal as Conditional
from telescopic import CoupledSampler, __version__, sample
from telescopic._parallel import Interrupt
from telescopic.engine import Settings, Status
from telescopic.initial import Empirical, Normal
from telescopic.kernels import HMC, MALA, Gibbs, Mixture, RandomWalk
from telescopic.sampler import _SlowPairs
from telescopic.warnings import ConvergenceWarning, InputWarning

INIT = Normal([2.0, -1.0], [4.0, 4.0])
KERNEL = RandomWalk(np.eye(2))


log_prob = gaussian_log_prob


def test_a_run_tunes_runs_the_pilot_and_records_both():
    result = sample(log_prob, INIT, 20, rng=1)
    assert result.tuning is not None
    assert result.kernel.is_tuned
    assert len(result.pilot) == 100
    assert (result.lag, result.m) == (result.k, 10 * result.k)
    np.testing.assert_array_equal(result.index, np.arange(20))


def test_two_runs_of_a_sampler_equal_one_run_of_both():
    # Point by point, so each replicate is bitwise independent of the batch.
    sampler = CoupledSampler(log_prob, INIT, rng=2)
    sampler.run(10)
    split = sampler.run(15)
    whole = CoupledSampler(log_prob, INIT, rng=2).run(25)
    np.testing.assert_array_equal(split.meeting_times, whole.meeting_times)
    np.testing.assert_array_equal(split.measure.atoms, whole.measure.atoms)


def test_a_sampler_keeps_its_settings():
    sampler = CoupledSampler(log_prob, INIT, kernel=KERNEL, rng=3)
    assert sampler.settings is None
    sampler.run(5, k=40, thin=2)
    assert sampler.run(5).thin == 2
    for change, message in (({"k": 41}, "k was 40"), ({"thin": 1}, "thin was 2")):
        with pytest.raises(ValueError, match=message):
            sampler.run(5, **change)
    with pytest.raises(AttributeError):
        sampler.settings = sampler.settings._replace(lag=2)


def unevaluated(x):
    raise AssertionError("the log-density was evaluated")


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"n_replicates": 10.0}, TypeError, "n_replicates must be an integer, not 10.0"),
        ({"n_replicates": True}, TypeError, "n_replicates must be an integer, not True"),
        ({"k": 5.5}, TypeError, "k must be an integer or 'auto', not 5.5"),
        ({"k": np.array([3, 4])}, TypeError, "k must be an integer or 'auto'"),
        ({"thin": 2.0}, TypeError, "thin must be an integer or None, not 2.0"),
        ({"max_iter": "9"}, TypeError, "max_iter must be an integer or None"),
        ({"lag": 0}, ValueError, "lag must be at least 1, not 0"),
        ({"thin": 0}, ValueError, "thin must be at least 1, not 0"),
        ({"m": -1}, ValueError, "m must be at least 0, not -1"),
        ({"max_iter": 1}, ValueError, r"max_iter must exceed max\(lag, m\) = 1, not 1"),
        ({"k": -1}, ValueError, "k must be at least 0, not -1"),
        ({"k": 5, "m": 3}, ValueError, "m must be at least k = 5, not 3"),
        ({"k": 40, "max_iter": 100}, ValueError, "max_iter must exceed"),
    ],
)
def test_invalid_settings_raise_before_tuning(arguments, error, message):
    sampler = CoupledSampler(unevaluated, INIT, rng=3)
    with pytest.raises(error, match=message):
        sampler.run(**{"n_replicates": 5} | arguments)


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"n": 2.5}, TypeError, "n must be an integer, not 2.5"),
        ({"max_iter": 1}, ValueError, "max_iter must exceed"),
    ],
)
def test_invalid_pilot_settings_raise_before_tuning(arguments, error, message):
    sampler = CoupledSampler(unevaluated, INIT, rng=3)
    with pytest.raises(error, match=message):
        sampler.meeting_times(**arguments)


def test_invalid_first_settings_are_not_kept():
    sampler = CoupledSampler(log_prob, INIT, kernel=KERNEL, rng=3)
    with pytest.raises(ValueError, match="m must be at least k = 40"):
        sampler.run(5, k=40, m=5)
    assert sampler.run(5, k=40, m=50).m == 50


@pytest.mark.parametrize(
    ("given", "expected"),
    [({"k": 40}, (40, 40, 400)), ({"k": 40, "m": 50, "lag": 2}, (40, 2, 50))],
)
def test_an_explicit_k_needs_no_pilot_and_sets_the_lag_and_m(given, expected):
    sampler = CoupledSampler(log_prob, INIT, kernel=KERNEL, rng=4)
    result = sampler.run(5, **given)
    assert (result.k, result.lag, result.m) == expected
    assert sampler.pilot is None


@pytest.mark.parametrize(
    "init",
    [lambda rng: rng.normal(size=2), np.random.default_rng(5).standard_normal((6, 2))],
    ids=["callable", "points"],
)
def test_initial_distributions_without_rvs_run(init):
    assert sample(log_prob, init, 4, kernel=KERNEL, k=40, rng=6).n_replicates == 4


def test_a_seed_reproduces_a_run_and_a_generator_moves_on():
    first, second = (sample(log_prob, INIT, 5, kernel=KERNEL, k=40, rng=7) for _ in range(2))
    np.testing.assert_array_equal(first.measure.atoms, second.measure.atoms)
    rng = np.random.default_rng(8)
    third, fourth = (sample(log_prob, INIT, 5, kernel=KERNEL, k=40, rng=rng) for _ in range(2))
    assert not np.array_equal(third.measure.atoms, fourth.measure.atoms)


def test_the_pilot_draws_from_streams_of_its_own():
    # Sharing the main run's streams, a lag-1 run would repeat its meeting times (R-TUNE-3).
    sampler = CoupledSampler(log_prob, INIT, kernel=KERNEL, rng=3)
    pilot = sampler.meeting_times(10)
    with pytest.warns(ConvergenceWarning, match="k may be too small"):
        result = sampler.run(10, k=1, m=2, lag=1)
    assert not np.array_equal(pilot, result.meeting_times)


def test_a_run_completes_a_pilot_of_fewer_than_100_pairs():
    sampler = CoupledSampler(log_prob, INIT, kernel=KERNEL, rng=3)
    sampler.meeting_times(30)
    result = sampler.run(20)
    whole = CoupledSampler(log_prob, INIT, kernel=KERNEL, rng=3).meeting_times()
    np.testing.assert_array_equal(result.pilot, whole)


def test_an_explicit_k_of_zero_gets_a_lag_of_one():
    sampler = CoupledSampler(log_prob, INIT, kernel=KERNEL, rng=4)
    with pytest.warns(ConvergenceWarning, match="met after k \\+ lag"):  # with k = 0, all do
        sampler.run(2, k=0)
    assert (sampler.settings.lag, sampler.settings.m) == (1, 0)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: CoupledSampler(log_prob, INIT, kernel=KERNEL).meeting_times(0), "n must be"),
        (
            lambda: CoupledSampler(log_prob, INIT, kernel=RandomWalk(np.eye(3))),
            "the kernel works in 3 dimensions, but the initial points have 2",
        ),
    ],
)
def test_invalid_arguments_raise_before_any_run(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def test_sample_refuses_fewer_than_two_replicates():
    # One replicate was tuned and run, and its result could give no estimate.
    def never(x):
        raise AssertionError("sample() did work before checking n_replicates")

    with pytest.raises(ValueError, match="n_replicates must be at least 2, not 1"):
        sample(never, INIT, 1)


def test_unknown_kernel_names_are_listed():
    with pytest.raises(ValueError, match="'auto', 'rwmh', 'mala', 'hmc', 'independence'"):
        CoupledSampler(log_prob, INIT, kernel="nuts")
    with pytest.raises(TypeError, match="kernel must be a Kernel"):
        CoupledSampler(log_prob, INIT, kernel=RandomWalk)


@pytest.mark.parametrize(("kernel", "kind"), [("mala", MALA), ("hmc", HMC)])
def test_gradient_kernels_are_tuned_by_name(kernel, kind):
    result = sample(log_prob, INIT, 5, grad=lambda x: -x, kernel=kernel, k=100, rng=13)
    assert isinstance(result.kernel, kind)
    assert result.kernel.is_tuned
    assert result.n_replicates == 5


def conditional(point, block):
    return Conditional(0.8 * point[1 - block], 0.6)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"kernel": "mala"}, "needs the gradient"),
        ({"log_prob": None}, "needs a log-density"),
        (
            {
                "log_prob": None,
                "kernel": Mixture(
                    [Gibbs(conditional, [[0], [1]]), RandomWalk(np.eye(2))], [0.5, 0.5]
                ),
            },
            "needs a log-density",
        ),
    ],
)
def test_kernels_without_what_they_need_raise(arguments, message):
    with pytest.raises(ValueError, match=message):
        CoupledSampler(**({"log_prob": log_prob, "init": INIT} | arguments))


@pytest.mark.parametrize(
    "kernel",
    [
        Gibbs(conditional, [[0], [1]]),
        Mixture(
            [Gibbs(conditional, [[0], [1]]), Gibbs(conditional, [[0], [1]], coupling="crn")],
            [0.5, 0.5],
        ),
    ],
    ids=["gibbs", "mixture"],
)
def test_gibbs_samplers_and_their_mixtures_run_without_a_log_density(kernel):
    assert sample(None, INIT, 5, kernel=kernel, k=20, rng=14).n_replicates == 5


def test_a_slow_pair_warns_once_when_unmet_at_the_threshold():
    # A pair still unmet at t = 10^4 meets after it, as diagnostics.issues counts slow.
    slow = _SlowPairs(lag=1, m=10, kind="replicate")
    slow(Status(0, 3, 9_999))
    with pytest.warns(ConvergenceWarning, match="replicate 3 has not met after 10000"):
        slow(Status(0, 3, 10_000))
    slow(Status(0, 3, 10_001))  # a second warning would fail under -W error


def test_a_slow_pair_seen_as_it_happens_says_that_the_run_waits_for_it():
    with pytest.warns(ConvergenceWarning, match="pilot pair 3 .* the run waits for it; the"):
        _SlowPairs(lag=1, m=10, kind="pilot pair")(Status(0, 3, 10_000))


def test_a_slow_replicate_names_ctrl_c_only_while_its_handler_listens():
    interrupt = Interrupt()
    clause = "waits for it: a first Ctrl-C starts no new replicate but lets this one finish"
    with interrupt.catching(), pytest.warns(ConvergenceWarning, match=clause):
        _SlowPairs(1, 10, "replicate", interrupt)(Status(0, 3, 10_000))
    # Outside its block, as in a thread or under a host that keeps its own handler, and
    # after a first Ctrl-C, the clause would not be true.
    with pytest.warns(ConvergenceWarning) as caught:
        _SlowPairs(1, 10, "replicate", interrupt)(Status(0, 3, 10_000))
    interrupt.stopping = True
    with interrupt.catching(), pytest.warns(ConvergenceWarning) as stopped:
        _SlowPairs(1, 10, "replicate", interrupt)(Status(0, 3, 10_000))
    assert "Ctrl-C" not in str(caught[0].message) + str(stopped[0].message)


def test_a_pair_censored_at_the_threshold_warns_as_slow_after_a_run():
    # The engine frees a censored pair's lane before its status is read, so only the
    # check after the run sees a pair censored at t = 10^4.
    replicates = synthetic([2, -1], [0.0], m=10, max_iter=10_000)._replicates
    with pytest.warns(ConvergenceWarning, match="replicate 1 has not met after 10000"):
        _SlowPairs(lag=1, m=10, kind="replicate").after(replicates)


def test_tuning_chains_in_separate_modes_warn():
    starts = itertools.cycle([[-20.0], [20.0]])

    def two_modes(x):
        return np.logaddexp(-0.5 * (x[0] + 20) ** 2, -0.5 * (x[0] - 20) ** 2)

    sampler = CoupledSampler(two_modes, lambda rng: next(starts), rng=12)
    with pytest.warns(ConvergenceWarning, match="or the chains may not have converged"):
        sampler.tune()


def test_a_run_with_censored_pairs_warns():
    kernel = RandomWalk(np.eye(2), scale=0.1)
    with pytest.warns(ConvergenceWarning) as record:  # censored pairs are also late
        sample(log_prob, INIT, 20, kernel=kernel, k=1, m=2, max_iter=3, rng=17)
    assert any("reached max_iter = 3" in str(warning.message) for warning in record)


def half_normal(x):
    return -0.5 * x[0] ** 2 if x[0] > 0 else -np.inf


@pytest.mark.parametrize(
    ("call", "kind"),
    [
        (lambda sampler: sampler.run(6, k=40), "replicates"),
        (lambda sampler: sampler.meeting_times(6), "pilot pairs"),
    ],
    ids=["replicates", "pilot"],
)
def test_starts_outside_the_support_warn(call, kind):
    sampler = CoupledSampler(
        half_normal, Empirical([[-1.0], [1.0]]), kernel=RandomWalk(np.eye(1)), rng=9
    )
    with pytest.warns(InputWarning, match=f"{kind} start where the target density is zero"):
        call(sampler)


def test_a_zero_density_start_warns_before_the_run_ends():
    # The log-density counts its calls, and the warning records the count when it comes.
    calls = []

    def counting(x):
        calls.append(1)
        return half_normal(x)

    seen = []
    sampler = CoupledSampler(
        counting, Empirical([[-1.0], [1.0]]), kernel=RandomWalk(np.eye(1)), rng=9
    )
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = lambda message, *_, **__: seen.append((str(message), len(calls)))
        sampler.run(6, k=40)
    outside = [n for message, n in seen if "start where the target density is zero" in message]
    assert outside
    assert outside[0] < len(calls)


def test_choose_k_failure_names_zero_density_starts():
    # From -50 a random-walk chain on the half-normal never proposes a point of positive
    # density, so the pairs that start there never meet within the pilot's cap.
    sampler = CoupledSampler(
        half_normal, Empirical([[-50.0], [1.0]]), kernel=RandomWalk(np.eye(1)), rng=9
    )
    with pytest.warns(InputWarning):
        sampler.meeting_times(100, max_iter=20)
    with pytest.raises(ValueError, match=r"pilot pairs never met .* of the 100 started where"):
        sampler.run(2)


def slow_pair_from_zero_density_as_it_happens(slow):
    slow(Status(0, 1, 10_000, 1, 1, True))


def slow_pair_from_zero_density_after_a_run(slow):
    replicates = synthetic([2, 20_000], [0.0, 1.0])._replicates
    replicates.zero_density = np.array([1])
    slow.after(replicates)


@pytest.mark.parametrize(
    "seen",
    [slow_pair_from_zero_density_as_it_happens, slow_pair_from_zero_density_after_a_run],
    ids=["as it happens", "after a run"],
)
def test_a_slow_pair_that_started_at_zero_density_says_so(seen):
    with pytest.warns(ConvergenceWarning, match="replicate 1 .* it started where the target"):
        seen(_SlowPairs(lag=1, m=10, kind="replicate"))


def test_a_run_that_would_store_gigabytes_warns():
    # Without building the averaging set, whose 10^18 times would not fit in memory.
    sampler = CoupledSampler(log_prob, INIT, kernel=KERNEL, rng=10)
    with pytest.warns(InputWarning, match="GB of atoms"):
        sampler._check_memory(Settings(0, 10**18, 1), 1)


def test_the_memory_warning_compares_the_peak_with_the_threshold():
    # A pilot pair meeting at 1001 stands for 1000 distances of 24 bytes and 2001 atoms
    # of 32 bytes at k = 0: 0.44 GB stored for 5,000 replicates, of which the atoms are
    # 0.32, and a peak of three times that, up to twice as much in two dimensions.
    sampler = CoupledSampler(log_prob, INIT, kernel=KERNEL, rng=10)
    sampler.pilot = np.full(100, 1001.0)
    with pytest.warns(InputWarning, match="store about 0.4 GB .* take about 1.3 to 2.6 GB"):
        sampler._check_memory(Settings(0, 0, 1), 5_000)


def test_the_readme_example_runs_without_a_warning(capsys):
    # Its code block as written; an earlier test ran the first draft's example, which
    # the README and the spec replaced (DECISIONS #35). Warnings are errors here.
    pytest.importorskip("matplotlib").use("Agg")
    text = (Path(__file__).parents[1] / "README.md").read_text()
    code = re.search(r"```python\n(.*?)```", text, re.DOTALL).group(1)
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        script = Path(directory) / "readme.py"
        script.write_text(code)
        runpy.run_path(str(script))
    pytest.importorskip("matplotlib.pyplot").close("all")
    assert "slope" in capsys.readouterr().out


def test_citation_version_matches_package_version():
    # The version is written in both files; the test group has no YAML parser.
    text = (Path(__file__).parents[1] / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r"^version:\s*[\"']?([^\"'\s]+)", text, re.MULTILINE)
    assert match, "CITATION.cff has no version line"
    assert match.group(1) == __version__


def test_the_package_exports_what_the_readme_uses():
    # A name left out of __all__ is left out of `from telescopic import *`.
    text = (Path(__file__).parents[1] / "README.md").read_text()
    code = re.search(r"```python\n(.*?)```", text, re.DOTALL).group(1)
    used = set(re.findall(r"\btelescopic\.(\w+)", code))
    assert used, "the README's example takes no name from telescopic"
    assert used | {"__version__"} <= set(telescopic.__all__)


def test_the_package_is_marked_as_typed():
    # PEP 561: without the marker, a user's type checker treats telescopic as untyped.
    assert (resources.files("telescopic") / "py.typed").is_file()


def test_chains_cannot_start_where_the_gradient_is_not_finite():
    # MALA chains that started there stayed at its edge: on a mixture whose gradient
    # underflows beyond 40, from N(45, 1), E[x] was 40.10 +- 0.03 (DECISIONS #67).
    def grad(x):
        return np.where(x > 1, np.nan, -x)

    # The gradient check, which skips such points, says first that it compared nothing.
    with (
        pytest.raises(ValueError, match="where log_prob is finite: chains cannot start"),
        pytest.warns(InputWarning, match="compared no line"),
    ):
        sample(
            log_prob, Normal([3.0], [0.01]), 10, grad=grad, kernel=MALA(1.0, cov=np.eye(1)), k=5
        )


def test_a_run_warns_when_it_first_takes_a_gradient_as_zero():
    # It warned after the run, so that a run whose pairs never met gave no sign: the count
    # in the warning is the run's so far.
    def grad(x):
        return np.where(x > 1, np.nan, -x)

    kernel = MALA(1.0, cov=np.eye(1))
    pattern = r"taken as 0 in \d+ of (\d+) evaluations of the replicates"
    with pytest.warns(InputWarning, match=pattern) as caught:
        result = sample(log_prob, Normal([0.0], [0.01]), 20, grad=grad, kernel=kernel, k=10, rng=1)
    message = next(str(w.message) for w in caught if "taken as 0" in str(w.message))
    evaluated = int(re.search(pattern, message).group(1))
    assert 0 < evaluated < result.n_grad
    assert result.n_grad_zeroed > 0


def test_a_sampler_with_a_gradient_kernel_checks_grad_unless_told_not_to():
    def wrong(x):
        return np.asarray(x)  # the sign of the standard normal's gradient flipped

    kernel = MALA(0.5, cov=np.eye(2))
    counted = Counted(log_prob), Counted(wrong)
    with pytest.warns(InputWarning, match="grad disagrees"):
        sampler = CoupledSampler(
            counted[0], INIT, grad=counted[1], vectorize=True, kernel=kernel, rng=7
        )
    assert sampler.gradient_check.ratio.max() > 1
    assert (counted[0].points, counted[1].points) == (40, 8)
    counted = Counted(log_prob), Counted(wrong)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sampler = CoupledSampler(
            counted[0],
            INIT,
            grad=counted[1],
            vectorize=True,
            kernel=kernel,
            check_grad=False,
            rng=7,
        )
    assert sampler.gradient_check is None
    assert (counted[0].points, counted[1].points) == (0, 0)


@pytest.mark.parametrize(
    "kernel",
    [MALA(), HMC(), MALA(0.5, cov=np.eye(2)), HMC(0.3, cov=np.eye(2))],
    ids=["MALA", "HMC", "MALA given", "HMC given"],
)
def test_the_gradient_check_changes_no_result(kernel):
    runs = [
        sample(
            log_prob,
            INIT,
            10,
            grad=lambda x: -x,
            vectorize=True,
            kernel=kernel,
            k=50,
            check_grad=check_grad,
            rng=np.random.default_rng(3),
        )
        for check_grad in (True, False)
    ]
    np.testing.assert_array_equal(runs[0].meeting_times, runs[1].meeting_times)
    np.testing.assert_array_equal(runs[0].measure.atoms, runs[1].measure.atoms)
    np.testing.assert_array_equal(runs[0].measure.weights, runs[1].measure.weights)


def test_the_gradient_check_runs_when_the_kernel_evaluates_grad():
    given = MALA(0.5, cov=np.eye(2))
    for kernel, vectorize in ((Mixture([KERNEL, given], [0.5, 0.5]), True), (given, False)):
        sampler = CoupledSampler(
            log_prob, INIT, grad=lambda x: -x, vectorize=vectorize, kernel=kernel, rng=8
        )
        assert len(sampler.gradient_check.gap) == 8
    assert CoupledSampler(log_prob, INIT, grad=lambda x: -x, kernel=KERNEL).gradient_check is None


def test_meeting_times_refuses_max_iter_none():
    # None ran an uncapped pilot, whose pairs then chose k.
    with pytest.raises(TypeError, match="max_iter must be an integer, not None"):
        CoupledSampler(log_prob, INIT, kernel=KERNEL, rng=0).meeting_times(5, max_iter=None)


def test_a_names_length_mismatch_names_the_initial_dimension():
    # It blamed names, "names must have 1 entries, not 2", for an initial distribution of
    # the wrong dimension.
    with pytest.raises(ValueError, match="draws points of dimension 1, but names gives 2"):
        CoupledSampler(log_prob, Normal([0.0], [1.0]), names=["p", "lam"])
