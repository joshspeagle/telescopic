# Two chains that meet

telescopic runs its chains in pairs that are built to become exactly equal. This page
explains what such a pair is, how two chains on a continuous space can land on the same
point, why they stay together once they have, and what the lag between them is for. It
uses the notation of [Theory](../theory.md#coupled-chains).

## A coupling

A pair consists of two chains, $X$ and $Y$, that move by the same kernel $P$ and start
from independent draws of the same initial distribution $\pi_0$. A **coupling** of $P$
moves the two chains together: it draws the random numbers of their two moves jointly,
so that the moves are related, while each chain on its own moves exactly as $P$ would
move it. Someone who watched only $X$, or only $Y$, would see an ordinary Markov chain
with kernel $P$, and at each time $t$ each chain has the law $\pi_t$ of an uncoupled chain
started from $\pi_0$. The coupling changes how the two chains move relative to each
other, and nothing about either one alone; in particular it does not make either chain
approach the target faster ([Theory](../theory.md#guarantees-and-their-limits)).

Each kernel of telescopic comes with its coupling
([Choose a kernel](../guide/kernels.md)). Random-walk Metropolis shows how a coupling
can make two chains equal. Each chain proposes a point from a Gaussian centered on its
current state and accepts it with the Metropolis probability. The coupling draws the two
proposals from the **reflection-maximal coupling** of the two Gaussians: it draws $X$'s
proposal, and with probability equal to the ratio of $Y$'s proposal density to $X$'s at
that point, capped at 1, it gives $Y$ the same point; otherwise it gives $Y$ the mirror
image of $X$'s proposal across the line, plane or hyperplane halfway between the two
centers, in coordinates where the proposal covariance is the identity. Each proposal on
its own has its chain's Gaussian law, and the two coincide as often as two draws from
those Gaussians can. Both chains then decide whether to accept with one shared uniform number,
so that when the proposals coincide and both chains accept, the chains are at the same
point.

```{diagram} reflection
A static frame of the reflection-maximal coupling of two Gaussian proposals with one
covariance, in coordinates where that covariance is the identity: $X$'s proposal density
in magenta, centered at $\mu_x$, and $Y$'s in teal, centered at $\mu_y$, with contours at
one and two standard deviations; where they overlap the inks overprint. Two draws of
{func}`telescopic.couplings.reflection_maximal` are marked. The first falls where $X$'s
density is far above $Y$'s, and $Y$'s proposal is its reflection across the mirror line.
The second falls in the overlap and is given to both chains. The proposals coincide with
probability $2\Phi(-\delta/2)$, with $\delta$ the distance between the centers.
```

The probability $2\Phi(-\delta/2)$ falls quickly as the two centers move apart,
measured in units of the proposal's spread, so the proposals of chains several
proposal widths apart rarely coincide. The meeting time of a pair is random, and it
varies from pair to pair.

## Equal, and then together

The **meeting time** $\tau$ is the first time at which the two chains are equal. Equality
is exact: telescopic compares the coordinates with `==`, without a tolerance. From the
meeting on, telescopic runs $X$ alone and sets $Y$ to the same state at every step, so
the chains stay equal, and a pair that has met costs one chain
([Theory](../theory.md#coupled-chains)). A meeting means that the two chains agree, not
that they have reached the target: they continue from wherever they met with the law
$\pi_t$ of a single chain ([How it works](../tutorials/how-it-works.md#pairs-of-chains)).

## The lag

The second chain runs $L$ steps behind the first: $X$ moves alone for its first $L$
steps, and from then on $X_t$ is paired with $Y_{t-L}$, which has the law
$\pi_{t-L}$ of the chain $L$ steps earlier. The meeting time is the first $t$ at which
$X_t = Y_{t-L}$.

```{diagram} two-chains
One pair run by telescopic with random-walk Metropolis on a standard normal target, the
gray density on its side at the right, from chains started at draws of
$\mathcal N(5, 1)$. $X_t$ is solid in magenta and $Y_{t-L}$ dashed in teal, drawn $L$
steps to the right, at the time at which $X$ is compared with it. The dashed line joins
the solid one at the meeting time $\tau$, marked by a dot, after which one line continues
in the ink of met chains. The estimator averages the shaded times $k$ to $m$, and the
differences $X_t - Y_{t-L}$ at the hatched times $k + L$ to $\tau - 1$ make its bias
correction.
```

The lag sets which differences the pair provides. A difference between $X_t$ and
$Y_{t-L}$ compares the chain at two times $L$ steps apart, which is what
[the telescoping correction](telescoping-correction.md) adds up. The lag also sets how
much correction a pair needs: a pair that meets within $k + L$ steps contributes no
correction, and its estimate is the ordinary average after the {term}`burn-in`. telescopic's
automatic settings take $L = \max(1, k)$, so that most pairs meet that early
([Settings](../guide/settings.md#the-settings)), and
[Choosing k, m and the lag](../tutorials/choosing-settings.md) compares a lag of 1 with
a lag of $k$ on one target.
