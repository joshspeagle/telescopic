# The identity of the documentation

This page records the design of the telescopic documentation site: its concept, its mark,
its colors and type, and the rules for drawing figures. Josh chose it on 2026-11-13, the
style sheet Overprint and the mark Focus, from a design board of five marks and three
style sheets (issue #103). The page is a design record in the repository and is not built
into the site.

Every color, font, type style, size and coordinate of the identity lives in one place,
`docs/_scripts/brand.py`; the layout (widths, spacing, breakpoints, and the relative
sizes of the landing page's name and tagline and of the sidebar) lives in the theme's
style sheet. That script
writes the theme's tokens, the font rules, the marks, the lockups, the favicon, the
landing page's hero, the Matplotlib style of the executed pages, and the tables below
between their markers. `python docs/_scripts/brand.py --check` fails, naming the file,
when any of them differs from what the script would write, and CI runs it before the
documentation build. A change to the design changes `brand.py` first, then the prose of
this page.

## Concept

A run of telescopic is a set of pairs of chains, and the site is drawn in two inks, one
for each chain. Chain $X$ is magenta and chain $Y$ is teal, and the two colors mean the
two chains and nothing else. Text is set in the deep navy ink that the two inks give
where they overprint, and a line turns to that ink once the chains have met: from the
meeting time on, the pair is one chain. Shapes of the two chains that overlap are
blended as inks are (`multiply` on the light ground, `screen` on the dark one), so the
overlap reads as a third, darker or brighter, color rather than as one shape over the
other.

## The mark

The mark, Focus, is the ray diagram of a refracting telescope drawn as a coupled pair.
Two chains enter an objective lens from the left, $X$ above and $Y$ below, and are
brought to one point, the focus, which is their meeting time. One line in the ink of
met chains leaves the focus through the eyepiece. A ring marks the meeting.

The geometry is fixed in a 256 by 256 square. The lens is
`M62 30 Q96 128 62 226 Q28 128 62 30 Z`; chain $X$ runs from (14, 70) through (62, 70)
to the focus at (178, 128), chain $Y$ from (14, 186) through (62, 186) to the focus; the
merged line runs from the focus to (244, 128); the eyepiece is
`M216 100 Q230 128 216 156 Q202 128 216 100 Z`. In the full cut each chain is 15 steps
jittered about its guide with the board's seed, so that it reads as a chain and not as a
ray, with its steps marked by small beads in the ground color.

The mark has three cuts. They share the geometry and differ in detail and stroke, so
that each reads at its size:

<!-- cuts:start -->
| Cut | Used at | Chain stroke | Detail | Files |
|---|---|---|---|---|
| full | 48 px and up | 11 | jittered steps, step beads, eyepiece, meeting ring | `brand/mark-full-{light,dark}.svg` |
| mid | 24 to 47 px | 18 | jittered steps, eyepiece, meeting ring | `brand/mark-mid-{light,dark}.svg` |
| tiny | below 24 px | 30 | straight chains, lens only | `brand/mark-tiny-{light,dark}.svg` |
<!-- cuts:end -->

Each cut is drawn for the light and the dark ground. The favicon is the tiny cut, and it
follows the browser's color scheme.

Rules for the mark:

- $X$ is always magenta and above, $Y$ always teal and below, and the merged line and the
  ring are always in the ink of met chains. Never recolor, mirror or rotate the mark.
- Use the cut made for the size: never scale the full cut below 48 px.
- Leave clear space of at least a quarter of the mark's height on every side.
- Put the mark on the ground it was drawn for: the light files on `bg` or `surface` of the
  light theme, the dark files on those of the dark theme.

## The wordmark

The wordmark is "telescopic" in lowercase Red Hat Display 800, with the headings' letter
spacing of −0.02 em, converted to outlines so that no font is needed to show it. In the
lockup it sits to the right of the mid cut, its x-height centered on the focus, so that
the merged line leads into the word. The mark's height is 34/22 of the type size, as on
the board. The lockup exists for each ground: `brand/lockup-light.svg` and
`brand/lockup-dark.svg`. The site shows it in the top bar and the footer.

## Color

The ratios are WCAG 2 contrast ratios computed from relative luminance by `brand.py`.
Text tokens (`ink`, `muted`, `accent`, `ochre`, `violet`) must reach 4.5 against both
`bg` and `surface`; the chain colors and `met`, which are marks and never text, must reach
3. The script refuses a token that falls below its minimum. `line` is decorative and has
no minimum; the borders of controls use `muted`.

<!-- colors:start -->
| Token | Role | Light | on `bg` | on `surface` | Dark | on `bg` | on `surface` |
|---|---|---|---|---|---|---|---|
| `bg` | page ground | `#FBFAF7` |  |  | `#0A1626` |  |  |
| `surface` | cards, code, bars | `#F1EFE9` |  |  | `#11223A` |  |  |
| `ink` | text | `#062A4F` | 13.86 | 12.58 | `#EEF2F6` | 16.15 | 14.21 |
| `muted` | secondary text, comments, control borders | `#55657A` | 5.70 | 5.18 | `#9FB0C3` | 8.20 | 7.21 |
| `line` | rules and borders (decorative) | `#D9D5CB` | 1.40 | 1.27 | `#22385A` | 1.54 | 1.36 |
| `cx` | chain X (magenta), in marks and figures only | `#D93A74` | 4.18 | 3.80 | `#FF6FA3` | 6.97 | 6.13 |
| `cy` | chain Y (teal), in marks and figures only | `#0095A6` | 3.44 | 3.12 | `#35C6D6` | 8.83 | 7.76 |
| `met` | the chains after they meet | `#062A4F` | 13.86 | 12.58 | `#FFFFFF` | 18.17 | 15.98 |
| `accent` | links, focus, interactive | `#006C7A` | 5.87 | 5.33 | `#5ED3E0` | 10.27 | 9.04 |
| `ochre` | strings in code; warnings | `#8A5A00` | 5.68 | 5.15 | `#E6B65C` | 9.70 | 8.53 |
| `violet` | numbers in code | `#6B3FA0` | 7.07 | 6.42 | `#C3A6F0` | 8.68 | 7.64 |
<!-- colors:end -->

The board fixed the first nine tokens. `ochre` and `violet` were added for code and
notes, because the chain colors are not syntax colors: strings are `ochre`, numbers
`violet`, keywords and built-in names `accent`, names `ink` and comments `muted`.
Warnings and errors in notes and outputs use `ochre`, the only warm hue that is neither
magenta nor teal.

Plots are drawn on the light ground in both themes, and the dark theme sets them on a
light panel. Series that are not the two chains take these colors, in order. None is
magenta or teal, so the chain colors keep their meaning; adjacent colors differ by a
Delta E of at least 23 under simulated deutan, protan and tritan vision.

<!-- plots:start -->
| Slot | Color | on the figure ground |
|---|---|---|
| 0 | `#2A5DB0` | 6.11 |
| 1 | `#C07A00` | 3.34 |
| 2 | `#7B4FB8` | 5.51 |
| 3 | `#5A8A1F` | 3.95 |
<!-- plots:end -->

## Type

Headings are Red Hat Display, running text Red Hat Text and code Red Hat Mono. The site
serves the Latin subset of the weights below as woff2 files, with the SIL Open Font
License beside them; `docs/_scripts/fetch_fonts.py` downloads them from the pinned
Fontsource packages and checks each package's SHA-256. Greek letters and mathematics are
set by MathJax, not by these faces, and other scripts fall back to the system font.

<!-- type:start -->
| Role | Face | Weight | Size | Line height | Letter spacing |
|---|---|---|---|---|---|
| page titles | Red Hat Display | 800 | 38 px | 1.1 | -0.02 em |
| page titles below 900 px | Red Hat Display | 800 | 30 px | 1.12 | -0.02 em |
| section headings | Red Hat Display | 800 | 24 px | 1.2 | -0.02 em |
| subsection headings | Red Hat Display | 600 | 19 px | 1.3 | -0.01 em |
| minor headings, card titles | Red Hat Display | 600 | 17 px | 1.3 | -0.01 em |
| running text | Red Hat Text | 400 | 16.5 px | 1.6 | 0 |
| code and outputs | Red Hat Mono | 400 | 14 px | 1.55 | 0 |
| captions, sidebar, outline | Red Hat Text | 400 | 14 px | 1.45 | 0 |
| labels, upper case | Red Hat Mono | 400 | 12 px | 1.4 | 0.06 em |
<!-- type:end -->

<!-- sizes:start -->
| Size | Value | Use |
|---|---|---|
| `radius` | 10 px | corner radius of cards, code, notes and controls |
| `rule` | 3 px | rules marking notes, cards and the current page |
| `focus` | 2 px | focus ring |
| mark : wordmark | 1.545 | the mark's height over the type size |
<!-- sizes:end -->

## Drawing rules for figures

1. **The two chain colors mean the two chains.** Magenta is $X$ and teal is $Y$, in every
   figure, and they mean nothing else. A figure whose series are not the two chains uses
   the plot colors above.
2. **A line turns `met` once the chains have met.** After the meeting time a pair is one
   chain, drawn in the ink of met chains.
3. **Uncertainty is drawn.** An estimate is shown with its error bar or band, never as a
   bare point or line.
4. **Direct labels over legends.** Label a line or a region where it is; use a legend only
   when labels would collide.
5. **Every figure has a caption and alternative text.** The caption says what is drawn
   and what to read from it, in complete sentences; the alternative text describes the
   figure for a reader who cannot see it.
6. **Plots are drawn on the light ground.** The shared Matplotlib style, which `conf.py`
   hands to every notebook kernel through `MATPLOTLIBRC`, sets the ground, the ink, the
   axes in `muted`, the grid in `line` and the plot colors.

The hero of the landing page follows these rules. It is one pair of chains run by
telescopic itself (`docs/_scripts/hero_pair.py` saves its coordinates, with the seed and
the package's commit, in `hero_pair.json`), drawn in the Focus composition: $X_t$ and,
as the estimator pairs them, $Y_{t-1}$ against time, joined at the actual meeting time,
after which one line continues. Its vertical scale is one for both chains. Without
JavaScript, and for a reader who prefers reduced motion, it is a still image; otherwise it
draws the pair over a few seconds, repeats, and has a pause button.

## Components

The theme `docs/_themes/telescopic/` inherits only Sphinx's `basic` theme, and the
extension `docs/_ext/telescopic_theme.py` gives its templates what they need.

- **Top bar:** the lockup, the sections (the captioned root toctrees of `docs/index.md`,
  the current one underlined in the ink of met chains), search, the theme toggle and a link to the
  repository. Below 1080 px the sections move into a menu.
- **Sidebar:** the pages of the current section, the current page marked by a rule in
  the ink of met chains. Below 900 px it becomes a full-screen menu.
- **On this page:** the outline of a page with two or more sections, its current section
  marked in `accent` as the reader scrolls.
- **Code:** blocks on `surface` with a `line` border and the corner radius, token colors
  as above, a copy button; notebook outputs joined below their cell on `bg`, with a rule
  in `ochre` for standard error and warnings; plots on a light panel.
- **Notes:** admonitions on `surface` with a rule in `accent`, or in `ochre` for warnings.
- **API:** signatures as a `surface` bar with an ink rule, numpydoc field names as
  labels.
- **Landing page:** the name, a tagline and two sentences, the hero, three cards routed
  by reader, and an executed example with its output, on one screen at 1440 by 900.
- **Ornament:** two overlapping translucent discs in the chain colors, blended as inks,
  used once, at the top of the landing page.

## Accessibility

The site aims at WCAG 2.2 AA. Text, links and code tokens meet 4.5:1 against both
grounds in both themes (the table above), and the chain colors meet 3:1 as marks.
Every interactive element has a visible focus ring in `accent`; the menu, search, theme
toggle and pause button are reachable by keyboard, and blocks that scroll sideways
(code, outputs, tables, equations, signatures) receive focus so that the keyboard can
scroll them. A skip link leads to the content. The theme follows the system's color
scheme until the reader picks one, which is remembered. The hero's animation stops for a
reader who prefers reduced motion and has a pause button otherwise. Without JavaScript
every page is complete, and the menu is shown as a list.
