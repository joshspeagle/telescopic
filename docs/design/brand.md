# The identity of the documentation

This page records the design of the telescopic documentation site: its concept, its mark,
its colors and type, and the rules for drawing figures. Josh chose it on 2026-11-13, the
style sheet Overprint and the mark Focus, from a design board of five marks and three
style sheets (issue #103). The page is a design record in the repository and is not built
into the site.

Every color, font, type style, size and coordinate of the identity is defined in
`docs/_scripts/brand.py`, whose docstring lists the files it writes; the tables below,
between their markers, are among them. The theme's style sheet holds the layout (widths,
spacing and breakpoints) and these sizes only: the landing page's name and tagline, the
top bar's links and search field, the hero's labels, the larger links and captions of the
phone menu, and sizes relative to the running text for inline code, tables, the copy
button and search results. A change to the design changes `brand.py` first, then the
prose of this page.

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

The geometry is fixed in a square, and `brand.py` holds its coordinates. In the full cut
each chain is jittered about its guide with the board's seed, so that it reads as a chain
and not as a ray, with its steps marked by small beads in the ground color.

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

The wordmark is "telescopic" in lowercase Red Hat Display 800, with the page titles'
letter spacing, converted to outlines so that no font is needed to show it. In the
lockup it sits to the right of the mid cut, its x-height centered on the focus, so that
the merged line leads into the word. The mark's height over the type size is the board's
(the sizes table). The lockup exists for each ground: `brand/lockup-light.svg` and
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
| `met` | the chains after they meet | `#062A4F` | 13.86 | 12.58 | `#FFFFFF` | 18.17 | 15.99 |
| `accent` | links, focus, interactive | `#006C7A` | 5.87 | 5.33 | `#5ED3E0` | 10.27 | 9.03 |
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
magenta or teal, so the chain colors keep their meaning. Under normal vision and under
simulated deutan, protan and tritan vision (Machado, Oliveira & Fernandes 2009), every
pair of plot colors, and each plot color against `cx`, `cy` and `met`, differ by a
CIEDE2000 Delta E of at least 15, and each plot color has 3:1 contrast on the figure
ground; `brand.py` refuses a palette that falls short. The second table gives every
difference and the least for each vision.

<!-- plots:start -->
| Slot | Color | on the figure ground |
|---|---|---|
| 0 | `#1E4BF0` | 6.06 |
| 1 | `#B48700` | 3.14 |
| 2 | `#960000` | 8.76 |
| 3 | `#00694B` | 6.44 |

| Delta E (CIEDE2000) | normal | deutan | protan | tritan |
|---|---|---|---|---|
| slot 0 and slot 1 | 67.1 | 70.2 | 65.2 | 47.7 |
| slot 0 and slot 2 | 46.4 | 61.5 | 57.8 | 56.7 |
| slot 1 and slot 2 | 41.1 | 25.9 | 31.6 | 27.0 |
| slot 0 and slot 3 | 44.4 | 39.0 | 44.3 | 17.4 |
| slot 1 and slot 3 | 39.7 | 31.6 | 23.3 | 48.7 |
| slot 2 and slot 3 | 56.9 | 17.1 | 17.0 | 53.2 |
| slot 0 and `cx` | 35.3 | 45.0 | 20.4 | 60.6 |
| slot 1 and `cx` | 50.3 | 19.8 | 38.4 | 15.8 |
| slot 2 and `cx` | 27.8 | 23.8 | 32.6 | 20.4 |
| slot 3 and `cx` | 67.2 | 17.3 | 20.3 | 61.2 |
| slot 0 and `cy` | 33.1 | 18.5 | 20.6 | 18.0 |
| slot 1 and `cy` | 42.9 | 44.8 | 39.0 | 47.0 |
| slot 2 and `cy` | 55.7 | 43.7 | 45.4 | 60.5 |
| slot 3 and `cy` | 25.9 | 27.3 | 29.5 | 19.2 |
| slot 0 and `met` | 22.5 | 24.0 | 27.0 | 24.6 |
| slot 1 and `met` | 57.7 | 61.7 | 55.9 | 53.7 |
| slot 2 and `met` | 40.9 | 45.6 | 37.2 | 44.2 |
| slot 3 and `met` | 35.5 | 30.1 | 33.8 | 18.9 |
| `cx` and `cy`, for comparison | 63.6 | 25.7 | 16.7 | 63.6 |
| least of the plot colors (refused below 15) | 22.5 | 17.1 | 17.0 | 15.8 |
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
| captions, sidebar, outline, card text | Red Hat Text | 400 | 14 px | 1.45 | 0 |
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
   the plot colors above. The package's own plots take the chain colors and `met` from
   the constants `CHAIN_X`, `CHAIN_Y` and `MET` of `telescopic.plotting`, which
   `brand.py` writes from the light values of `cx`, `cy` and `met`.
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
  repository. Below 1080 px the sections leave the top bar: on the landing page they move
  into a menu, and on other pages the sidebar lists every section.
- **Sidebar:** the pages of the current section, in the caption style, the current page
  marked by a rule in the ink of met chains. Below 900 px it becomes a full-screen menu.
- **On this page:** the outline of a page with two or more sections, in the caption
  style, its current section marked in `accent` as the reader scrolls.
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

The site aims at WCAG 2.2 AA, and the colors meet the minimums of the Color section in
both themes. Every interactive element has a visible focus ring in `accent`; the menu, search, theme
toggle and pause button are reachable by keyboard, and blocks that scroll sideways
(code, outputs, tables, equations, signatures) receive focus so that the keyboard can
scroll them. A skip link leads to the content. The theme follows the system's color
scheme until the reader picks one, which is remembered. Without JavaScript
every page is complete, and the menu is shown as a list.
