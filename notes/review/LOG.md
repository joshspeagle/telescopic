# Review log

Append only. One entry per working session.

## 2026-10-24: the hub

- Josh asked for an interactive guide to a full teardown of the codebase, across levels
  (big picture to implementation) and depths (statistics to numerics), with the
  experiments behind the decisions, framed against the spec, and a log of questions and
  decisions handled at checkpoints or when he asks to clear the queue.
- The hub is published (`ARTIFACTS.md`): ten phases and 33 steps, each at a default
  level (L1 overview, L2 guided tour, L3 deep dive, as in pastry's walkthrough) and with
  its depths; the spec section by section with the steps and decisions that cover it;
  the 27 design-time and 57 later decisions; the 18 experiments with their outputs; the
  module map. Its queue starts with the eight open questions of `notes/QUESTIONS.md`,
  each answered with its default.
- Built at dc2f0df. It is rebuilt after the second audit's fixes, before the review starts.
