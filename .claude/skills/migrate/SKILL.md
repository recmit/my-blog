---
name: migrate
description: Run one phase of the fastpages-to-Astro migration described in docs/migration-plan.md.
argument-hint: "[phase-number]"
arguments: [phase]
disable-model-invocation: true
---

Run exactly one phase of the Astro migration.

## 1. Orient

Read, in this order:

1. `docs/migration-plan.md` — invariants, pre-made decisions, the "Do not" list,
   the phases, and the handoff log at the bottom
2. `AGENTS.md` — conventions
3. `docs/code-map.md` — where things belong

Work out which phase is yours:

- If `$phase` is set, that is your phase.
- Otherwise read the `**Status:**` line at the top of the plan and take the
  phase named after `Next:`.
- If that line is missing, ambiguous, or disagrees with what is actually on
  disk, **stop and ask.** Do not guess.

Say which phase you are starting and what its gate is, then begin.

## 2. Work

Do that phase and nothing else. If you notice a problem belonging to another
phase, record it in the handoff log — do not fix it.

The plan's pre-made decisions are settled. If one turns out to be impossible,
**stop and ask** rather than substituting an alternative.

## 3. Finish

Before opening a pull request:

1. Confirm your phase's gate passes. If it does not, stop and report why.
   Do not open the PR.
2. Update the `**Status:**` line to `Phase N complete. Next: Phase N+1.`
3. Append your entry to the handoff log at the bottom of the plan.
   **Ten lines maximum.** Only: what differed from the plan, what the next agent
   needs to know, and anything left broken.
4. If a step in the plan was wrong, **edit that step** so it is right. Do not
   annotate it with what you did instead.
5. If a real decision changed, add one short entry to `docs/decisions.md`.

Then open a pull request against `master`. **Do not merge it.** The PR
description is for the repository owner, not the next agent — say what you did,
what surprised you, and what wants a human eye.

## Never

- Restructure or expand docs outside your phase. The line budgets in the plan
  are limits, not targets.
- Write handoff notes about phases you did not run.
- Merge anything, or push to `master`.
- Re-execute notebooks, change published URLs, or delete the Jekyll tree before
  Phase 7.

---

This skill is temporary. Delete it when the migration lands.
