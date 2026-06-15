# Legacy Evaluation Asset Map

This redesign deliberately reuses evaluation ideas from `../legacy/` without inheriting the old serving architecture.

## Reused sources

- `../legacy/eval/datasets.py`
  - reused as a prompt source for stable FAQ, freshness-sensitive prompts, anchored identifier prompts, and ambiguous follow-up prompts
- `../legacy/eval/replay.py`
  - reused as a design reference for offline replay and metrics-driven evaluation instead of live-only validation
- `../legacy/eval/miss_diagnostics.py`
  - reused as a design reference for wrong-reuse analysis and for explaining why a candidate should not have been reused
- legacy trace and replay prompts
  - adapted into the seed routing dataset to keep the redesign grounded in real prompt shapes

## Explicitly not reused

- analyzer and gatekeeper abstractions
- neighbor judge logic
- rich semantic extraction objects
- pipeline toggles that switch between legacy architectures

## Mapping into the new system

- legacy reusable documentation-style prompts -> `SEMANTIC_OK`
- legacy freshness-sensitive prompts -> `SKIP_CACHE`
- legacy anchored identifier prompts -> `EXACT_ONLY`
- legacy mutation/follow-up prompts -> `THREAD_SCOPED_ONLY`

The point of reuse is measurement and failure memory, not code compatibility.
