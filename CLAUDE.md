# Whisper-Tensor

## This Project's Character

Whisper-tensor is a long-lived, architecturally intensive project. The architecture is not sacred — it is the best design we have come up with so far, and any part of it may be revisited when a better approach becomes clear. The user built and iterated on this codebase for months before involving LLM agents, routinely tearing up working code to rebuild it when experience revealed a better structure. That willingness to rework is the point, not a sign of failure.

### What this means for you

- **Do a small piece well.** A single context window cannot complete a major rework of this project, and you should not try. Pick a bounded scope, do it to a standard you would be proud of, and report honestly where you got to. Three ops implemented correctly is worth more than nine ops implemented with shortcuts. Do not pad progress with things that will need to be redone.

- **General before specific.** When building or replacing a subsystem, implement the fully general path first, even if it's slow. Fast paths are optimizations layered on top of a correct foundation — never the foundation itself. A subsystem that handles 93% of inputs via fast paths and falls back for the remaining 7% has the architecture inverted: the 7% is where the design needs to live, and the fast paths should be drop-in optimizations that preserve it.

- **Friction is a signal, not an obstacle.** When your implementation doesn't fit cleanly into the existing architecture, that is valuable information. Maybe the architecture needs to evolve — or maybe your approach is wrong. Either way, surface it. "This doesn't fit cleanly and here's why" is one of the most useful things you can report. Do not silently work around friction with wrappers, hacks, hardcoded special cases, or `extern "C"` call boundaries the user didn't approve. Architectural decisions that affect the performance model (call boundaries, memory layout, fallback paths) require discussion before implementation, even when they seem like the obvious choice.

- **Unfinished is better than faked.** Report partial progress honestly: "6 of 9 ops work, these 3 need discussion." Never disable tests, narrow scope, label things "unsupported," or rewrite design docs to make your work look more complete than it is. The user plans next steps based on your reports — false completion wastes far more time than honest incompleteness. If a plan specifies testing gates or success criteria, evaluate your results against them explicitly before declaring a phase complete.

- **After correction, re-derive from the original goal.** When the user corrects your approach, do not patch the correction onto your previous attempt and keep going. Re-read the original instructions and think the problem through from scratch. The pattern of agreeing with a correction and then reverting to the same shortcut in the next paragraph is a sign you are iterating on the wrong starting point.
