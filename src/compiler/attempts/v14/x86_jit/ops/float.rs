//! Float arithmetic on compute slots.
//!
//! Add, Sub, Mul, Div, Min, Max, comparisons (Equal/Less/etc.), and
//! the unary float ops. Operands and results live in XMM registers
//! (the compute representation slot for floats).
//!
//! Per dtype contract §5.3, Min/Max use IEEE 754-2008 minNum/maxNum
//! semantics — the non-NaN operand wins when exactly one is NaN. This
//! requires extra emission compared to bare `minss`/`maxss` (which
//! follows IEEE 754-2019 propagating semantics on x86).
//!
//! Phase 0 stub. Implementation lands in phase 3.
