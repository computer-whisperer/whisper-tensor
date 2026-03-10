//! Attempt 7: parallel-ready crystal execution planning.
//!
//! v7 starts by separating schedule recovery from execution dispatch:
//! we convert recovered additive-reduction loops into explicit tile tasks
//! and execute them with either serial or loop-parallel CPU dispatch.

pub mod codegen;
pub mod executor;
pub mod planner;
