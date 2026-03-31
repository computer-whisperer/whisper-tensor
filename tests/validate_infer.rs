//! Tests that validate `infer()` correctness by running the validation harness
//! against test_set cases.
//!
//! For each test case in the internal test suite, feeds the MilliOpGraph and its
//! inputs into `validate_infer_against_pool_eval`, which checks that `infer()`
//! never returns *wrong* information at any ablation level.

use whisper_tensor::test_set::build_test_set;

#[test]
fn validate_infer_all_test_set_cases() {
    let cases = build_test_set();
    let mut total_pass = 0;
    let mut total_unable = 0;
    let mut total_fail = 0;
    let mut total_data_sets = 0;

    for case in &cases {
        for ds in &case.data_sets {
            total_data_sets += 1;

            let views: Vec<_> = ds.inputs.iter().map(|(id, t)| (*id, t.view())).collect();
            let view_map = views.iter().map(|(id, v)| (*id, v)).collect();

            let report = case.graph.validate_infer_against_pool_eval(&view_map);

            total_pass += report.pass_count;
            total_unable += report.unable_to_infer_count;
            total_fail += report.failure_count;

            assert_eq!(
                report.failure_count, 0,
                "Infer validation failures for '{}' / '{}':\n{}",
                case.name, ds.label, report
            );
        }
    }

    eprintln!(
        "validate_infer: {} cases, {} data sets, {} pass, {} unable-to-infer, {} failures",
        cases.len(),
        total_data_sets,
        total_pass,
        total_unable,
        total_fail
    );
}
