use std::collections::BTreeMap;
use whisper_tensor_server::SuperGraphExecutionReport;

#[derive(Clone, Debug)]
struct TierProgress {
    numerator: f64,
    denominator: f64,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct SuperGraphProgressWidgetState {
    by_tier: BTreeMap<i64, TierProgress>,
}

impl SuperGraphProgressWidgetState {
    pub fn clear(&mut self) {
        self.by_tier.clear();
    }

    pub fn ingest_reports(&mut self, reports: Vec<SuperGraphExecutionReport>) {
        for report in reports {
            self.ingest_report(report);
        }
    }

    pub fn ingest_report(&mut self, report: SuperGraphExecutionReport) {
        for (_path, tier, numerator, denominator) in report.progress_reports {
            self.by_tier.insert(
                tier,
                TierProgress {
                    numerator,
                    denominator,
                },
            );
        }
    }

    pub fn is_empty(&self) -> bool {
        self.by_tier.is_empty()
    }

    pub fn show(&self, ui: &mut egui::Ui) {
        if self.by_tier.is_empty() {
            return;
        }

        ui.group(|ui| {
            ui.label("Progress");
            for (tier, progress) in &self.by_tier {
                let ratio = normalized_ratio(progress.numerator, progress.denominator);
                let text = format!(
                    "{:.1}/{:.1} ({:.1}%)",
                    progress.numerator,
                    progress.denominator,
                    ratio * 100.0
                );
                ui.horizontal(|ui| {
                    ui.label(format!("Tier {tier}"));
                    ui.add(
                        egui::ProgressBar::new(ratio as f32)
                            .desired_width(220.0)
                            .show_percentage()
                            .text(text),
                    );
                });
            }
        });
    }
}

fn normalized_ratio(numerator: f64, denominator: f64) -> f64 {
    if !numerator.is_finite() || !denominator.is_finite() || denominator <= 0.0 {
        return 0.0;
    }
    (numerator / denominator).clamp(0.0, 1.0)
}
