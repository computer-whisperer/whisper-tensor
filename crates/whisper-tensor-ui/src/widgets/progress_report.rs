use std::collections::BTreeMap;
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;
use whisper_tensor_server::SuperGraphExecutionReport;

#[derive(Clone, Debug)]
struct TierProgress {
    numerator: f64,
    denominator: f64,
}

#[derive(Clone, Debug)]
struct LoadingWeightStatus {
    weight_name: Option<String>,
    observed_at: Instant,
}

const WEIGHT_STATUS_TTL: Duration = Duration::from_secs(4);

#[derive(Clone, Debug, Default)]
pub(crate) struct SuperGraphProgressWidgetState {
    by_tier: BTreeMap<i64, TierProgress>,
    loading_weight_status: Option<LoadingWeightStatus>,
}

impl SuperGraphProgressWidgetState {
    pub fn clear(&mut self) {
        self.by_tier.clear();
        self.loading_weight_status = None;
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
        for (_path, weight_name, age) in report.loading_weight_reports {
            self.loading_weight_status = Some(LoadingWeightStatus {
                weight_name,
                observed_at: Instant::now() - age,
            });
        }
    }

    pub fn is_empty(&self) -> bool {
        self.by_tier.is_empty() && self.current_weight_status().is_none()
    }

    fn current_weight_status(&self) -> Option<(&LoadingWeightStatus, Duration)> {
        let status = self.loading_weight_status.as_ref()?;
        let age = Instant::now().saturating_duration_since(status.observed_at);
        if age >= WEIGHT_STATUS_TTL {
            None
        } else {
            Some((status, age))
        }
    }

    pub fn show(&self, ui: &mut egui::Ui) {
        let loading_status = self.current_weight_status();
        if self.by_tier.is_empty() && loading_status.is_none() {
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
            if let Some((status, age)) = loading_status {
                let ttl = WEIGHT_STATUS_TTL.as_secs_f32();
                let fade = (1.0 - age.as_secs_f32() / ttl).clamp(0.0, 1.0);
                let text = match &status.weight_name {
                    Some(name) => format!("Loading weight: {name}"),
                    None => "Loading weight...".to_string(),
                };
                let color =
                    egui::Color32::from_rgba_unmultiplied(220, 220, 220, (255.0 * fade) as u8);
                ui.label(egui::RichText::new(text).color(color));
                ui.ctx().request_repaint_after(Duration::from_millis(33));
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
