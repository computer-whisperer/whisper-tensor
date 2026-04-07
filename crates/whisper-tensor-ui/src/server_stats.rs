//! Client-side history of server resource snapshots and rendering for the
//! Resources tab.
//!
//! The server pushes [`ServerStatsSnapshot`]s ~1 Hz over the websocket.
//! [`ServerStatsHistory`] keeps a fixed-capacity ring of recent samples;
//! [`render_server_stats`] draws line plots over them.

use std::collections::VecDeque;

use egui_plot::{Line, Plot, PlotPoints};
use whisper_tensor_server::ServerStatsSnapshot;

/// How many samples to retain. At ~1 Hz this is roughly 5 minutes of history.
const HISTORY_CAPACITY: usize = 300;

pub struct ServerStatsHistory {
    samples: VecDeque<ServerStatsSnapshot>,
}

impl ServerStatsHistory {
    pub fn new() -> Self {
        Self {
            samples: VecDeque::with_capacity(HISTORY_CAPACITY),
        }
    }

    pub fn push(&mut self, snapshot: ServerStatsSnapshot) {
        // Guard against the watch channel's Default snapshot leaking through
        // (would plot at unix epoch 0 = ~56 years ago on the time axis).
        if snapshot.sample_unix_ms == 0 {
            return;
        }
        if self.samples.len() == HISTORY_CAPACITY {
            self.samples.pop_front();
        }
        self.samples.push_back(snapshot);
    }

    pub fn latest(&self) -> Option<&ServerStatsSnapshot> {
        self.samples.back()
    }

    /// Build plot points using `seconds-ago` as the x axis (so the live edge
    /// is always at x=0 and older samples have negative x).
    fn points<F: Fn(&ServerStatsSnapshot) -> f64>(&self, f: F) -> PlotPoints<'_> {
        let Some(latest) = self.samples.back() else {
            return PlotPoints::default();
        };
        let now_ms = latest.sample_unix_ms as f64;
        self.samples
            .iter()
            .map(|s| {
                let dt_secs = (s.sample_unix_ms as f64 - now_ms) / 1000.0;
                [dt_secs, f(s)]
            })
            .collect()
    }
}

impl Default for ServerStatsHistory {
    fn default() -> Self {
        Self::new()
    }
}

fn format_bytes(b: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    let v = b as f64;
    if v >= GB {
        format!("{:.2} GiB", v / GB)
    } else if v >= MB {
        format!("{:.2} MiB", v / MB)
    } else if v >= KB {
        format!("{:.2} KiB", v / KB)
    } else {
        format!("{b} B")
    }
}

fn format_uptime(secs: u64) -> String {
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    if h > 0 {
        format!("{h}h {m:02}m {s:02}s")
    } else if m > 0 {
        format!("{m}m {s:02}s")
    } else {
        format!("{s}s")
    }
}

pub fn render_server_stats(ui: &mut egui::Ui, history: &ServerStatsHistory) {
    ui.heading("Server Resources");

    let Some(latest) = history.latest() else {
        ui.label("Waiting for first sample from server...");
        return;
    };

    egui::Grid::new("server_stats_summary")
        .striped(true)
        .show(ui, |ui| {
            ui.label("Process RSS");
            ui.label(format_bytes(latest.process_rss_bytes));
            ui.end_row();
            ui.label("Process VSZ");
            ui.label(format_bytes(latest.process_vsz_bytes));
            ui.end_row();
            ui.label("Process CPU");
            ui.label(format!("{:.1}%", latest.process_cpu_percent));
            ui.end_row();
            ui.label("In-flight jobs");
            ui.label(latest.in_flight_jobs.to_string());
            ui.end_row();
            ui.label("Uptime");
            ui.label(format_uptime(latest.uptime_secs));
            ui.end_row();
            ui.label("Samples");
            ui.label(format!("{} / {}", history.samples.len(), HISTORY_CAPACITY));
            ui.end_row();
        });

    ui.add_space(8.0);
    ui.separator();
    ui.add_space(8.0);

    egui::ScrollArea::vertical().show(ui, |ui| {
        // Memory plot — RSS and VSZ in MiB.
        ui.label("Memory (MiB)");
        Plot::new("plot_memory")
            .height(180.0)
            .x_axis_label("seconds (live edge = 0)")
            .show(ui, |plot_ui| {
                plot_ui.line(Line::new(
                    "RSS",
                    history.points(|s| s.process_rss_bytes as f64 / (1024.0 * 1024.0)),
                ));
                plot_ui.line(Line::new(
                    "VSZ",
                    history.points(|s| s.process_vsz_bytes as f64 / (1024.0 * 1024.0)),
                ));
            });

        ui.add_space(8.0);

        // CPU plot — percent.
        ui.label("CPU (%)");
        Plot::new("plot_cpu")
            .height(150.0)
            .x_axis_label("seconds (live edge = 0)")
            .show(ui, |plot_ui| {
                plot_ui.line(Line::new(
                    "CPU%",
                    history.points(|s| s.process_cpu_percent as f64),
                ));
            });

        ui.add_space(8.0);

        // In-flight jobs plot.
        ui.label("In-flight scheduler jobs");
        Plot::new("plot_in_flight")
            .height(120.0)
            .x_axis_label("seconds (live edge = 0)")
            .show(ui, |plot_ui| {
                plot_ui.line(Line::new(
                    "jobs",
                    history.points(|s| s.in_flight_jobs as f64),
                ));
            });
    });
}
