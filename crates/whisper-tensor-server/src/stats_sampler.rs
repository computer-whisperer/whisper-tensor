//! Periodic process-level resource sampler.
//!
//! Owns a [`tokio::sync::watch`] channel of [`ServerStatsSnapshot`]. A single
//! background task wakes once a second, asks `sysinfo` for the current
//! process's RSS / VSZ / CPU%, attaches the in-flight scheduler job count,
//! and broadcasts. Per-handler tasks subscribe via [`StatsSampler::subscribe`]
//! and forward each new snapshot to their connected client.
//!
//! The sampler runs unconditionally — `tokio::sync::watch` makes this cheap
//! when no receivers exist (the value is just replaced in place).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};
use tokio::sync::watch;
use tokio::time;

use crate::ServerStatsSnapshot;

/// Period between snapshots.
const SAMPLE_INTERVAL: Duration = Duration::from_millis(1000);

/// Owns the watch channel and the in-flight job counter shared with the
/// scheduler. Construct once at server startup, hand out subscribers, then
/// spawn [`StatsSampler::run`] to drive periodic updates.
pub struct StatsSampler {
    sender: watch::Sender<ServerStatsSnapshot>,
    receiver: watch::Receiver<ServerStatsSnapshot>,
    in_flight_jobs: Arc<AtomicUsize>,
    started_at: Instant,
}

impl StatsSampler {
    pub fn new(in_flight_jobs: Arc<AtomicUsize>) -> Self {
        let (sender, receiver) = watch::channel(ServerStatsSnapshot::default());
        Self {
            sender,
            receiver,
            in_flight_jobs,
            started_at: Instant::now(),
        }
    }

    /// Hand out a fresh subscriber. Each client websocket handler should call
    /// this once at session start.
    pub fn subscribe(&self) -> watch::Receiver<ServerStatsSnapshot> {
        self.receiver.clone()
    }

    /// Drive the sampling loop. Intended to be spawned with `tokio::spawn`.
    pub async fn run(self) {
        let pid = Pid::from_u32(std::process::id());
        let mut sys = System::new();
        let refresh_kind = ProcessRefreshKind::new().with_memory().with_cpu();

        // Prime CPU sampling: sysinfo computes %CPU as a delta between two
        // refreshes, so the first reading after a single refresh is always 0.
        sys.refresh_processes_specifics(ProcessesToUpdate::Some(&[pid]), true, refresh_kind);

        let mut ticker = time::interval(SAMPLE_INTERVAL);
        ticker.set_missed_tick_behavior(time::MissedTickBehavior::Delay);

        loop {
            ticker.tick().await;

            sys.refresh_processes_specifics(ProcessesToUpdate::Some(&[pid]), true, refresh_kind);

            let snapshot = if let Some(process) = sys.process(pid) {
                ServerStatsSnapshot {
                    process_rss_bytes: process.memory(),
                    process_vsz_bytes: process.virtual_memory(),
                    process_cpu_percent: process.cpu_usage(),
                    in_flight_jobs: self.in_flight_jobs.load(Ordering::Relaxed) as u64,
                    uptime_secs: self.started_at.elapsed().as_secs(),
                    sample_unix_ms: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(0),
                }
            } else {
                // Process disappeared? Send a zeroed snapshot rather than
                // skipping a tick — keeps the time series continuous.
                ServerStatsSnapshot {
                    in_flight_jobs: self.in_flight_jobs.load(Ordering::Relaxed) as u64,
                    uptime_secs: self.started_at.elapsed().as_secs(),
                    sample_unix_ms: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(0),
                    ..ServerStatsSnapshot::default()
                }
            };

            // send_replace ignores the "no receivers" case — we want sampling
            // to keep running and to keep the latest value cached for any
            // client that connects later.
            self.sender.send_replace(snapshot);
        }
    }
}
