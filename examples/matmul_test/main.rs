mod cranelift_0;
mod cranelift_1;
mod cranelift_2;
mod cranelift_3;
mod cranelift_4;
mod cranelift_5;
mod cranelift_6;
mod reports;

use ndarray::{Array2, Axis};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::env;
use std::time::Instant;

// Cranelift imports
use cranelift_codegen::settings::{self, Flags, Configurable};

use cranelift_module::{Linkage, Module};
use cranelift_native;
use cranelift_codegen::ir::InstBuilder;

trait MatmulImpl {
    fn name(&self) -> &'static str;
    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32>;
    // Generate an implementation-specific markdown report into the given directory.
    // Default: no-op.
    fn make_report(&self, _out_dir: &std::path::Path) {}
}

trait MatmulImplBuilder {
    fn name(&self) -> &'static str;
    fn build(&self, m: usize, k: usize, n: usize) -> Box<dyn MatmulImpl>;
}

struct NdarrayImpl {
    m: usize,
    k: usize,
    n: usize,
}

impl MatmulImpl for NdarrayImpl {
    fn name(&self) -> &'static str {
        "ndarray"
    }

    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        let (m, k1) = a.dim();
        let (k2, n) = b.dim();
        assert_eq!(m, self.m, "A rows must equal m");
        assert_eq!(n, self.n, "B cols must equal n");
        assert_eq!(k1, self.k, "A cols must equal k");
        assert_eq!(k2, self.k, "B rows must equal k");
        a.dot(b)
    }
}

struct NdarrayBuilder;

impl MatmulImplBuilder for NdarrayBuilder {
    fn name(&self) -> &'static str { "ndarray" }
    fn build(&self, m: usize, k: usize, n: usize) -> Box<dyn MatmulImpl> {
        Box::new(NdarrayImpl { m, k, n })
    }
}



fn gen_random_matrix(rows: usize, cols: usize, rng: &mut StdRng) -> Array2<f32> {
    let mut data = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        // Uniform in [-1, 1)
        let v: f32 = rng.gen_range(-1.0f32..1.0f32);
        data.push(v);
    }
    Array2::from_shape_vec((rows, cols), data).expect("shape must match")
}

fn allclose(a: &Array2<f32>, b: &Array2<f32>, atol: f32, rtol: f32) -> bool {
    if a.raw_dim() != b.raw_dim() {
        return false;
    }
    let mut ok = true;
    // Iterate by rows for cache locality
    for (row_a, row_b) in a.axis_iter(Axis(0)).zip(b.axis_iter(Axis(0))) {
        for (&va, &vb) in row_a.iter().zip(row_b.iter()) {
            let diff = (va - vb).abs();
            let tol = atol + rtol * va.abs().max(vb.abs());
            if diff > tol {
                ok = false;
                break;
            }
        }
        if !ok {
            break;
        }
    }
    ok
}

fn parse_sizes_env() -> Vec<usize> {
    if let Ok(s) = env::var("MATMUL_SIZES") {
        let mut sizes = Vec::new();
        for part in s.split(',') {
            if let Ok(v) = part.trim().parse::<usize>() {
                if v > 0 {
                    sizes.push(v);
                }
            }
        }
        if !sizes.is_empty() {
            return sizes;
        }
    }
    // Default standard square sizes
    vec![64, 128, 256, 512]
}

fn parse_env_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_shapes_env() -> Option<Vec<(usize, usize, usize)>> {
    if let Ok(s) = env::var("MATMUL_SHAPES") {
        let mut shapes = Vec::new();
        for item in s.split(';') {
            let norm = item.replace('X', "x");
            let norm = norm.replace(',', "x");
            let parts: Vec<_> = norm.split('x').collect();
            if parts.len() == 3 {
                if let (Ok(m), Ok(k), Ok(n)) = (
                    parts[0].trim().parse::<usize>(),
                    parts[1].trim().parse::<usize>(),
                    parts[2].trim().parse::<usize>(),
                ) {
                    if m > 0 && k > 0 && n > 0 { shapes.push((m,k,n)); }
                }
            }
        }
        if !shapes.is_empty() { return Some(shapes); }
    }
    None
}

fn main() {
    // Configurable via environment
    let sizes = parse_sizes_env();
    let warmup = parse_env_usize("MATMUL_WARMUP", 10);
    let iters = parse_env_usize("MATMUL_ITERS", 50);

    let atol: f32 = env::var("MATMUL_ATOL").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-4);
    let rtol: f32 = env::var("MATMUL_RTOL").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-3);

    let builders: Vec<Box<dyn MatmulImplBuilder>> = vec![
            Box::new(NdarrayBuilder),
            Box::new(cranelift_0::CraneliftBuilder::new()),
            Box::new(cranelift_1::CraneliftBuilder::new()),
            Box::new(cranelift_2::CraneliftBuilder::new()),
          //  Box::new(cranelift_3::CraneliftBuilder::new()),
            Box::new(cranelift_4::CraneliftBuilder::new()),
            Box::new(cranelift_5::CraneliftBuilder::new()),
           // Box::new(cranelift_6::CraneliftBuilder::new()),
        ];

    println!("Matmul benchmark using ndarray as ground truth");
    println!(
        "Config: sizes={:?}, warmup={}, iters={}, atol={}, rtol={}",
        sizes, warmup, iters, atol, rtol
    );

    // One RNG per program run with fixed seed for reproducibility across sizes
    let mut rng = StdRng::seed_from_u64(0x5EED_u64);

    // Auto-generate per-implementation reports for 64x64 before benchmarks
    let report_dir = env::var("MATMUL_REPORT_DIR").unwrap_or_else(|_| "reports".to_string());
    let report_dir_path = std::path::Path::new(&report_dir);
    if let Err(e) = std::fs::create_dir_all(&report_dir_path) {
        eprintln!("Failed to create report dir {}: {}", report_dir, e);
    } else {
        for bldr in &builders {
            let mm = bldr.build(64, 64, 64);
            mm.make_report(report_dir_path);
        }
        println!("Wrote per-impl 64x64 reports to {}", report_dir);
    }

    // Storage for markdown reporting
    #[derive(Clone)]
    struct BenchRow {
        impl_name: String,
        m: usize,
        k: usize,
        n: usize,
        ms_per_iter: f64,
        gflops: f64,
        checksum: f64,
    }
    let mut rows: Vec<BenchRow> = Vec::new();

    let shapes: Vec<(usize,usize,usize)> = if let Some(v) = parse_shapes_env() { v } else { sizes.iter().map(|&n| (n,n,n)).collect() };
    for &(m, k, n) in &shapes {
        let a = gen_random_matrix(m, k, &mut rng);
        let b = gen_random_matrix(k, n, &mut rng);

        // Ground truth using ndarray
        let gt = a.dot(&b);

        let flops_per_mul = 2.0f64 * (m as f64) * (n as f64) * (k as f64);

        for bldr in &builders {
            let mm = bldr.build(m, k, n);
            // Validate correctness
            let out = mm.matmul(&a, &b);
            let valid = allclose(&out, &gt, atol, rtol);

            if !valid {
                eprintln!(
                    "Validation FAILED for impl={} size={}x{} (k={})",
                    mm.name(), m, n, k
                );
                std::process::exit(1);
            }

            // Warmup
            let mut checksum: f64 = 0.0;
            for _ in 0..warmup {
                let r = mm.matmul(&a, &b);
                checksum += r.sum() as f64;
            }

            // Timed runs
            let start = Instant::now();
            checksum = 0.0;
            for _ in 0..iters {
                let r = mm.matmul(&a, &b);
                checksum += r.sum() as f64;
            }
            let elapsed = start.elapsed();
            let secs = elapsed.as_secs_f64();
            let gflops = (flops_per_mul * (iters as f64)) / secs / 1e9;
            let ms_per_iter = (secs / (iters as f64)) * 1e3;

            println!(
                "impl={:<8} size={:>4}x{:<4} k={:<4} | time/iter={:>8.3} ms | throughput={:>7.2} GFLOP/s | checksum={:.3}",
                mm.name(), m, n, k, ms_per_iter, gflops, checksum
            );

            rows.push(BenchRow { impl_name: mm.name().to_string(), m, k, n, ms_per_iter, gflops, checksum });
        }
    }

    // Build Markdown report
    let report_path = env::var("MATMUL_REPORT").unwrap_or_else(|_| "matmul_report.md".to_string());
    let mut md = String::new();
    md.push_str("# Matmul Benchmark Report\n\n");
    md.push_str("## Configuration\n");
    md.push_str(&format!("- Sizes: {:?}\n- Warmup iterations: {}\n- Timed iterations: {}\n- ATOL: {}\n- RTOL: {}\n\n", sizes, warmup, iters, atol, rtol));

    // Summary table: GFLOP/s by implementation (rows) and shape (columns)
    md.push_str("## GFLOP/s by implementation and shape\n\n");

    // Collect unique shapes and implementations
    let mut shapes_set: std::collections::BTreeSet<(usize,usize,usize)> = std::collections::BTreeSet::new();
    let mut impls_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for r in &rows {
        shapes_set.insert((r.m, r.k, r.n));
        impls_set.insert(r.impl_name.clone());
    }
    let shapes_vec: Vec<(usize,usize,usize)> = shapes_set.into_iter().collect();
    let impls_vec: Vec<String> = impls_set.into_iter().collect();

    // Map (impl, shape) -> gflops
    let mut gmap: std::collections::HashMap<(String,(usize,usize,usize)), f64> = std::collections::HashMap::new();
    for r in &rows {
        gmap.insert((r.impl_name.clone(), (r.m, r.k, r.n)), r.gflops);
    }

    // Header row
    md.push_str("| Impl |");
    for (m,k,n) in &shapes_vec {
        md.push_str(&format!(" {}x{}x{} |", m,k,n));
    }
    md.push_str("\n");

    // Alignment row (right-align numeric columns)
    md.push_str("|------|");
    for _ in &shapes_vec { md.push_str("-----------:|"); }
    md.push_str("\n");

    // Data rows
    for impl_name in &impls_vec {
        md.push_str(&format!("| {} |", impl_name));
        for shape in &shapes_vec {
            if let Some(val) = gmap.get(&(impl_name.clone(), *shape)) {
                md.push_str(&format!(" {:>10.2} |", val));
            } else {
                md.push_str("     -      |");
            }
        }
        md.push_str("\n");
    }

    md.push_str("\n## Results\n\n");
    md.push_str("| Impl | M | K | N | Time/iter (ms) | Throughput (GFLOP/s) | Checksum |\n");
    md.push_str("|------|---:|---:|---:|---------------:|----------------------:|---------:|\n");

    // Sort rows for stable presentation
    rows.sort_by(|a, b| (a.m, a.k, a.n, a.impl_name.clone()).cmp(&(b.m, b.k, b.n, b.impl_name.clone())));

    for r in &rows {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {:.3} | {:.2} | {:.3} |\n",
            r.impl_name, r.m, r.k, r.n, r.ms_per_iter, r.gflops, r.checksum
        ));
    }

    // Optionally include a per-size summary (best GFLOP/s)
    use std::collections::HashMap;
    let mut best_by_shape: HashMap<(usize,usize,usize), &BenchRow> = HashMap::new();
    for r in &rows {
        best_by_shape.entry((r.m, r.k, r.n)).and_modify(|e| if r.gflops > e.gflops { *e = r }).or_insert(r);
    }
    md.push_str("\n## Best per shape (by GFLOP/s)\n\n");
    md.push_str("| M | K | N | Best Impl | Throughput (GFLOP/s) | Time/iter (ms) |\n");
    md.push_str("|---:|---:|---:|-----------|----------------------:|---------------:|\n");
    let mut keys: Vec<_> = best_by_shape.keys().cloned().collect();
    keys.sort();
    for key in keys {
        if let Some(r) = best_by_shape.get(&key) {
            md.push_str(&format!("| {} | {} | {} | {} | {:.2} | {:.3} |\n", r.m, r.k, r.n, r.impl_name, r.gflops, r.ms_per_iter));
        }
    }

    // Write report to file
    if let Err(e) = std::fs::write(&report_path, md.as_bytes()) {
        eprintln!("Failed to write markdown report to {}: {}", report_path, e);
    } else {
        println!("\nMarkdown report written to {}", report_path);
    }

    println!("Done.");
}

