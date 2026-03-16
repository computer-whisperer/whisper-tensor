#![allow(
    clippy::all,
    dead_code,
    unreachable_patterns,
    unused_variables,
    unused_imports
)]
//! creative3: Load-optimal agglomerative clustering + SA input assignment.
//!
//! A genuinely different approach that directly optimizes the cost metric at
//! every step:
//!
//! 1. Extract output chains and input dependencies.
//!
//! 2. **Load-driven agglomerative clustering**: Start with each chain as its
//!    own group. Maintain a priority queue of (group_i, group_j, merge_benefit)
//!    where merge_benefit = reduction in total loads from merging i and j.
//!    Greedily merge the pair with highest benefit until we have k groups.
//!    Unlike Jaccard (which is a similarity proxy), this directly measures
//!    how many loads are saved by co-locating two groups.
//!
//! 3. **SA polish on input assignment**: After chain groups are fixed, SA
//!    on input-to-kernel assignment to squeeze out remaining waste.
//!
//! Key insight: two chains sharing rare inputs (low fan-out) should merge first
//! because co-locating them eliminates loads for inputs that few other kernels
//! share. Two chains sharing common inputs (high fan-out) benefit less because
//! those inputs are needed by many kernels regardless. The merge benefit
//! naturally captures this -- it counts how many unique (input, kernel) load
//! pairs disappear when two groups merge.

use super::simple_dag::*;
use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap, HashMap, HashSet};

pub struct Creative3Partitioner;

impl SimplePartitioner for Creative3Partitioner {
    fn partition(&self, dag: &SimpleDag, hw: &HardwareConfig) -> SimplePartition {
        let n = dag.ops.len();
        if n == 0 {
            return SimplePartition { kernels: vec![] };
        }

        let (chains, chain_deps) = extract_chains(dag);
        if chains.is_empty() {
            return SimplePartition {
                kernels: vec![SimpleKernel {
                    ops: (0..n as u32).collect(),
                }],
            };
        }

        let k = hw.parallelism.min(chains.len()).max(1) as u32;
        if k <= 1 {
            return SimplePartition {
                kernels: vec![SimpleKernel {
                    ops: (0..n as u32).collect(),
                }],
            };
        }

        let info = InputInfo::build(&chain_deps);

        // Phase 1: Agglomerative clustering by merge benefit
        let chain_groups = agglomerative_cluster(&info, chains.len(), k);

        // Phase 2: Greedy + SA input assignment
        let input_assign = optimize_input_assignment(&info, &chain_groups, k);

        assemble(dag, &chains, &info, &chain_groups, &input_assign)
    }
}

// ─── Helper methods on SimpleDag ─────────────────────────────────────────────

/// Returns the list of op indices that are DAG outputs.
fn output_ops(dag: &SimpleDag) -> Vec<u32> {
    dag.outputs.clone()
}

/// Returns true if this op kind is a "source" (Input or Literal — no inputs).
fn is_source(kind: &OpKind) -> bool {
    matches!(kind, OpKind::Input | OpKind::Literal)
}

/// Compute consumers for each op: consumers[i] = list of ops that use op i as input.
fn consumers(dag: &SimpleDag) -> Vec<Vec<u32>> {
    let n = dag.ops.len();
    let mut cons = vec![Vec::new(); n];
    for (i, op) in dag.ops.iter().enumerate() {
        for &inp in &op.inputs {
            cons[inp as usize].push(i as u32);
        }
    }
    cons
}

// ─── Chain extraction ───────────────────────────────────────────────────────

struct OutputChain {
    ops: BTreeSet<u32>,
}

fn extract_chains(dag: &SimpleDag) -> (Vec<OutputChain>, Vec<BTreeSet<u32>>) {
    let fo = dag.fan_out();
    let mut chains = Vec::new();
    let mut deps = Vec::new();
    for &root in &output_ops(dag) {
        if is_source(&dag.ops[root as usize].kind) {
            continue;
        }
        let mut ops = BTreeSet::new();
        let mut inp = BTreeSet::new();
        let mut stk = vec![root];
        while let Some(cur) = stk.pop() {
            if !ops.insert(cur) {
                continue;
            }
            for &i in &dag.ops[cur as usize].inputs {
                if is_source(&dag.ops[i as usize].kind) {
                    inp.insert(i);
                } else if fo[i as usize] == 1 {
                    stk.push(i);
                } else {
                    gather_inputs(dag, i, &mut inp);
                }
            }
        }
        chains.push(OutputChain { ops });
        deps.push(inp);
    }
    (chains, deps)
}

fn gather_inputs(dag: &SimpleDag, start: u32, out: &mut BTreeSet<u32>) {
    let mut stk = vec![start];
    let mut vis = HashSet::new();
    while let Some(cur) = stk.pop() {
        if !vis.insert(cur) {
            continue;
        }
        if is_source(&dag.ops[cur as usize].kind) {
            out.insert(cur);
        } else {
            for &i in &dag.ops[cur as usize].inputs {
                stk.push(i);
            }
        }
    }
}

// ─── Input info ─────────────────────────────────────────────────────────────

struct InputInfo {
    inputs: Vec<u32>,
    used_by_chains: Vec<Vec<usize>>,
    chain_inputs: Vec<Vec<usize>>,
}

impl InputInfo {
    fn build(chain_deps: &[BTreeSet<u32>]) -> Self {
        let mut all: BTreeSet<u32> = BTreeSet::new();
        for d in chain_deps {
            all.extend(d);
        }
        let inputs: Vec<u32> = all.into_iter().collect();
        let idx: HashMap<u32, usize> = inputs.iter().enumerate().map(|(i, &o)| (o, i)).collect();
        let mut ub: Vec<Vec<usize>> = vec![Vec::new(); inputs.len()];
        let mut ci: Vec<Vec<usize>> = Vec::with_capacity(chain_deps.len());
        for (c, d) in chain_deps.iter().enumerate() {
            let mut v: Vec<usize> = d.iter().filter_map(|o| idx.get(o).copied()).collect();
            v.sort();
            for &ii in &v {
                ub[ii].push(c);
            }
            ci.push(v);
        }
        InputInfo {
            inputs,
            used_by_chains: ub,
            chain_inputs: ci,
        }
    }
}

// ─── Agglomerative clustering ───────────────────────────────────────────────

/// Union-Find for tracking group membership.
struct UF {
    parent: Vec<usize>,
    rank: Vec<usize>,
    size: Vec<usize>,
}

impl UF {
    fn new(n: usize) -> Self {
        UF {
            parent: (0..n).collect(),
            rank: vec![0; n],
            size: vec![1; n],
        }
    }
    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }
    fn union(&mut self, a: usize, b: usize) -> usize {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return ra;
        }
        let (big, small) = if self.rank[ra] >= self.rank[rb] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small] = big;
        self.size[big] += self.size[small];
        if self.rank[big] == self.rank[small] {
            self.rank[big] += 1;
        }
        big
    }
}

#[derive(Clone)]
struct MergeCandidate {
    g1: usize,
    g2: usize,
    benefit: i32, // positive = good merge (reduces loads)
}

impl PartialEq for MergeCandidate {
    fn eq(&self, o: &Self) -> bool {
        self.benefit == o.benefit
    }
}
impl Eq for MergeCandidate {}
impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for MergeCandidate {
    fn cmp(&self, o: &Self) -> Ordering {
        self.benefit.cmp(&o.benefit)
    }
}

/// Cluster chains into k groups by greedily merging the pair with highest
/// load-reduction benefit.
fn agglomerative_cluster(info: &InputInfo, nc: usize, k: u32) -> Vec<u32> {
    if nc <= k as usize {
        return (0..nc as u32).collect();
    }

    let ni = info.inputs.len();

    // For each input, track which groups use it (initially each chain is its own group).
    // input_groups[ii] = set of groups that contain chains using input ii
    let mut input_groups: Vec<HashSet<usize>> = vec![HashSet::new(); ni];
    for (ii, users) in info.used_by_chains.iter().enumerate() {
        for &ci in users {
            input_groups[ii].insert(ci);
        }
    }

    // Compute shared input count between all pairs of groups that share at least one input.
    // shared[i][j] = number of inputs shared between group i and group j
    let mut shared: HashMap<(usize, usize), u32> = HashMap::new();
    for (ii, groups) in input_groups.iter().enumerate() {
        let gs: Vec<usize> = groups.iter().copied().collect();
        for (a, &gi) in gs.iter().enumerate() {
            for &gj in &gs[a + 1..] {
                let key = if gi < gj { (gi, gj) } else { (gj, gi) };
                *shared.entry(key).or_default() += 1;
            }
        }
    }

    let mut uf = UF::new(nc);
    let mut num_groups = nc;

    // Build priority queue
    let mut heap: BinaryHeap<MergeCandidate> = BinaryHeap::new();
    for (&(g1, g2), &count) in &shared {
        heap.push(MergeCandidate {
            g1,
            g2,
            benefit: count as i32,
        });
    }

    // Group inputs: for each group root, which input indices it contains
    let mut group_inputs: Vec<HashSet<usize>> = Vec::with_capacity(nc);
    for ci in 0..nc {
        let iset: HashSet<usize> = info.chain_inputs[ci].iter().copied().collect();
        group_inputs.push(iset);
    }

    // Max group size: don't let any group exceed 2x the average
    let max_size = (nc / k as usize) * 2 + 2;

    while num_groups > k as usize {
        let cand = match heap.pop() {
            Some(c) => c,
            None => break,
        };

        let r1 = uf.find(cand.g1);
        let r2 = uf.find(cand.g2);
        if r1 == r2 {
            continue;
        } // already merged

        // Check if this candidate is stale (groups have been merged with others)
        // We need to recompute the benefit if the groups have changed.
        if r1 != cand.g1 || r2 != cand.g2 {
            // Recompute: shared inputs between the actual groups r1, r2
            let shared_count = group_inputs[r1].intersection(&group_inputs[r2]).count();
            if shared_count > 0 {
                heap.push(MergeCandidate {
                    g1: r1,
                    g2: r2,
                    benefit: shared_count as i32,
                });
            }
            continue;
        }

        // Check size constraint
        if uf.size[r1] + uf.size[r2] > max_size {
            continue;
        }

        // Merge
        let new_root = uf.union(r1, r2);
        let other = if new_root == r1 { r2 } else { r1 };

        // Merge input sets
        let other_inputs: HashSet<usize> = group_inputs[other].clone();
        group_inputs[new_root].extend(other_inputs);

        num_groups -= 1;

        // Add new merge candidates between merged group and its neighbors
        // Find all groups that share inputs with the merged group
        let mut neighbor_shared: HashMap<usize, u32> = HashMap::new();
        for &ii in &group_inputs[new_root] {
            for &gi in &input_groups[ii] {
                let ri = uf.find(gi);
                if ri != new_root {
                    *neighbor_shared.entry(ri).or_default() += 1;
                }
            }
        }

        // Update input_groups: replace old roots with new root
        for &ii in &group_inputs[new_root] {
            input_groups[ii].remove(&r1);
            input_groups[ii].remove(&r2);
            input_groups[ii].insert(new_root);
        }

        for (neighbor, count) in neighbor_shared {
            heap.push(MergeCandidate {
                g1: new_root,
                g2: neighbor,
                benefit: count as i32,
            });
        }
    }

    // Map chains to group IDs
    let mut group_map: HashMap<usize, u32> = HashMap::new();
    let mut next_group = 0u32;
    let mut chain_groups = vec![0u32; nc];
    for ci in 0..nc {
        let root = uf.find(ci);
        let gid = *group_map.entry(root).or_insert_with(|| {
            let g = next_group;
            next_group += 1;
            g
        });
        chain_groups[ci] = gid;
    }

    // If we have fewer than k groups (due to max_size constraint), split largest
    while next_group < k {
        // Find the largest group
        let mut groups_by_id: HashMap<u32, Vec<usize>> = HashMap::new();
        for (ci, &gid) in chain_groups.iter().enumerate() {
            groups_by_id.entry(gid).or_default().push(ci);
        }
        let mut largest_group = 0u32;
        let mut largest_size = 0;
        for (&gid, chains) in &groups_by_id {
            if chains.len() > largest_size {
                largest_size = chains.len();
                largest_group = gid;
            }
        }
        if largest_size <= 1 {
            break;
        }

        // Split largest group in half
        let members = &groups_by_id[&largest_group];
        let mid = members.len() / 2;
        let new_gid = next_group;
        next_group += 1;
        for &ci in &members[mid..] {
            chain_groups[ci] = new_gid;
        }
    }

    chain_groups
}

// ─── Input assignment optimization ──────────────────────────────────────────

fn optimize_input_assignment(info: &InputInfo, chain_groups: &[u32], k: u32) -> Vec<u32> {
    let ni = info.inputs.len();
    let nc = chain_groups.len();

    // Build need[ii * k + kernel]
    let mut need = vec![0u32; ni * k as usize];
    for (ci, iis) in info.chain_inputs.iter().enumerate() {
        let ck = chain_groups[ci] as usize;
        if ck < k as usize {
            for &ii in iis {
                need[ii * k as usize + ck] += 1;
            }
        }
    }

    // Greedy: assign each input to the kernel with highest need
    let mut inp_k = vec![0u32; ni];
    for ii in 0..ni {
        let base = ii * k as usize;
        let (bk, _) = (0..k)
            .map(|kk| (kk, need[base + kk as usize]))
            .max_by_key(|x| x.1)
            .unwrap();
        inp_k[ii] = bk;
    }

    let compute_loads = |ik: &[u32]| -> i64 {
        let mut loads = 0i64;
        for ii in 0..ni {
            let base = ii * k as usize;
            let ok = ik[ii] as usize;
            for kk in 0..k as usize {
                if kk != ok && need[base + kk] > 0 {
                    loads += 1;
                }
            }
        }
        loads
    };

    let mut total_loads = compute_loads(&inp_k);
    let mut best_loads = total_loads;
    let mut best_ik = inp_k.clone();

    // SA polish
    let mut rng: u64 = 42;
    let mut rand = || -> u64 {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        rng >> 33
    };

    let iters = (ni * 200).min(300_000);
    let t0 = (total_loads as f64 * 0.2).max(3.0);

    for it in 0..iters {
        let temp = t0 * (1.0 - it as f64 / iters as f64);
        if temp < 0.001 {
            break;
        }

        let ii = (rand() as usize) % ni;
        if k <= 1 {
            break;
        }
        let ok = inp_k[ii];
        let nk = (ok + 1 + (rand() as u32) % (k - 1)) % k;
        let base = ii * k as usize;

        let mut delta = 0i32;
        for kk in 0..k as usize {
            if need[base + kk] > 0 {
                if kk != ok as usize {
                    delta -= 1;
                }
                if kk != nk as usize {
                    delta += 1;
                }
            }
        }

        if accept_sa(delta, temp, &mut rand) {
            inp_k[ii] = nk;
            total_loads += delta as i64;
        }
        if total_loads < best_loads {
            best_loads = total_loads;
            best_ik = inp_k.clone();
        }
    }

    best_ik
}

fn accept_sa(delta: i32, temp: f64, rng: &mut impl FnMut() -> u64) -> bool {
    if delta <= 0 {
        return true;
    }
    let p = (-delta as f64 / temp).exp();
    (rng() % 10000) as f64 / 10000.0 < p
}

// ─── Assemble ───────────────────────────────────────────────────────────────

fn assemble(
    dag: &SimpleDag,
    chains: &[OutputChain],
    info: &InputInfo,
    chain_groups: &[u32],
    input_assign: &[u32],
) -> SimplePartition {
    let n = dag.ops.len();
    let mut kof: Vec<Option<u32>> = vec![None; n];
    for (ci, ch) in chains.iter().enumerate() {
        for &op in &ch.ops {
            kof[op as usize] = Some(chain_groups[ci]);
        }
    }
    for (ii, &op) in info.inputs.iter().enumerate() {
        kof[op as usize] = Some(input_assign[ii]);
    }
    let cons = consumers(dag);
    let mut changed = true;
    while changed {
        changed = false;
        for i in (0..n).rev() {
            if kof[i].is_some() {
                continue;
            }
            let mut votes: HashMap<u32, u32> = HashMap::new();
            for &c in &cons[i] {
                if let Some(k) = kof[c as usize] {
                    *votes.entry(k).or_default() += 1;
                }
            }
            if let Some((&bk, _)) = votes.iter().max_by_key(|&(_, &v)| v) {
                kof[i] = Some(bk);
                changed = true;
            }
        }
        for i in 0..n {
            if kof[i].is_some() {
                continue;
            }
            for &inp in &dag.ops[i].inputs {
                if let Some(k) = kof[inp as usize] {
                    kof[i] = Some(k);
                    changed = true;
                    break;
                }
            }
        }
    }
    for k in kof.iter_mut() {
        if k.is_none() {
            *k = Some(0);
        }
    }

    // Remap to contiguous kernel IDs and build SimplePartition
    let mut seen = HashMap::new();
    let mut next = 0u32;
    let kernel_of: Vec<u32> = kof
        .into_iter()
        .map(|k| {
            let kid = k.unwrap_or(0);
            let e = *seen.entry(kid).or_insert_with(|| {
                let id = next;
                next += 1;
                id
            });
            e
        })
        .collect();

    // Build kernel op lists
    let num_kernels = next as usize;
    let mut kernel_ops: Vec<Vec<u32>> = vec![Vec::new(); num_kernels];
    for (i, &kid) in kernel_of.iter().enumerate() {
        kernel_ops[kid as usize].push(i as u32);
    }

    SimplePartition {
        kernels: kernel_ops
            .into_iter()
            .map(|ops| SimpleKernel { ops })
            .collect(),
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn run_test(name: &str, dag: &SimpleDag) -> SimpleCost {
        let hw = HardwareConfig::default();
        let partition = Creative3Partitioner.partition(dag, &hw);
        let cost = evaluate_cost(dag, &partition, &hw);
        println!("=== {} ===", name);
        println!("  ops: {} | {}", dag.ops.len(), cost);
        for (i, kernel) in partition.kernels.iter().enumerate() {
            if kernel.ops.len() <= 25 {
                let kinds: Vec<&str> = kernel
                    .ops
                    .iter()
                    .map(|&o| match dag.ops[o as usize].kind {
                        OpKind::Input => "In",
                        OpKind::Literal => "Lit",
                        OpKind::Add => "+",
                        OpKind::Mul => "*",
                        OpKind::Neg => "-",
                        OpKind::Tanh => "T",
                        _ => "?",
                    })
                    .collect();
                println!("  k{}: {} ops {:?}", i, kernel.ops.len(), kinds);
            } else {
                println!("  k{}: {} ops", i, kernel.ops.len());
            }
        }
        println!();
        cost
    }

    #[test]
    fn test_elementwise_add() {
        let dag = build_elementwise_add(8);
        let c = run_test("elementwise_add", &dag);
        assert!(c.num_kernels >= 1);
        assert_eq!(c.total_ops, 8); // 8 Add ops
    }

    #[test]
    fn test_unary_chain() {
        let dag = build_unary_chain(8, &[OpKind::Exp, OpKind::Neg, OpKind::Tanh]);
        let c = run_test("unary_chain", &dag);
        assert!(c.num_kernels >= 1);
        assert_eq!(c.total_ops, 8 * 3); // 3 unary stages * 8 elements
    }

    #[test]
    fn test_matmul_small() {
        let dag = build_matmul(4, 8, 16);
        let c = run_test("matmul(4,8,16)", &dag);
        assert!(c.num_kernels > 1, "got {}", c.num_kernels);
        assert!(c.num_kernels < 64, "got {}", c.num_kernels);
        println!("  >>> LOADS: {}", c.total_loads);
    }

    #[test]
    fn test_matmul_activation() {
        let dag = build_matmul_activation(2, 3, 2, OpKind::Tanh);
        let c = run_test("matmul_activation(2,3,2)", &dag);
        assert!(c.num_kernels >= 1);
    }

    #[test]
    fn test_matmul_chain() {
        let dag = build_matmul_chain(2, 3, 4, 2);
        let c = run_test("matmul_chain", &dag);
        assert!(c.num_kernels >= 1);
    }

    #[test]
    fn test_parallel_matmuls() {
        let dag = build_parallel_matmuls(4, 8, 16, 8, 16);
        let c = run_test("parallel_matmuls", &dag);
        assert!(c.num_kernels >= 2, "got {}", c.num_kernels);
    }

    #[test]
    fn test_shared_input_matmuls() {
        let dag = build_shared_input_matmuls(4, 8, 16, 16);
        let c = run_test("shared_input_matmuls", &dag);
        let fo = dag.fan_out();
        // X[0,0] fans out to Dq + Dk
        assert!(fo[0] >= 2, "idx=0 fo={}", fo[0]);
        assert!(c.num_kernels >= 1);
    }

    #[test]
    fn test_qkv_projections() {
        let dag = build_qkv_projections(4, 8, 16);
        let c = run_test("qkv_projections", &dag);
        let fo = dag.fan_out();
        // X[0,0] fans out to 3 * d_head
        assert!(fo[0] >= 6, "idx=0 fo={}", fo[0]);
        assert!(c.num_kernels >= 1);
        println!(
            "  QKV: {} kernels, {} transfers",
            c.num_kernels, c.total_loads
        );
    }

    #[test]
    fn test_matmul_tiling_structure() {
        let dag = build_matmul(4, 4, 4);
        let c = run_test("matmul_tiling(4,4,4)", &dag);
        println!("  4x4x4: k={}, loads={}", c.num_kernels, c.total_loads);
    }

    #[test]
    fn test_larger_matmul() {
        let dag = build_matmul(8, 16, 8);
        let c = run_test("matmul(8,16,8)", &dag);
        println!("  8x16x8: k={}, loads={}", c.num_kernels, c.total_loads);
    }

    #[test]
    fn test_partition_validity() {
        let cases: Vec<(&str, SimpleDag)> = vec![
            ("add", build_elementwise_add(8)),
            (
                "chain",
                build_unary_chain(8, &[OpKind::Exp, OpKind::Neg, OpKind::Tanh]),
            ),
            ("matmul", build_matmul(2, 3, 2)),
            ("act", build_matmul_activation(2, 3, 2, OpKind::Tanh)),
            ("chain_mm", build_matmul_chain(2, 3, 4, 2)),
            ("par", build_parallel_matmuls(4, 8, 16, 8, 16)),
            ("shared", build_shared_input_matmuls(4, 8, 16, 16)),
            ("qkv", build_qkv_projections(4, 8, 16)),
        ];
        let hw = HardwareConfig::default();
        for (name, dag) in &cases {
            let p = Creative3Partitioner.partition(dag, &hw);
            let errors = p.validate(dag);
            assert!(errors.is_empty(), "{}: {:?}", name, errors);
            let cost = evaluate_cost(dag, &p, &hw);
            assert_eq!(
                cost.total_ops,
                dag.ops
                    .iter()
                    .filter(|op| !matches!(op.kind, OpKind::Input | OpKind::Literal))
                    .count() as u64,
                "{}",
                name
            );
        }
    }
}
