// Common reporting utilities for Cranelift JIT'd functions.
//
// Provides:
// - write_cranelift_report: writes a Markdown report with CLIF and full machine-code disassembly
// - x86_64-only disassembly via iced-x86
//
// Note: This expects cranelift-jit 0.111's JITModule API to expose
//   - get_finalized_function(func_id) -> *const u8
//   - get_finalized_function_size(func_id) -> usize
//
// If you're on a different arch, we emit a placeholder note.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use cranelift_module::FuncId;
use cranelift_jit::JITModule;

#[cfg(target_arch = "x86_64")]
fn disassemble_x86_64(ptr: *const u8, len: usize, base_addr: u64) -> String {
    use iced_x86::{Decoder, DecoderOptions, Formatter, IntelFormatter, Instruction};
    if ptr.is_null() || len == 0 {
        return "<no code bytes available>".to_string();
    }

    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    let mut decoder = Decoder::with_ip(64, bytes, base_addr, DecoderOptions::NONE);
    let mut formatter = IntelFormatter::new();

    // Configure formatter to show RIP-relative addresses etc.
    formatter.options_mut().set_uppercase_hex(false);
    formatter.options_mut().set_space_after_operand_separator(true);
    formatter
        .options_mut()
        .set_first_operand_char_index(10); // Indent operands for nicer columns

    let mut out = String::new();
    let mut instr = Instruction::default();

    while decoder.can_decode() {
        decoder.decode_out(&mut instr);
        // Format
        let mut text = String::new();
        formatter.format(&instr, &mut text);

        // Show bytes
        let start = (instr.ip() - base_addr) as usize;
        let end = (instr.next_ip() - base_addr) as usize;
        let slice = &bytes[start..end];
        let bytes_str = slice
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<Vec<_>>()
            .join(" ");

        // Address, bytes, formatted instruction
        out.push_str(&format!("{:#018x}:  {:<32}  {}\n", instr.ip(), bytes_str, text));
    }

    out
}

#[cfg(not(target_arch = "x86_64"))]
fn disassemble_x86_64(_ptr: *const u8, _len: usize, _base_addr: u64) -> String {
    "Disassembly currently implemented only for x86_64 targets.".to_string()
}


// Tail-trimming heuristic to avoid decoding into padding.
// Environment knobs:
// - REPORT_MIN_PREFIX: minimum leading bytes to keep before heuristics (default: 64)
// - REPORT_ZERO_RUN:  cut at first run of this many 0x00 bytes (default: 32)
// - REPORT_NOP_RUN:   cut at first run of this many 0x90 bytes (default: 48)
fn trim_code_len(bytes: &[u8]) -> usize {
    let len = bytes.len();
    if len == 0 {
        return 0;
    }

    let min_prefix = std::env::var("REPORT_MIN_PREFIX").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(64);
    let zero_run = std::env::var("REPORT_ZERO_RUN").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(32);
    let nop_run  = std::env::var("REPORT_NOP_RUN").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(48);

    let mut cut = len;

    if len > min_prefix {
        // Look for a block of zero bytes
        if zero_run > 0 && zero_run <= len {
            let mut i = min_prefix;
            while i + zero_run <= len {
                if bytes[i..i + zero_run].iter().all(|&b| b == 0x00) {
                    cut = i;
                    break;
                }
                i += 1;
            }
        }

        // If no zero block found, look for a long run of single-byte NOPs (0x90)
        if cut == len && nop_run > 0 {
            let mut i = min_prefix;
            let mut run = 0usize;
            while i < len {
                if bytes[i] == 0x90 {
                    run += 1;
                    if run >= nop_run {
                        cut = i + 1 - run;
                        break;
                    }
                } else {
                    run = 0;
                }
                i += 1;
            }
        }
    }

    cut
}

// Returns a best-effort (ptr, size) pair for the finalized function.
// - Uses JITModule::get_finalized_function for ptr.
// - Bounded by current memory region via `region`.
// - Then trimmed by a simple tail heuristic (see trim_code_len).
// - Window is capped by REPORT_MAX_BYTES (default: 4096).
fn get_finalized_code_ptr_and_size(module: &cranelift_jit::JITModule, _func_id: cranelift_module::FuncId) -> Option<(*const u8, usize)> {
    let ptr = module.get_finalized_function(_func_id);
    if ptr.is_null() {
        return None;
    }

    let cap = std::env::var("REPORT_MAX_BYTES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(4096);

    // Safety: querying VMA metadata only.
    let region_info = match region::query(ptr) {
        Ok(r) => r,
        Err(_) => {
            // Fallback: use cap and still apply trimming on that window
            let size = cap;
            let bytes = unsafe { std::slice::from_raw_parts(ptr, size) };
            let trimmed = trim_code_len(bytes).max(1);
            return Some((ptr, trimmed.min(size)));
        }
    };

    let region_base = region_info.as_ptr() as *const u8;
    let region_len = region_info.len();

    let start_addr = ptr as usize;
    let base_addr = region_base as usize;

    let offset_in_region = start_addr.saturating_sub(base_addr);
    let max_in_region = region_len.saturating_sub(offset_in_region);

    let size = cap.min(max_in_region);
    if size == 0 {
        return None;
    }

    // Apply heuristic trimming within the bounded window
    let bytes = unsafe { std::slice::from_raw_parts(ptr, size) };
    let trimmed = trim_code_len(bytes);
    let final_size = trimmed.clamp(1, size);

    Some((ptr, final_size))
}





pub fn write_cranelift_report(
    out_dir: &std::path::Path,
    impl_name: &str,
    m: usize,
    k: usize,
    n: usize,
    clif_text: &str,
    module: &JITModule,
    func_id: cranelift_module::FuncId,
) {
    use std::fs;
    use std::io::Write;
    use std::path::PathBuf;

    let file_name = format!("{}_{}x{}x{}.md", impl_name, m, k, n);
    let mut path = PathBuf::from(out_dir);
    path.push(file_name);

    let (code_ptr, code_size) = match get_finalized_code_ptr_and_size(module, func_id) {
        Some(v) => v,
        None => {
            let _ = fs::create_dir_all(out_dir);
            let mut f = fs::File::create(&path).expect("create report file");
            let _ = writeln!(f, "# Cranelift Report: {} ({}x{}x{})", impl_name, m, k, n);
            let _ = writeln!(f, "");
            let _ = writeln!(f, "- Target triple: {}", cranelift_native::builder().unwrap().triple());
            let _ = writeln!(f, "");
            let _ = writeln!(f, "## Cranelift IR (CLIF)");
            let _ = writeln!(f, "");
            let _ = writeln!(f, "```clifir");
            let _ = writeln!(f, "{}", clif_text);
            let _ = writeln!(f, "```");
            let _ = writeln!(f, "");
            let _ = writeln!(f, "## Machine-code Disassembly");
            let _ = writeln!(f, "");
            let _ = writeln!(f, "_Code bytes unavailable (no public size API in cranelift-jit); using CLIF only._");
            return;
        }
    };

    let base_addr = code_ptr as u64;
    let disasm = disassemble_x86_64(code_ptr, code_size, base_addr);

    let _ = fs::create_dir_all(out_dir);
    let mut f = fs::File::create(&path).expect("create report file");
    writeln!(f, "# Cranelift Report: {} ({}x{}x{})", impl_name, m, k, n).unwrap();
    writeln!(f, "").unwrap();
    writeln!(f, "- Target triple: {}", cranelift_native::builder().unwrap().triple()).unwrap();
    writeln!(f, "- Code address: {:#x}", base_addr).unwrap();
    writeln!(f, "- Disassembly window: {} bytes (bounded by REPORT_MAX_BYTES and current memory region)", code_size).unwrap();

    writeln!(f, "").unwrap();
    writeln!(f, "## Cranelift IR (CLIF)").unwrap();
    writeln!(f, "").unwrap();
    writeln!(f, "```clifir").unwrap();
    write!(f, "{}", clif_text).unwrap();
    writeln!(f, "```").unwrap();

    writeln!(f, "").unwrap();
    writeln!(f, "## Machine-code Disassembly").unwrap();
    writeln!(f, "").unwrap();
    writeln!(f, "```text").unwrap();
    write!(f, "{}", disasm).unwrap();
    writeln!(f, "```").unwrap();
}

