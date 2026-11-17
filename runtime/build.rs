//! Build script detecting whether the active `rustc` supports stabilized x86 AVX-512 intrinsics (`>= 1.89.0`).

use std::process::Command;

fn parse_rustc_minor(version_stdout: &str) -> Option<(u32, u32)> {
    // Expected format: `rustc 1.85.1 (4eb161250 2025-03-15)` or `rustc 1.89.0-nightly ...`
    let ver_token = version_stdout.split_whitespace().nth(1)?;
    let core_ver = ver_token.split('-').next()?;
    let mut parts = core_ver.split('.');
    let major: u32 = parts.next()?.parse().ok()?;
    let minor: u32 = parts.next()?.parse().ok()?;
    Some((major, minor))
}

fn main() {
    println!("cargo::rustc-check-cfg=cfg(terra_has_stable_avx512)");
    println!("cargo:rerun-if-env-changed=RUSTC");

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    if let Ok(output) = Command::new(rustc).arg("--version").output() {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Some((major, minor)) = parse_rustc_minor(&stdout) {
                if major > 1 || (major == 1 && minor >= 89) {
                    println!("cargo:rustc-cfg=terra_has_stable_avx512");
                }
            }
        }
    }
}
