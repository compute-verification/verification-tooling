use cmake::Config;
use std::process::Command;

fn cuda_arch() -> String {
    let value = std::env::var("ICICLE_CUDA_ARCH").unwrap_or_else(|_| {
        let output = Command::new("nvidia-smi")
            .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
            .output()
            .expect("ICICLE requires nvidia-smi or an explicit ICICLE_CUDA_ARCH");
        assert!(output.status.success(), "nvidia-smi failed while detecting ICICLE_CUDA_ARCH");
        String::from_utf8(output.stdout)
            .expect("nvidia-smi returned non-UTF-8 output")
            .lines()
            .next()
            .expect("nvidia-smi returned no GPU compute capability")
            .replace('.', "")
    });
    assert!(
        !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()),
        "ICICLE_CUDA_ARCH must contain a CUDA architecture number such as 86"
    );
    value
}

fn main() {
    println!("cargo:rerun-if-env-changed=CXXFLAGS");
    println!("cargo:rerun-if-env-changed=ICICLE_CUDA_ARCH");
    println!("cargo:rerun-if-changed=../../../../icicle");

    // Base config
    let mut config = Config::new("../../../../icicle");
    config
        .define("BUILD_TESTS", "OFF")
        .define("CURVE", "bn254")
        .define("CUDA_ARCH", cuda_arch())
        .define("CMAKE_BUILD_TYPE", "Release");

    // Optional Features
    #[cfg(feature = "g2")]
    config.define("G2_DEFINED", "ON");

    #[cfg(feature = "ec_ntt")]
    config.define("ECNTT_DEFINED", "ON");

    #[cfg(feature = "devmode")]
    config.define("DEVMODE", "ON");

    // Build
    let out_dir = config
        .build_target("icicle")
        .build();

    println!("cargo:rustc-link-search={}/build", out_dir.display());

    println!("cargo:rustc-link-lib=ingo_bn254");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=cudart");
}
