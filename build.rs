use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

const MODEL_URL: &str =
    "https://huggingface.co/maxdj/lossprint/resolve/ae0526389bb0fa3d8cfed08ecfeae0bd7930a8c8/model.onnx?download=true";
const MODEL_VERSION: &str = "v0.5";

fn install(source: &Path, destination: &Path) -> io::Result<()> {
    if destination.exists() {
        fs::remove_file(destination)?;
    }
    if fs::hard_link(source, destination).is_err() {
        fs::copy(source, destination)?;
    }
    Ok(())
}

fn cache_path() -> PathBuf {
    let cargo_home = env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".cargo")))
        .or_else(|| env::var_os("USERPROFILE").map(|home| PathBuf::from(home).join(".cargo")))
        .unwrap_or_else(env::temp_dir);
    cargo_home
        .join("lossprint")
        .join("models")
        .join(MODEL_VERSION)
        .join("model.onnx")
}

fn download(destination: &Path) -> io::Result<()> {
    let parent = destination.parent().expect("cache path has a parent");
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!("model.onnx.download.{}", std::process::id()));
    let _ = fs::remove_file(&temporary);

    eprintln!("lossprint: downloading model from {MODEL_URL}");
    let status = Command::new("curl")
        .args([
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--output",
        ])
        .arg(&temporary)
        .arg(MODEL_URL)
        .status()?;
    if !status.success() {
        let _ = fs::remove_file(temporary);
        return Err(io::Error::other(format!(
            "could not download model from {MODEL_URL}"
        )));
    }

    match fs::rename(&temporary, destination) {
        Ok(()) => Ok(()),
        Err(_) if destination.exists() => {
            fs::remove_file(temporary)?;
            Ok(())
        }
        Err(error) => Err(error),
    }
}

fn main() -> io::Result<()> {
    println!("cargo:rerun-if-env-changed=LOSSPRINT_MODEL_PATH");

    let destination =
        PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set")).join("model.onnx");
    if let Some(source) = env::var_os("LOSSPRINT_MODEL_PATH") {
        install(&PathBuf::from(source), &destination)?;
        return Ok(());
    }

    let cached = cache_path();
    if !cached.exists() {
        download(&cached)?;
    }
    install(&cached, &destination)?;
    Ok(())
}
