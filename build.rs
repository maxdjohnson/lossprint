use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::Command;

const MODEL_URL: &str =
    "https://huggingface.co/maxdj/lossprint/resolve/c6ca3dd209e39c21b8ce48e235f759ac931cf914/model.onnx?download=true";
const MODEL_VERSION: &str = "v0.7";
const MODEL_SHA256: &str = "33c74bde418b8330f7e67222afb2ab53706c136281bddd19ec0870b81ddce89a";

fn cache_path() -> PathBuf {
    PathBuf::from(env::var_os("CARGO_HOME").expect("CARGO_HOME is set"))
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

    if let Err(error) = verify(&temporary) {
        let _ = fs::remove_file(temporary);
        return Err(error);
    }

    match fs::rename(&temporary, destination) {
        Ok(()) => Ok(()),
        // Another build installed the same verified model first.
        Err(_) if verify(destination).is_ok() => {
            let _ = fs::remove_file(temporary);
            Ok(())
        }
        Err(error) => {
            let _ = fs::remove_file(temporary);
            Err(error)
        }
    }
}

fn verify(path: &Path) -> io::Result<()> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let actual = format!("{:x}", hasher.finalize());
    if actual != MODEL_SHA256 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "model checksum mismatch for {}: expected {MODEL_SHA256}, got {actual}",
                path.display()
            ),
        ));
    }
    Ok(())
}

fn main() -> io::Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=DOCS_RS");

    let destination =
        PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set")).join("model.onnx");
    if env::var_os("DOCS_RS").is_some() {
        fs::write(destination, [])?;
        return Ok(());
    }
    let cached = cache_path();
    if cached.exists() {
        verify(&cached)?;
    } else {
        download(&cached)?;
    }
    fs::copy(cached, destination)?;
    Ok(())
}
