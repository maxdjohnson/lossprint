use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

const MODEL_PATH: &str = "model/model.onnx";
const MODEL_SHA256: &str = "33c74bde418b8330f7e67222afb2ab53706c136281bddd19ec0870b81ddce89a";

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
    println!("cargo:rerun-if-changed={MODEL_PATH}");

    let model =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set"))
            .join(MODEL_PATH);
    verify(&model).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "could not use the model at {}: {error}. Published crates include it; from a Git checkout, run `python tools/fetch_model.py` first",
                model.display()
            ),
        )
    })
}
