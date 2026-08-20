use lossprint::Scanner;
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(path) = std::env::args_os().nth(1) else {
        eprintln!("usage: cargo run --example score -- AUDIO_FILE");
        return Ok(());
    };

    let scanner = Scanner::new()?;
    let score = scanner.score(File::open(path)?)?;
    println!("transcode\t{:.7}", score.transcode_probability());
    for (codec, probability) in score.codec_probabilities() {
        println!("{codec}\t{probability:.7}");
    }
    Ok(())
}
