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
    for (encoder, probability) in score.encoder_probabilities() {
        println!("{encoder}\t{probability:.7}");
    }
    Ok(())
}
