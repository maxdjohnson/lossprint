use lossprint::Scanner;

fn main() -> lossprint::Result<()> {
    let Some(path) = std::env::args_os().nth(1) else {
        eprintln!("usage: cargo run --example score -- AUDIO_FILE");
        return Ok(());
    };

    let scanner = Scanner::new()?;
    let score = scanner.score_file(path)?;
    println!("transcode\t{:.7}", score.transcode_probability());
    for (codec, probability) in score.codec_probabilities().iter() {
        println!("{}\t{probability:.7}", codec.as_str());
    }
    Ok(())
}
