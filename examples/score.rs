use lossprint::Scanner;

fn main() -> lossprint::Result<()> {
    let Some(path) = std::env::args_os().nth(1) else {
        eprintln!("usage: cargo run --example score -- AUDIO_FILE");
        return Ok(());
    };

    let mut scanner = Scanner::new()?;
    let score = scanner.score_file(path)?;
    println!("transcode\t{:.7}", score.prob_transcode);
    println!("mp3\t{:.7}", score.prob_codec.mp3);
    println!("aac\t{:.7}", score.prob_codec.aac);
    println!("aac_at\t{:.7}", score.prob_codec.aac_at);
    println!("fdk_aac\t{:.7}", score.prob_codec.fdk_aac);
    println!("vorbis\t{:.7}", score.prob_codec.vorbis);
    println!("opus\t{:.7}", score.prob_codec.opus);
    Ok(())
}
