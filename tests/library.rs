#[test]
fn supported_extensions_are_case_insensitive() {
    for path in ["track.wav", "track.AIF", "track.aiff", "track.FLAC"] {
        assert!(lossprint::has_supported_extension(path));
    }
    assert!(!lossprint::has_supported_extension("track.mp3"));
}
