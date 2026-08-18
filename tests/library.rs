use lossprint::{CodecProbabilities, ErrorKind, Scanner, TrackScore, MAX_BATCH_TRACKS};

#[test]
fn score_is_plain_probability_data() {
    let score = TrackScore {
        prob_transcode: 0.75,
        prob_codec: CodecProbabilities {
            mp3: 0.1,
            aac: 0.2,
            aac_at: 0.3,
            fdk_aac: 0.15,
            vorbis: 0.05,
            opus: 0.2,
        },
    };

    assert_eq!(score.prob_transcode, 0.75);
    assert_eq!(score.prob_codec.mp3, 0.1);
    assert_eq!(score.prob_codec.aac, 0.2);
    assert_eq!(score.prob_codec.aac_at, 0.3);
    assert_eq!(score.prob_codec.fdk_aac, 0.15);
    assert_eq!(score.prob_codec.vorbis, 0.05);
    assert_eq!(score.prob_codec.opus, 0.2);
}

#[test]
fn supported_extensions_are_case_insensitive() {
    for path in ["track.wav", "track.AIF", "track.aiff", "track.FLAC"] {
        assert!(lossprint::is_supported_file(path));
    }
    assert!(!lossprint::is_supported_file("track.mp3"));
}

#[test]
fn public_builder_validates_batch_size_before_loading_a_model() {
    for batch_size in [0, MAX_BATCH_TRACKS + 1] {
        let error = match Scanner::builder()
            .batch_size(batch_size)
            .build_from_model_bytes(&[])
        {
            Ok(_) => panic!("invalid batch size unexpectedly built a scanner"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), ErrorKind::Configuration);
    }
}

#[test]
fn caller_supplied_invalid_model_is_an_initialization_error() {
    let error = match Scanner::from_model_bytes(b"not an ONNX model") {
        Ok(_) => panic!("invalid model unexpectedly built a scanner"),
        Err(error) => error,
    };
    assert_eq!(error.kind(), ErrorKind::Initialization);
}
