//! Native-rate decoding for lossless containers.

use std::io::{Read, Seek, SeekFrom};
use symphonia::core::audio::GenericAudioBufferRef;
use symphonia::core::codecs::audio::well_known::{CODEC_ID_PCM_ALAW, CODEC_ID_PCM_MULAW};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::formats::TrackType;
use symphonia::core::io::{MediaSource as SymphoniaMediaSource, MediaSourceStream};
use symphonia::default::{get_codecs, get_probe};

const MAX_SECONDS: usize = 20;
const MIN_SAMPLE_RATE: u32 = 8_000;
const MAX_SAMPLE_RATE: u32 = 384_000;

/// A seekable byte source containing one audio file.
///
/// Files, in-memory cursors, and other types implementing the required
/// standard-library traits implement this trait automatically. The source
/// must be positioned at byte zero and its [`Seek`] implementation must work.
pub trait MediaSource: Read + Seek + Send + Sync {}

impl<T: ?Sized> MediaSource for T where T: Read + Seek + Send + Sync {}

/// A failure while decoding or validating an input audio file.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The container could not be recognized.
    #[error("could not recognize audio format: {0}")]
    Probe(#[source] Box<dyn std::error::Error + Send + Sync + 'static>),
    /// The container has no usable audio track.
    #[error("audio file has no usable audio track")]
    NoAudioTrack,
    /// The track uses companded rather than linear PCM.
    #[error("A-law and mu-law audio are not lossless PCM")]
    LossyPcm,
    /// The audio track does not declare a sample rate.
    #[error("audio track has no sample rate")]
    MissingSampleRate,
    /// The sample rate is outside the model's supported range.
    #[error(
        "{0} Hz sample rate is outside the supported {MIN_SAMPLE_RATE}-{MAX_SAMPLE_RATE} Hz range"
    )]
    UnsupportedSampleRate(u32),
    /// The audio track does not declare a channel layout.
    #[error("audio track has no channel layout")]
    MissingChannelLayout,
    /// The track is neither mono nor stereo.
    #[error("{0} channels; only mono and stereo are supported")]
    UnsupportedChannelCount(usize),
    /// A decoder could not be constructed for the audio track.
    #[error("could not create audio decoder: {0}")]
    CreateDecoder(#[source] Box<dyn std::error::Error + Send + Sync + 'static>),
    /// The next encoded packet could not be read.
    #[error("could not read audio packet: {0}")]
    ReadPacket(#[source] Box<dyn std::error::Error + Send + Sync + 'static>),
    /// An encoded packet could not be decoded.
    #[error("could not decode audio packet: {0}")]
    DecodePacket(#[source] Box<dyn std::error::Error + Send + Sync + 'static>),
    /// The track contains no decoded samples.
    #[error("audio track contains no decodable samples")]
    NoSamples,
    /// The track cannot fill one analysis window.
    #[error("audio is shorter than 0.5 seconds")]
    TooShort,
}

pub(crate) struct Clip {
    pub(crate) channels: Vec<Vec<f32>>,
    pub(crate) sample_rate: u32,
}

struct Source<S>(S);

impl<S: MediaSource> Read for Source<S> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        self.0.read(buffer)
    }

    fn read_vectored(&mut self, buffers: &mut [std::io::IoSliceMut<'_>]) -> std::io::Result<usize> {
        self.0.read_vectored(buffers)
    }
}

impl<S: MediaSource> Seek for Source<S> {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.0.seek(position)
    }
}

impl<S: MediaSource> SymphoniaMediaSource for Source<S> {
    fn is_seekable(&self) -> bool {
        true
    }

    fn byte_len(&self) -> Option<u64> {
        None
    }
}

/// Decode at most 20 seconds without resampling, downmixing, or requantizing.
pub(crate) fn decode<S: MediaSource>(source: S) -> Result<Clip, Error> {
    let stream = MediaSourceStream::new(Box::new(Source(source)), Default::default());

    let mut format = get_probe()
        .probe(
            &Default::default(),
            stream,
            Default::default(),
            Default::default(),
        )
        .map_err(|error| Error::Probe(Box::new(error)))?;
    let codec = format
        .default_track(TrackType::Audio)
        .and_then(|track| track.codec_params.as_ref())
        .and_then(|params| params.audio())
        .ok_or(Error::NoAudioTrack)?;
    if codec.codec == CODEC_ID_PCM_ALAW || codec.codec == CODEC_ID_PCM_MULAW {
        return Err(Error::LossyPcm);
    }
    let sample_rate = codec.sample_rate.ok_or(Error::MissingSampleRate)?;
    if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&sample_rate) {
        return Err(Error::UnsupportedSampleRate(sample_rate));
    }
    let channel_count = codec
        .channels
        .as_ref()
        .ok_or(Error::MissingChannelLayout)?
        .count();
    if !(1..=2).contains(&channel_count) {
        return Err(Error::UnsupportedChannelCount(channel_count));
    }
    let wanted_frames = MAX_SECONDS * sample_rate as usize;
    let mut channels = (0..channel_count)
        .map(|_| Vec::with_capacity(wanted_frames))
        .collect::<Vec<_>>();
    let mut decoder = get_codecs()
        .make_audio_decoder(codec, &AudioDecoderOptions::default())
        .map_err(|error| Error::CreateDecoder(Box::new(error)))?;

    while let Some(packet) = format
        .next_packet()
        .map_err(|error| Error::ReadPacket(Box::new(error)))?
    {
        let decoded = decoder
            .decode(&packet)
            .map_err(|error| Error::DecodePacket(Box::new(error)))?;
        debug_assert_eq!(decoded.spec().rate(), sample_rate);
        debug_assert_eq!(decoded.spec().channels().count(), channel_count);
        append(decoded, &mut channels);
        if channels[0].len() >= wanted_frames {
            break;
        }
    }

    if channels[0].is_empty() {
        return Err(Error::NoSamples);
    }
    for channel in &mut channels {
        channel.truncate(wanted_frames);
    }
    Ok(Clip {
        channels,
        sample_rate,
    })
}

fn append(decoded: GenericAudioBufferRef<'_>, output: &mut [Vec<f32>]) {
    let frames = decoded.frames();
    let mut tails = output
        .iter_mut()
        .map(|plane| {
            let start = plane.len();
            plane.resize(start + frames, 0.0);
            &mut plane[start..]
        })
        .collect::<Vec<_>>();
    decoded.copy_to_slice_planar::<f32, _>(&mut tails);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Cursor;
    use std::path::{Path, PathBuf};

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/audio")
            .join(name)
    }

    fn decode_fixture(name: &str) -> Clip {
        decode(File::open(fixture(name)).unwrap()).unwrap()
    }

    #[test]
    fn wav_aiff_and_flac_decode_to_the_same_pcm() {
        let wav = decode_fixture("pcm16.wav");
        let aiff = decode_fixture("pcm16.aiff");
        let flac = decode_fixture("pcm16.flac");
        assert_eq!(wav.sample_rate, 44_100);
        assert_eq!(wav.channels, aiff.channels);
        assert_eq!(wav.channels, flac.channels);
        assert_eq!(
            wav.channels,
            vec![
                vec![-1.0, -0.5, -1.0 / 32_768.0, 0.0],
                vec![
                    32_767.0 / 32_768.0,
                    0.5,
                    1.0 / 32_768.0,
                    12_345.0 / 32_768.0,
                ],
            ]
        );
    }

    #[test]
    fn preserves_native_float_samples_and_layout() {
        let clip = decode_fixture("float32-stereo.wav");

        assert_eq!(clip.sample_rate, 48_000);
        assert_eq!(clip.channels, vec![vec![1.25, 0.25], vec![-1.5, -0.75]]);
    }

    #[test]
    fn preserves_mono_instead_of_duplicating_it() {
        let clip = decode_fixture("float32-mono.wav");

        assert_eq!(clip.sample_rate, 32_000);
        assert_eq!(clip.channels, vec![vec![0.5, -0.25]]);
    }

    #[test]
    fn decodes_an_in_memory_source() {
        let bytes = std::fs::read(fixture("pcm16.wav")).unwrap();
        let clip = decode(Cursor::new(bytes.as_slice())).unwrap();

        assert_eq!(clip.sample_rate, 44_100);
        assert_eq!(clip.channels.len(), 2);
    }
}
