//! Native-rate decoding for lossless containers.

use crate::{validate, DecodeError, ScoreError};
use std::io::{Read, Seek, SeekFrom};
use std::sync::{Mutex, PoisonError};
use symphonia::core::audio::GenericAudioBufferRef;
use symphonia::core::codecs::audio::well_known::{CODEC_ID_PCM_ALAW, CODEC_ID_PCM_MULAW};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::formats::TrackType;
use symphonia::core::io::{MediaSource as SymphoniaMediaSource, MediaSourceStream};
use symphonia::default::{get_codecs, get_probe};

pub(crate) const MAX_SECONDS: usize = 20;
pub(crate) const MIN_SAMPLE_RATE: u32 = 8_000;
pub(crate) const MAX_SAMPLE_RATE: u32 = 384_000;

/// A seekable byte source containing one audio file.
pub trait MediaSource: Read + Seek + Send {}

impl<T: ?Sized> MediaSource for T where T: Read + Seek + Send {}

pub(crate) struct Clip {
    pub(crate) channels: Vec<Vec<f32>>,
    pub(crate) sample_rate: u32,
}

/// Adapt a [`Send`] reader into the `Send + Sync` source Symphonia requires.
///
/// Symphonia's own `MediaSource` demands [`Sync`] even though the stream is only
/// ever driven from one thread. [`Mutex`] is [`Sync`] for any [`Send`] payload,
/// and `get_mut` reaches the reader through `&mut self` without taking the lock,
/// so the wrapper costs nothing at runtime. `std::sync::Exclusive` expresses
/// this directly but is still unstable.
struct SyncSource<S>(Mutex<S>);

impl<S> SyncSource<S> {
    fn get(&mut self) -> &mut S {
        // Never locked, so never actually poisoned.
        self.0.get_mut().unwrap_or_else(PoisonError::into_inner)
    }
}

impl<S: Read> Read for SyncSource<S> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        self.get().read(buffer)
    }

    fn read_vectored(&mut self, buffers: &mut [std::io::IoSliceMut<'_>]) -> std::io::Result<usize> {
        self.get().read_vectored(buffers)
    }
}

impl<S: Seek> Seek for SyncSource<S> {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.get().seek(position)
    }
}

impl<S: Read + Seek + Send> SymphoniaMediaSource for SyncSource<S> {
    fn is_seekable(&self) -> bool {
        true
    }

    fn byte_len(&self) -> Option<u64> {
        None
    }
}

/// Decode at most 20 seconds without resampling, downmixing, or requantizing.
///
/// Deliberately non-generic, since Symphonia boxes the source into a trait
/// object anyway.
pub(crate) fn decode(source: &mut dyn MediaSource) -> Result<Clip, ScoreError> {
    let stream =
        MediaSourceStream::new(Box::new(SyncSource(Mutex::new(source))), Default::default());

    let mut format = get_probe()
        .probe(
            &Default::default(),
            stream,
            Default::default(),
            Default::default(),
        )
        .map_err(malformed)?;
    let codec = format
        .default_track(TrackType::Audio)
        .and_then(|track| track.codec_params.as_ref())
        .and_then(|params| params.audio())
        .ok_or(DecodeError::NoUsableAudio)?;
    if codec.codec == CODEC_ID_PCM_ALAW || codec.codec == CODEC_ID_PCM_MULAW {
        return Err(DecodeError::NotLinearPcm.into());
    }
    let sample_rate = codec.sample_rate.ok_or(DecodeError::NoUsableAudio)?;
    let channel_count = codec
        .channels
        .as_ref()
        .ok_or(DecodeError::NoUsableAudio)?
        .count();
    validate(sample_rate, channel_count)?;

    let wanted_frames = MAX_SECONDS * sample_rate as usize;
    let mut channels = (0..channel_count)
        .map(|_| Vec::with_capacity(wanted_frames))
        .collect::<Vec<_>>();
    let mut decoder = get_codecs()
        .make_audio_decoder(codec, &AudioDecoderOptions::default())
        .map_err(malformed)?;

    while let Some(packet) = format.next_packet().map_err(malformed)? {
        let decoded = decoder.decode(&packet).map_err(malformed)?;
        debug_assert_eq!(decoded.spec().rate(), sample_rate);
        debug_assert_eq!(decoded.spec().channels().count(), channel_count);
        append(decoded, &mut channels);
        if channels[0].len() >= wanted_frames {
            break;
        }
    }

    if channels[0].is_empty() {
        return Err(DecodeError::NoUsableAudio.into());
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

fn malformed(error: impl std::error::Error + Send + Sync + 'static) -> ScoreError {
    ScoreError::Decode(DecodeError::Malformed {
        source: Box::new(error),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::fs::File;
    use std::io::Cursor;
    use std::path::{Path, PathBuf};

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/audio")
            .join(name)
    }

    fn decode_fixture(name: &str) -> Clip {
        decode(&mut File::open(fixture(name)).unwrap()).unwrap()
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
        let clip = decode(&mut Cursor::new(bytes.as_slice())).unwrap();

        assert_eq!(clip.sample_rate, 44_100);
        assert_eq!(clip.channels.len(), 2);
    }

    /// A `Send` but `!Sync` reader, which Symphonia's own trait would reject.
    struct NotSync {
        inner: Cursor<Vec<u8>>,
        reads: Cell<usize>,
    }

    impl Read for NotSync {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            self.reads.set(self.reads.get() + 1);
            self.inner.read(buffer)
        }
    }

    impl Seek for NotSync {
        fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
            self.inner.seek(position)
        }
    }

    #[test]
    fn decodes_a_send_but_not_sync_source() {
        let mut source = NotSync {
            inner: Cursor::new(std::fs::read(fixture("pcm16.wav")).unwrap()),
            reads: Cell::new(0),
        };
        let clip = decode(&mut source).unwrap();

        assert_eq!(clip.sample_rate, 44_100);
        assert!(source.reads.get() > 0);
    }
}
