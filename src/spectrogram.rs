//! Native-rate dual-resolution mid/side log-magnitude spectrograms.

use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use std::collections::HashMap;
use std::f32::consts::PI;
use std::sync::{Arc, Mutex, PoisonError};

/// Input planes: mid and side at the long window, then mid and side at the
/// short window, upsampled onto the long window's frequency grid.
pub(crate) const CHANNELS: usize = 4;
pub(crate) const N_BINS: usize = 1024;
const SHORT_BINS: usize = N_BINS / 4;
/// `clamp(20 * log10(0 + 1e-7), -100, 40) / 50`: the value of a silent bin.
const SILENCE: f32 = -2.0;

#[derive(Default)]
pub(crate) struct TransformCache {
    transforms: Mutex<HashMap<u32, Arc<Transform>>>,
}

impl TransformCache {
    pub(crate) fn get(&self, sample_rate: u32) -> Arc<Transform> {
        // Recovering beats poisoning the cache for the process's lifetime.
        let mut transforms = self
            .transforms
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        Arc::clone(
            transforms
                .entry(sample_rate)
                .or_insert_with(|| Arc::new(Transform::new(sample_rate))),
        )
    }
}

pub(crate) struct Transform {
    hop: usize,
    long_window: Vec<f32>,
    short_window: Vec<f32>,
    long_fft: Arc<dyn Fft<f32>>,
    short_fft: Arc<dyn Fft<f32>>,
}

/// The model's fixed FFT size for a sample-rate family.
pub(crate) fn fft_size(sample_rate: u32) -> usize {
    if sample_rate <= 50_000 {
        2048
    } else if sample_rate <= 100_000 {
        4096
    } else {
        8192
    }
}

/// STFT frames in `samples` without centering; zero below one FFT of audio.
#[cfg(test)]
pub(crate) fn frame_count(samples: usize, sample_rate: u32) -> usize {
    frames_in(samples, fft_size(sample_rate))
}

fn frames_in(samples: usize, n_fft: usize) -> usize {
    if samples < n_fft {
        0
    } else {
        (samples - n_fft) / (n_fft / 4) + 1
    }
}

fn periodic_hann(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| 0.5_f32 - 0.5_f32 * (2.0_f32 * PI * index as f32 / len as f32).cos())
        .collect()
}

fn log_magnitude(magnitude: f32) -> f32 {
    (20.0_f32 * (magnitude + 1e-7_f32).log10()).clamp(-100.0, 40.0) / 50.0
}

impl Transform {
    pub(crate) fn new(sample_rate: u32) -> Self {
        let n_fft = fft_size(sample_rate);
        let mut planner = FftPlanner::new();
        Self {
            hop: n_fft / 4,
            long_window: periodic_hann(n_fft),
            short_window: periodic_hann(n_fft / 4),
            long_fft: planner.plan_fft_forward(n_fft),
            short_fft: planner.plan_fft_forward(n_fft / 4),
        }
    }

    /// Return one `[4, 1024, frames]` window in row-major order with its
    /// frame count. Every channel must hold at least one FFT of samples.
    pub(crate) fn write_window(&self, channels: &[&[f32]]) -> (Vec<f32>, usize) {
        debug_assert!((1..=2).contains(&channels.len()));
        let n_fft = self.long_window.len();
        let samples = channels
            .iter()
            .map(|channel| channel.len())
            .min()
            .unwrap_or(0);
        let frames = frames_in(samples, n_fft);
        debug_assert!(frames > 0);
        let plane = N_BINS * frames;

        let mut output = vec![SILENCE; CHANNELS * plane];
        let mut buffer = vec![Complex32::new(0.0, 0.0); n_fft];
        let mut scratch = vec![
            Complex32::new(0.0, 0.0);
            self.long_fft
                .get_inplace_scratch_len()
                .max(self.short_fft.get_inplace_scratch_len())
        ];
        let (long, short) = output.split_at_mut(2 * plane);
        let (long_mid, long_side) = long.split_at_mut(plane);
        let (short_mid, short_side) = short.split_at_mut(plane);

        if let [mono] = channels {
            let mid = |index: usize| mono[index];
            self.write_plane(false, &mid, long_mid, frames, &mut buffer, &mut scratch);
            self.write_plane(true, &mid, short_mid, frames, &mut buffer, &mut scratch);
            // A mono file has an all-zero side channel, which is `SILENCE`.
        } else {
            let (left, right) = (channels[0], channels[1]);
            let mid = |index: usize| (left[index] + right[index]) * 0.5_f32;
            let side = |index: usize| (left[index] - right[index]) * 0.5_f32;
            self.write_plane(false, &mid, long_mid, frames, &mut buffer, &mut scratch);
            self.write_plane(false, &side, long_side, frames, &mut buffer, &mut scratch);
            self.write_plane(true, &mid, short_mid, frames, &mut buffer, &mut scratch);
            self.write_plane(true, &side, short_side, frames, &mut buffer, &mut scratch);
        }
        (output, frames)
    }

    /// Fill one `[1024, frames]` plane. The short window shares the long
    /// window's hop and frame starts; each of its bins is repeated four times
    /// so bin `k` sits at the same frequency as long-window bin `k`.
    fn write_plane<F>(
        &self,
        short: bool,
        sample: &F,
        output: &mut [f32],
        frames: usize,
        buffer: &mut [Complex32],
        scratch: &mut [Complex32],
    ) where
        F: Fn(usize) -> f32,
    {
        let (fft, window, bins, repeat) = if short {
            (
                &self.short_fft,
                &self.short_window,
                SHORT_BINS,
                N_BINS / SHORT_BINS,
            )
        } else {
            (&self.long_fft, &self.long_window, N_BINS, 1)
        };
        let buffer = &mut buffer[..window.len()];
        for frame in 0..frames {
            let start = frame * self.hop;
            for (offset, slot) in buffer.iter_mut().enumerate() {
                *slot = Complex32::new(sample(start + offset) * window[offset], 0.0);
            }
            fft.process_with_scratch(buffer, scratch);
            for (bin, value) in buffer[..bins].iter().enumerate() {
                let value = log_magnitude(value.norm());
                for row in bin * repeat..(bin + 1) * repeat {
                    output[row * frames + frame] = value;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(sample_rate: u32, hz: f32, samples: usize) -> Vec<f32> {
        (0..samples)
            .map(|index| (2.0 * PI * hz * index as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn fft_sizes_follow_the_sample_rate_families() {
        assert_eq!(fft_size(8_000), 2048);
        assert_eq!(fft_size(44_100), 2048);
        assert_eq!(fft_size(48_000), 2048);
        assert_eq!(fft_size(88_200), 4096);
        assert_eq!(fft_size(96_000), 4096);
        assert_eq!(fft_size(192_000), 8192);
    }

    #[test]
    fn frame_counts_match_an_uncentered_stft() {
        assert_eq!(frame_count(2047, 44_100), 0);
        assert_eq!(frame_count(2048, 44_100), 1);
        assert_eq!(frame_count(2048 + 511, 44_100), 1);
        assert_eq!(frame_count(2048 + 512, 44_100), 2);
        // 14.5 seconds: the reference window length.
        assert_eq!(frame_count(639_450, 44_100), 1245);
        assert_eq!(frame_count(696_000, 48_000), 1356);
        assert_eq!(frame_count(1_392_000, 96_000), 1356);
    }

    #[test]
    fn silence_hits_the_log_floor() {
        assert_eq!(log_magnitude(0.0), SILENCE);
        assert!(log_magnitude(1.0).abs() < 1e-6);
        assert!((log_magnitude(1e6) - 40.0 / 50.0).abs() < 1e-6);
    }

    #[test]
    fn mono_fills_silent_side_planes_and_upsampled_short_planes() {
        let rate = 44_100;
        let transform = Transform::new(rate);
        let channel = tone(rate, 1_000.0, 2048 + 512 * 3);
        let (output, frames) = transform.write_window(&[&channel]);
        assert_eq!(frames, 4);
        let plane = N_BINS * frames;
        assert_eq!(output.len(), CHANNELS * plane);
        assert!(output[plane..2 * plane].iter().all(|&v| v == SILENCE));
        assert!(output[3 * plane..].iter().all(|&v| v == SILENCE));
        let short = &output[2 * plane..3 * plane];
        for bin in 0..SHORT_BINS {
            let rows: Vec<&[f32]> = (0..4)
                .map(|k| &short[(bin * 4 + k) * frames..(bin * 4 + k + 1) * frames])
                .collect();
            assert!(rows.iter().all(|row| *row == rows[0]));
        }
    }

    #[test]
    fn a_tone_peaks_in_the_same_bin_at_both_resolutions() {
        let rate = 48_000;
        let transform = Transform::new(rate);
        // Bin 200 of a 2048-point FFT at 48 kHz is 4,687.5 Hz.
        let channel = tone(rate, 200.0 * 48_000.0 / 2048.0, 2048 + 512 * 7);
        let (output, frames) = transform.write_window(&[&channel, &channel]);
        let plane = N_BINS * frames;
        for start in [0, 2 * plane] {
            let frame0: Vec<f32> = (0..N_BINS)
                .map(|bin| output[start + bin * frames])
                .collect();
            let peak = frame0
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(bin, _)| bin)
                .unwrap();
            assert!((198..=203).contains(&peak), "peak bin {peak}");
        }
        // Identical channels have no side signal.
        assert!(output[plane..2 * plane].iter().all(|&v| v == SILENCE));
    }

    #[test]
    fn windows_are_reproducible() {
        let rate = 8_000;
        let transform = Transform::new(rate);
        let channel = (0..3 * rate as usize)
            .map(|index| (index % 97) as f32 / 97.0)
            .collect::<Vec<_>>();
        let first = transform.write_window(&[&channel]);
        let second = transform.write_window(&[&channel]);
        assert_eq!(first, second);
    }
}
