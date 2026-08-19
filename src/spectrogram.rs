//! Native-rate mid/side log-magnitude spectrograms.

use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub(crate) const N_BINS: usize = 513;
pub(crate) const N_FRAMES: usize = 44;
pub(crate) const WINDOW_VALUES: usize = 2 * N_BINS * N_FRAMES;
const MAX_WINDOWS: usize = 16;
const LOG_FLOOR: f32 = -13.815_511; // ln(1e-6)

#[derive(Default)]
pub(crate) struct TransformCache {
    transforms: Mutex<HashMap<u32, Arc<Transform>>>,
}

impl TransformCache {
    pub(crate) fn get(&self, sample_rate: u32) -> Arc<Transform> {
        let mut transforms = self
            .transforms
            .lock()
            .expect("transform cache mutex poisoned");
        if let Some(transform) = transforms.get(&sample_rate) {
            return Arc::clone(transform);
        }
        let transform = Arc::new(Transform::new(sample_rate));
        transforms.insert(sample_rate, Arc::clone(&transform));
        transform
    }
}

pub(crate) struct Transform {
    n_fft: usize,
    hop: usize,
    crop_len: usize,
    window: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
}

impl Transform {
    pub(crate) fn new(sample_rate: u32) -> Self {
        let n_fft = fft_size(sample_rate);
        let hop = n_fft / 2;
        let crop_len = sample_rate as usize / 2;
        let window = (0..n_fft)
            .map(|index| {
                0.5_f32
                    - 0.5_f32 * (2.0_f32 * std::f32::consts::PI * index as f32 / n_fft as f32).cos()
            })
            .collect();
        let fft = FftPlanner::new().plan_fft_forward(n_fft);
        Self {
            n_fft,
            hop,
            crop_len,
            window,
            fft,
        }
    }

    /// Return `[window, 2, 513, 44]` mid/side spectrograms in row-major order.
    pub(crate) fn write_windows(&self, channels: &[Vec<f32>], starts: &[usize]) -> Vec<f32> {
        let crop_len = self.crop_len;
        debug_assert!((1..=2).contains(&channels.len()));
        debug_assert!(starts.iter().all(|&start| channels
            .iter()
            .all(|channel| start + crop_len <= channel.len())));

        let mut output = vec![0.0_f32; starts.len() * WINDOW_VALUES];
        let mut buffer = vec![Complex32::new(0.0, 0.0); self.n_fft];
        let mut scratch = vec![Complex32::new(0.0, 0.0); self.fft.get_inplace_scratch_len()];
        for (&start, output) in starts.iter().zip(output.chunks_exact_mut(WINDOW_VALUES)) {
            self.write_window(channels, start, output, &mut buffer, &mut scratch);
        }
        output
    }

    fn write_window(
        &self,
        channels: &[Vec<f32>],
        start: usize,
        output: &mut [f32],
        buffer: &mut [Complex32],
        scratch: &mut [Complex32],
    ) {
        let plane = N_BINS * N_FRAMES;

        if channels.len() == 1 {
            self.write_plane(
                |index| channels[0][start + index],
                &mut output[..plane],
                buffer,
                scratch,
            );
            output[plane..].fill(LOG_FLOOR);
        } else {
            self.write_plane(
                |index| (channels[0][start + index] + channels[1][start + index]) * 0.5_f32,
                &mut output[..plane],
                buffer,
                scratch,
            );
            self.write_plane(
                |index| channels[0][start + index] - channels[1][start + index],
                &mut output[plane..],
                buffer,
                scratch,
            );
        }
    }

    fn write_plane<F>(
        &self,
        sample: F,
        output: &mut [f32],
        buffer: &mut [Complex32],
        scratch: &mut [Complex32],
    ) where
        F: Fn(usize) -> f32,
    {
        let padding = self.n_fft / 2;
        let bins = (self.n_fft / 2 + 1).min(N_BINS);
        let frames = (self.crop_len / self.hop + 1).min(N_FRAMES);
        output.fill(LOG_FLOOR);
        for frame in 0..frames {
            let frame_start = frame * self.hop;
            for (offset, slot) in buffer.iter_mut().enumerate() {
                let padded = frame_start + offset;
                let source = reflect(padded as isize - padding as isize, self.crop_len);
                *slot = Complex32::new(sample(source) * self.window[offset], 0.0);
            }
            self.fft.process_with_scratch(buffer, scratch);
            for bin in 0..bins {
                output[bin * N_FRAMES + frame] = (buffer[bin].norm() + 1e-6_f32).ln();
            }
        }
    }
}

/// Scale the FFT size to preserve the model's 44.1 kHz bin spacing.
fn fft_size(sample_rate: u32) -> usize {
    ((u64::from(sample_rate) * 1024 + 22_050) / 44_100) as usize
}

pub(crate) fn window_offsets(total_frames: usize, crop_len: usize) -> Vec<usize> {
    if total_frames < crop_len {
        return Vec::new();
    }
    let hop = crop_len / 2;
    let available = (total_frames - crop_len) / hop + 1;
    if available <= MAX_WINDOWS {
        return (0..available).map(|index| index * hop).collect();
    }
    (0..MAX_WINDOWS)
        .map(|index| index * (available - 1) / (MAX_WINDOWS - 1) * hop)
        .collect()
}

fn reflect(index: isize, length: usize) -> usize {
    let last = length as isize - 1;
    debug_assert!((-last..=2 * last).contains(&index));
    if index < 0 {
        (-index) as usize
    } else if index > last {
        (2 * last - index) as usize
    } else {
        index as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_sizes_match_the_model_frontend() {
        assert_eq!(fft_size(32_000), 743);
        assert_eq!(fft_size(44_100), 1024);
        assert_eq!(fft_size(48_000), 1115);
        assert_eq!(fft_size(88_200), 2048);
        assert_eq!(fft_size(96_000), 2229);
    }

    #[test]
    fn twenty_seconds_uses_the_reference_windows() {
        let offsets = window_offsets(20 * 44_100, 44_100 / 2);
        let seconds = offsets
            .iter()
            .map(|offset| *offset as f32 / 44_100.0)
            .collect::<Vec<_>>();
        assert_eq!(seconds.len(), 16);
        assert_eq!(seconds[0], 0.0);
        assert_eq!(seconds[15], 19.5);
    }

    #[test]
    fn reflect_padding_excludes_the_edge_sample() {
        assert_eq!(reflect(-1, 4), 1);
        assert_eq!(reflect(-2, 4), 2);
        assert_eq!(reflect(-3, 4), 3);
        assert_eq!(reflect(4, 4), 2);
        assert_eq!(reflect(5, 4), 1);
        assert_eq!(reflect(6, 4), 0);
    }

    #[test]
    fn mono_has_a_log_spectrogram_and_zero_side_plane() {
        // 8,075 Hz naturally produces 43 STFT frames; the frontend pads the
        // model's required 44th frame with the log floor.
        let rate = 8_075;
        let transform = Transform::new(rate);
        let channels = vec![vec![0.0_f32; rate as usize / 2]];
        let output = transform.write_windows(&channels, &[0]);
        assert!(output.iter().all(|value| (*value - LOG_FLOOR).abs() < 1e-6));
    }

    #[test]
    fn shared_fft_storage_preserves_each_window() {
        let rate = 8_000;
        let transform = Transform::new(rate);
        let channels = vec![(0..3 * rate as usize)
            .map(|index| (index % 97) as f32 / 97.0)
            .collect::<Vec<_>>()];
        let starts = [0, rate as usize];
        let together = transform.write_windows(&channels, &starts);
        let separately = starts
            .iter()
            .flat_map(|&start| transform.write_windows(&channels, &[start]))
            .collect::<Vec<_>>();
        assert_eq!(together, separately);
    }
}
