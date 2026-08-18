//! CPU-only ONNX inference through tract.

use anyhow::{bail, Context, Result};
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;
use tract_onnx::prelude::*;

use crate::spectrogram::{N_BINS, N_FRAMES, WINDOW_VALUES};

pub(crate) const CODEC_COUNT: usize = 9;

#[cfg(feature = "bundled-model")]
const MODEL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/model.onnx"));

pub(crate) struct WindowScore {
    pub transcode_probability: f32,
    pub codec_probabilities: [f32; CODEC_COUNT],
}

pub(crate) struct Model {
    runnable: Arc<TypedRunnableModel>,
}

impl Model {
    #[cfg(feature = "bundled-model")]
    pub fn bundled() -> Result<Self> {
        Self::from_bytes(MODEL)
    }

    pub fn from_bytes(model: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(model);
        let model = tract_onnx::onnx()
            .model_for_read(&mut cursor)
            .context("could not parse the ONNX model")?;
        Self::prepare(model)
    }

    pub fn from_file(model: &Path) -> Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(model)
            .with_context(|| format!("could not parse ONNX model {}", model.display()))?;
        Self::prepare(model)
    }

    fn prepare(model: InferenceModel) -> Result<Self> {
        let runnable = model
            .into_optimized()
            .context("could not optimize the ONNX model")?
            .into_runnable()
            .context("could not prepare the ONNX model")?;
        Ok(Self { runnable })
    }

    pub fn run(&mut self, input: &[f32]) -> Result<Vec<WindowScore>> {
        debug_assert!(!input.is_empty());
        debug_assert!(input.len().is_multiple_of(WINDOW_VALUES));
        let batch = input.len() / WINDOW_VALUES;
        let input =
            tract_ndarray::Array4::from_shape_vec((batch, 2, N_BINS, N_FRAMES), input.to_vec())
                .context("could not shape model input")?
                .into_tensor();
        let outputs = self
            .runnable
            .run(tvec!(input.into()))
            .context("ONNX inference failed")?;
        if outputs.len() != 3 {
            bail!("model returned {} outputs; expected 3", outputs.len())
        }
        let transcode = outputs[0]
            .to_plain_array_view::<f32>()
            .context("transcode_probability is not float32")?;
        let codecs = outputs[1]
            .to_plain_array_view::<f32>()
            .context("encoder_probability is not float32")?;
        let codecs = codecs
            .as_slice()
            .context("encoder_probability is not contiguous")?;
        if transcode.len() != batch || codecs.len() != batch * CODEC_COUNT {
            bail!("model returned predictions with the wrong shape")
        }
        Ok(transcode
            .iter()
            .zip(codecs.chunks_exact(CODEC_COUNT))
            .map(|(&transcode_probability, probabilities)| WindowScore {
                transcode_probability,
                codec_probabilities: std::array::from_fn(|index| probabilities[index]),
            })
            .collect())
    }
}
