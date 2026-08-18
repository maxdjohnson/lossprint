//! ONNX Runtime inference.

use anyhow::{bail, Context, Result};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use ort::ep;
use ort::{session::Session, value::TensorRef};
use std::path::Path;

use crate::spectrogram::{N_BINS, N_FRAMES, WINDOW_VALUES};

pub(crate) const CODEC_COUNT: usize = 6;

#[cfg(feature = "bundled-model")]
const MODEL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/model.onnx"));

pub(crate) struct WindowScore {
    pub transcode_probability: f32,
    pub codec_probabilities: [f32; CODEC_COUNT],
}

pub(crate) struct Model {
    session: Session,
}

impl Model {
    #[cfg(feature = "bundled-model")]
    pub fn bundled() -> Result<Self> {
        Self::from_bytes(MODEL)
    }

    pub fn from_bytes(model: &[u8]) -> Result<Self> {
        let session = Self::builder()?
            .commit_from_memory(model)
            .context("could not prepare the ONNX model")?;
        Ok(Self { session })
    }

    pub fn from_file(model: &Path) -> Result<Self> {
        let session = Self::builder()?
            .commit_from_file(model)
            .with_context(|| format!("could not prepare ONNX model {}", model.display()))?;
        Ok(Self { session })
    }

    fn builder() -> Result<ort::session::builder::SessionBuilder> {
        let mut builder = Session::builder().context("could not create inference session")?;
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let coreml = ep::CoreML::default()
                .with_model_format(ep::coreml::ModelFormat::MLProgram)
                .with_compute_units(ep::coreml::ComputeUnits::CPUAndGPU)
                .with_specialization_strategy(ep::coreml::SpecializationStrategy::FastPrediction)
                .build();
            builder = builder
                .with_intra_threads(1)
                .map_err(|error| anyhow::anyhow!("could not configure inference threads: {error}"))?
                .with_execution_providers([coreml])
                .map_err(|error| anyhow::anyhow!("could not enable Core ML: {error}"))?;
        }
        Ok(builder)
    }

    pub fn run(&mut self, input: &[f32]) -> Result<Vec<WindowScore>> {
        debug_assert!(!input.is_empty());
        debug_assert!(input.len().is_multiple_of(WINDOW_VALUES));
        let batch = input.len() / WINDOW_VALUES;
        let tensor = TensorRef::from_array_view(([batch, 2, N_BINS, N_FRAMES], input))?;
        let outputs = self
            .session
            .run(ort::inputs![tensor])
            .context("ONNX inference failed")?;
        let transcode = outputs
            .get("transcode_probability")
            .context("model did not return transcode_probability")?
            .try_extract_tensor::<f32>()?
            .1;
        let codecs = outputs
            .get("encoder_probability")
            .context("model did not return encoder_probability")?
            .try_extract_tensor::<f32>()?
            .1;
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
