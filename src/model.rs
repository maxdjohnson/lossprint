//! ONNX Runtime inference.

use anyhow::{bail, Context, Result};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use ort::ep;
use ort::{session::Session, value::TensorRef};

use crate::spectrogram::{N_BINS, N_FRAMES, WINDOW_VALUES};

const MODEL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/model.onnx"));
pub const ENCODER_LABELS: [&str; 6] = ["mp3", "aac", "aac_at", "fdk_aac", "vorbis", "opus"];

pub struct WindowScore {
    pub transcode_probability: f32,
    pub encoder_probabilities: [f32; ENCODER_LABELS.len()],
}

pub struct Model {
    session: Session,
}

impl Model {
    pub fn new() -> Result<Self> {
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
        let session = builder
            .commit_from_memory(MODEL)
            .context("could not prepare the ONNX model")?;
        Ok(Self { session })
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
        let transcode = outputs["transcode_probability"]
            .try_extract_tensor::<f32>()?
            .1;
        let encoder = outputs["encoder_probability"]
            .try_extract_tensor::<f32>()?
            .1;
        if transcode.len() != batch || encoder.len() != batch * ENCODER_LABELS.len() {
            bail!("model returned predictions with the wrong shape")
        }
        Ok(transcode
            .iter()
            .zip(encoder.chunks_exact(ENCODER_LABELS.len()))
            .map(|(&transcode_probability, probabilities)| WindowScore {
                transcode_probability,
                encoder_probabilities: std::array::from_fn(|index| probabilities[index]),
            })
            .collect())
    }
}
