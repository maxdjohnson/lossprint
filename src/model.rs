//! CPU-only ONNX inference through tract.

use std::io::Cursor;
use std::sync::Arc;
use tract_onnx::prelude::*;

use crate::spectrogram::{N_BINS, N_FRAMES, WINDOW_VALUES};

pub(crate) const ENCODER_COUNT: usize = 9;

const MODEL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/model.onnx"));

/// A failure while loading the embedded model.
#[derive(Debug, thiserror::Error)]
pub(crate) enum InitializationError {
    /// The embedded ONNX bytes could not be parsed.
    #[error("could not parse the embedded ONNX model: {0}")]
    Parse(#[source] Box<dyn std::error::Error + Send + Sync>),
    /// The parsed model could not be optimized.
    #[error("could not optimize the embedded ONNX model: {0}")]
    Optimize(#[source] Box<dyn std::error::Error + Send + Sync>),
    /// The optimized model could not be made runnable.
    #[error("could not prepare the embedded ONNX model: {0}")]
    Prepare(#[source] Box<dyn std::error::Error + Send + Sync>),
}

/// A failure while running the embedded model.
#[derive(Debug, thiserror::Error)]
pub(crate) enum InferenceError {
    /// The inference runtime failed.
    #[error("could not run the ONNX model: {0}")]
    Run(#[source] Box<dyn std::error::Error + Send + Sync>),
    /// The fixed model returned the wrong number of output heads.
    #[error("model returned {actual} outputs; expected 3")]
    UnexpectedOutputCount {
        /// Number of output heads returned by the model.
        actual: usize,
    },
    /// The fixed model's transcode output is not float32.
    #[error("model's transcode output is not float32")]
    InvalidTranscodeOutputType,
    /// The fixed model's encoder output is not float32.
    #[error("model's encoder output is not float32")]
    InvalidEncoderOutputType,
    /// The fixed model's encoder output is not stored contiguously.
    #[error("model's encoder output is not contiguous")]
    NonContiguousEncoderOutput,
    /// The fixed model returned the wrong number of predictions.
    #[error(
        "model returned the wrong output shape for {windows} windows: \
         {transcode_values} transcode values and {encoder_values} encoder values"
    )]
    UnexpectedOutputShape {
        /// Number of input windows.
        windows: usize,
        /// Number of returned transcode values.
        transcode_values: usize,
        /// Number of returned encoder values.
        encoder_values: usize,
    },
}

pub(crate) struct WindowScore {
    pub transcode_probability: f32,
    pub encoder_probabilities: [f32; ENCODER_COUNT],
}

pub(crate) struct Model {
    runnable: Arc<TypedRunnableModel>,
}

impl Model {
    pub(crate) fn new() -> Result<Self, InitializationError> {
        let mut cursor = Cursor::new(MODEL);
        let model = tract_onnx::onnx()
            .model_for_read(&mut cursor)
            .map_err(|error| InitializationError::Parse(error.into_boxed_dyn_error()))?;
        let runnable = model
            .into_optimized()
            .map_err(|error| InitializationError::Optimize(error.into_boxed_dyn_error()))?
            .into_runnable()
            .map_err(|error| InitializationError::Prepare(error.into_boxed_dyn_error()))?;
        Ok(Self { runnable })
    }

    pub(crate) fn run(&self, input: &[f32]) -> Result<Vec<WindowScore>, InferenceError> {
        debug_assert!(!input.is_empty());
        debug_assert!(input.len().is_multiple_of(WINDOW_VALUES));
        let batch = input.len() / WINDOW_VALUES;
        let input =
            tract_ndarray::Array4::from_shape_vec((batch, 2, N_BINS, N_FRAMES), input.to_vec())
                .expect("window-aligned input fits the model shape")
                .into_tensor();
        let outputs = self
            .runnable
            .run(tvec!(input.into()))
            .map_err(|error| InferenceError::Run(error.into_boxed_dyn_error()))?;
        if outputs.len() != 3 {
            return Err(InferenceError::UnexpectedOutputCount {
                actual: outputs.len(),
            });
        }
        let transcode = outputs[0]
            .to_plain_array_view::<f32>()
            .map_err(|_| InferenceError::InvalidTranscodeOutputType)?;
        let encoders = outputs[1]
            .to_plain_array_view::<f32>()
            .map_err(|_| InferenceError::InvalidEncoderOutputType)?;
        let encoders = encoders
            .as_slice()
            .ok_or(InferenceError::NonContiguousEncoderOutput)?;
        if transcode.len() != batch || encoders.len() != batch * ENCODER_COUNT {
            return Err(InferenceError::UnexpectedOutputShape {
                windows: batch,
                transcode_values: transcode.len(),
                encoder_values: encoders.len(),
            });
        }
        Ok(transcode
            .iter()
            .zip(encoders.chunks_exact(ENCODER_COUNT))
            .map(|(&transcode_probability, probabilities)| WindowScore {
                transcode_probability,
                encoder_probabilities: std::array::from_fn(|index| probabilities[index]),
            })
            .collect())
    }
}
