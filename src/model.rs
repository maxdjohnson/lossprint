//! CPU-only ONNX inference through tract.

use std::io::Cursor;
use std::sync::Arc;
use tract_onnx::prelude::*;

use crate::spectrogram::{N_BINS, N_FRAMES, WINDOW_VALUES};

pub(crate) const ENCODER_COUNT: usize = 9;

const MODEL: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/model/model.onnx"));

pub(crate) struct WindowScore {
    pub transcode_probability: f32,
    pub encoder_probabilities: [f32; ENCODER_COUNT],
}

pub(crate) struct Model {
    runnable: Arc<TypedRunnableModel>,
}

impl Model {
    pub(crate) fn new() -> TractResult<Self> {
        let mut cursor = Cursor::new(MODEL);
        let runnable = tract_onnx::onnx()
            .model_for_read(&mut cursor)?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { runnable })
    }

    pub(crate) fn run(&self, input: Vec<f32>) -> TractResult<Vec<WindowScore>> {
        let batch = input.len() / WINDOW_VALUES;
        let input = tract_ndarray::Array4::from_shape_vec((batch, 2, N_BINS, N_FRAMES), input)
            .expect("window-aligned input fits the model shape")
            .into_tensor();
        let outputs = self.runnable.run(tvec!(input.into()))?;
        let [transcode, encoders, _bandwidth] = outputs.as_slice() else {
            unreachable!("embedded model has exactly three outputs")
        };
        let transcode = transcode
            .to_plain_array_view::<f32>()
            .expect("embedded model's transcode output is float32");
        let encoders = encoders
            .to_plain_array_view::<f32>()
            .expect("embedded model's encoder output is float32");
        let encoders = encoders
            .as_slice()
            .expect("tract outputs contiguous encoder probabilities");
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
