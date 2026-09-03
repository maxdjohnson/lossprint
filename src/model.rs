//! CPU-only ONNX inference through tract.

use std::io::Cursor;
use std::sync::Arc;
use tract_onnx::prelude::*;

use crate::spectrogram::{CHANNELS, N_BINS};

pub(crate) const FAMILY_COUNT: usize = 7;

const MODEL: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/model/model.onnx"));

pub(crate) struct WindowScore {
    pub transcode_probability: f32,
    pub family_probabilities: [f32; FAMILY_COUNT],
    pub bitrate_kbps: f32,
}

pub(crate) struct Model {
    runnable: Arc<TypedRunnableModel>,
}

impl Model {
    pub(crate) fn new() -> TractResult<Self> {
        let mut cursor = Cursor::new(MODEL);
        let model = tract_onnx::onnx().model_for_read(&mut cursor)?;
        // One window per call; the frame axis stays symbolic so every window
        // length runs through the same optimized graph.
        let frames = model.sym("frames");
        let runnable = model
            .with_input_fact(
                0,
                InferenceFact::dt_shape(
                    f32::datum_type(),
                    tvec![
                        1.to_dim(),
                        CHANNELS.to_dim(),
                        N_BINS.to_dim(),
                        frames.to_dim()
                    ],
                ),
            )?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { runnable })
    }

    pub(crate) fn run(&self, window: Vec<f32>, frames: usize) -> TractResult<WindowScore> {
        let input = tract_ndarray::Array4::from_shape_vec((1, CHANNELS, N_BINS, frames), window)
            .expect("window-aligned input fits the model shape")
            .into_tensor();
        let outputs = self.runnable.run(tvec!(input.into()))?;
        let [transcode, families, bitrate] = outputs.as_slice() else {
            unreachable!("embedded model has exactly three outputs")
        };
        let scalar = |tensor: &TValue| {
            tensor
                .to_plain_array_view::<f32>()
                .expect("embedded model outputs float32")
                .iter()
                .next()
                .copied()
                .expect("embedded model outputs one value per window")
        };
        let families = families
            .to_plain_array_view::<f32>()
            .expect("embedded model's family output is float32");
        let families = families
            .as_slice()
            .expect("tract outputs contiguous family probabilities");
        Ok(WindowScore {
            transcode_probability: scalar(transcode),
            family_probabilities: std::array::from_fn(|index| families[index]),
            bitrate_kbps: scalar(bitrate),
        })
    }
}
