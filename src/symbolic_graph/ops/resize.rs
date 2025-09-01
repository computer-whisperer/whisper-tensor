use crate::milli_graph::MilliOpGraph;
use crate::onnx;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{
    ONNXDecodingError, SymbolicGraphTensorId, query_attribute_bool, query_attribute_float,
    query_attribute_ints, query_attribute_string,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum ResizeCoordinateTransformationMode {
    HalfPixel,
    HalfPixelSymmetric,
    PytorchHalfPixel,
    AlignCorners,
    Asymmetric,
    TFCropAndResize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum ResizeKeepAspectRatioPolicy {
    Stretch,
    NotLarger,
    NotSmaller,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum ResizeMode {
    Nearest,
    Linear,
    Cubic,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum ResizeNearestMode {
    RoundPreferFloor,
    Ceil,
    Floor,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResizeOperation {
    input: SymbolicGraphTensorId,
    roi: Option<SymbolicGraphTensorId>,
    scales: Option<SymbolicGraphTensorId>,
    sizes: Option<SymbolicGraphTensorId>,
    output: SymbolicGraphTensorId,
    antialias: bool,
    axes: Vec<i64>,
    coordinate_transformation_mode: ResizeCoordinateTransformationMode,
    cubic_coeff_a: f32,
    exclude_outside: bool,
    extrapolation_value: f32,
    keep_aspect_ratio_policy: ResizeKeepAspectRatioPolicy,
    mode: ResizeMode,
    nearest_mode: ResizeNearestMode,
}

impl ResizeOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() || inputs.len() > 4 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Resize"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Resize"));
        }

        let antialias = query_attribute_bool(attributes, "antialias").unwrap_or_default();
        let axes = query_attribute_ints(attributes, "axes").unwrap_or_default();
        let coordinate_transformation_mode =
            query_attribute_string(attributes, "coordinate_transformation_mode")
                .unwrap_or("half_pixel".to_string());
        let coordinate_transformation_mode = match coordinate_transformation_mode.as_str() {
            "half_pixel_symmetric" => ResizeCoordinateTransformationMode::HalfPixelSymmetric,
            "align_corners" => ResizeCoordinateTransformationMode::AlignCorners,
            "asymmetric" => ResizeCoordinateTransformationMode::Asymmetric,
            "pytorch_half_pixel" => ResizeCoordinateTransformationMode::PytorchHalfPixel,
            "tf_crop_and_resize" => ResizeCoordinateTransformationMode::TFCropAndResize,
            "half_pixel" => ResizeCoordinateTransformationMode::HalfPixel,
            _ => ResizeCoordinateTransformationMode::HalfPixel,
        };
        let cubic_coeff_a = query_attribute_float(attributes, "cubic_coeff_a").unwrap_or(-0.75);
        let exclude_outside = query_attribute_bool(attributes, "exclude_outside").unwrap_or(false);
        let extrapolation_value =
            query_attribute_float(attributes, "extrapolation_value").unwrap_or(0.0);
        let keep_aspect_ratio_policy =
            query_attribute_string(attributes, "keep_aspect_ratio_policy")
                .unwrap_or("stretch".to_string());
        let keep_aspect_ratio_policy = match keep_aspect_ratio_policy.as_str() {
            "stretch" => ResizeKeepAspectRatioPolicy::Stretch,
            "not_larger" => ResizeKeepAspectRatioPolicy::NotLarger,
            "not_smaller" => ResizeKeepAspectRatioPolicy::NotSmaller,
            _ => ResizeKeepAspectRatioPolicy::Stretch,
        };
        let mode = query_attribute_string(attributes, "mode").unwrap_or("nearest".to_string());
        let mode = match mode.as_str() {
            "nearest" => ResizeMode::Nearest,
            "linear" => ResizeMode::Linear,
            "cubic" => ResizeMode::Cubic,
            _ => ResizeMode::Nearest,
        };
        let nearest_mode = query_attribute_string(attributes, "nearest_mode")
            .unwrap_or("round_prefer_floor".to_string());
        let nearest_mode = match nearest_mode.as_str() {
            "round_prefer_floor" => ResizeNearestMode::RoundPreferFloor,
            "floor" => ResizeNearestMode::Floor,
            "ceil" => ResizeNearestMode::Ceil,
            _ => ResizeNearestMode::RoundPreferFloor,
        };

        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Resize"))?,
            roi: if inputs.len() > 1 {
                Some(inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Resize"))?)
            } else {
                None
            },
            scales: if inputs.len() > 2 {
                Some(inputs[2].ok_or(ONNXDecodingError::InvalidOperatorInputs("Resize"))?)
            } else {
                None
            },
            sizes: if inputs.len() > 3 {
                Some(inputs[3].ok_or(ONNXDecodingError::InvalidOperatorInputs("Resize"))?)
            } else {
                None
            },
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Resize"))?,
            antialias,
            axes,
            coordinate_transformation_mode,
            cubic_coeff_a,
            exclude_outside,
            extrapolation_value,
            keep_aspect_ratio_policy,
            mode,
            nearest_mode,
        })
    }
}

impl Operation for ResizeOperation {
    fn get_op_type_name(&self) -> String {
        "Resize".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        let mut ret = vec![self.input];
        if let Some(roi) = &self.roi {
            ret.push(*roi);
        }
        if let Some(scales) = &self.scales {
            ret.push(*scales);
        }
        if let Some(sizes) = &self.sizes {
            ret.push(*sizes)
        }
        ret
    }

    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        todo!()
    }
}
