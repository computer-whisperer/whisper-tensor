use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops as milli_ops;
use crate::onnx;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{
    ONNXDecodingError, query_attribute_bool, query_attribute_float, query_attribute_ints,
    query_attribute_string,
};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    RoundPreferCeil,
    Ceil,
    Floor,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResizeOperation {
    global_id: GlobalId,
    input: GlobalId,
    roi: Option<GlobalId>,
    scales: Option<GlobalId>,
    sizes: Option<GlobalId>,
    output: GlobalId,
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
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
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
            "round_prefer_ceil" => ResizeNearestMode::RoundPreferCeil,
            "floor" => ResizeNearestMode::Floor,
            "ceil" => ResizeNearestMode::Ceil,
            _ => ResizeNearestMode::RoundPreferFloor,
        };

        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Resize"))?,
            roi: inputs.get(1).copied().flatten(),
            scales: inputs.get(2).copied().flatten(),
            sizes: inputs.get(3).copied().flatten(),
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

impl Node for ResizeOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Resize".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
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
        Box::new(ret.into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}
impl Operation for ResizeOperation {
    fn parameters(&self) -> Vec<Property> {
        let mode_str = match &self.mode {
            ResizeMode::Nearest => "nearest",
            ResizeMode::Linear => "linear",
            ResizeMode::Cubic => "cubic",
        };
        let coord_mode_str = match &self.coordinate_transformation_mode {
            ResizeCoordinateTransformationMode::HalfPixel => "half_pixel",
            ResizeCoordinateTransformationMode::HalfPixelSymmetric => "half_pixel_symmetric",
            ResizeCoordinateTransformationMode::PytorchHalfPixel => "pytorch_half_pixel",
            ResizeCoordinateTransformationMode::AlignCorners => "align_corners",
            ResizeCoordinateTransformationMode::Asymmetric => "asymmetric",
            ResizeCoordinateTransformationMode::TFCropAndResize => "tf_crop_and_resize",
        };
        let nearest_mode_str = match &self.nearest_mode {
            ResizeNearestMode::RoundPreferFloor => "round_prefer_floor",
            ResizeNearestMode::RoundPreferCeil => "round_prefer_ceil",
            ResizeNearestMode::Ceil => "ceil",
            ResizeNearestMode::Floor => "floor",
        };
        let aspect_policy_str = match &self.keep_aspect_ratio_policy {
            ResizeKeepAspectRatioPolicy::Stretch => "stretch",
            ResizeKeepAspectRatioPolicy::NotLarger => "not_larger",
            ResizeKeepAspectRatioPolicy::NotSmaller => "not_smaller",
        };

        let mut params = vec![
            Property::new("mode", PropertyValue::String(mode_str.to_string())),
            Property::new(
                "coordinate_transform",
                PropertyValue::String(coord_mode_str.to_string()),
            ),
        ];

        if matches!(self.mode, ResizeMode::Nearest) {
            params.push(Property::new(
                "nearest_mode",
                PropertyValue::String(nearest_mode_str.to_string()),
            ));
        }
        if matches!(self.mode, ResizeMode::Cubic) {
            params.push(Property::new(
                "cubic_coeff_a",
                PropertyValue::Float(self.cubic_coeff_a as f64),
            ));
        }
        if self.antialias {
            params.push(Property::new("antialias", PropertyValue::Bool(true)));
        }
        if self.exclude_outside {
            params.push(Property::new("exclude_outside", PropertyValue::Bool(true)));
        }
        if self.extrapolation_value != 0.0 {
            params.push(Property::new(
                "extrapolation_value",
                PropertyValue::Float(self.extrapolation_value as f64),
            ));
        }
        if !self.axes.is_empty() {
            params.push(Property::new(
                "axes",
                PropertyValue::IntList(self.axes.clone()),
            ));
        }
        if !matches!(
            self.keep_aspect_ratio_policy,
            ResizeKeepAspectRatioPolicy::Stretch
        ) {
            params.push(Property::new(
                "keep_aspect_ratio_policy",
                PropertyValue::String(aspect_policy_str.to_string()),
            ));
        }

        params
    }

    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);

        let milli_mode = match &self.mode {
            ResizeMode::Nearest => milli_ops::ResizeMode::Nearest,
            ResizeMode::Linear => milli_ops::ResizeMode::Linear,
            ResizeMode::Cubic => milli_ops::ResizeMode::Cubic,
        };
        let milli_coord = match &self.coordinate_transformation_mode {
            ResizeCoordinateTransformationMode::HalfPixel => {
                milli_ops::ResizeCoordTransform::HalfPixel
            }
            ResizeCoordinateTransformationMode::HalfPixelSymmetric => {
                milli_ops::ResizeCoordTransform::HalfPixelSymmetric
            }
            ResizeCoordinateTransformationMode::PytorchHalfPixel => {
                milli_ops::ResizeCoordTransform::PytorchHalfPixel
            }
            ResizeCoordinateTransformationMode::AlignCorners => {
                milli_ops::ResizeCoordTransform::AlignCorners
            }
            ResizeCoordinateTransformationMode::Asymmetric => {
                milli_ops::ResizeCoordTransform::Asymmetric
            }
            ResizeCoordinateTransformationMode::TFCropAndResize => {
                milli_ops::ResizeCoordTransform::TFCropAndResize
            }
        };
        let milli_nearest = match &self.nearest_mode {
            ResizeNearestMode::RoundPreferFloor => milli_ops::ResizeNearestMode::RoundPreferFloor,
            ResizeNearestMode::RoundPreferCeil => milli_ops::ResizeNearestMode::RoundPreferCeil,
            ResizeNearestMode::Ceil => milli_ops::ResizeNearestMode::Ceil,
            ResizeNearestMode::Floor => milli_ops::ResizeNearestMode::Floor,
        };
        let milli_policy = match &self.keep_aspect_ratio_policy {
            ResizeKeepAspectRatioPolicy::Stretch => milli_ops::ResizeKeepAspectRatioPolicy::Stretch,
            ResizeKeepAspectRatioPolicy::NotLarger => {
                milli_ops::ResizeKeepAspectRatioPolicy::NotLarger
            }
            ResizeKeepAspectRatioPolicy::NotSmaller => {
                milli_ops::ResizeKeepAspectRatioPolicy::NotSmaller
            }
        };

        let out = milli_ops::Resize::push_new(
            &mut graph,
            input_map[&self.input],
            self.roi.map(|id| input_map[&id]),
            self.scales.map(|id| input_map[&id]),
            self.sizes.map(|id| input_map[&id]),
            milli_mode,
            milli_coord,
            milli_nearest,
            self.cubic_coeff_a,
            self.antialias,
            self.exclude_outside,
            self.extrapolation_value,
            self.axes.clone(),
            milli_policy,
            rng,
        );

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
