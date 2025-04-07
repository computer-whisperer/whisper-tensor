use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::default::Default;
use crate::tensor::{Shape, TensorData};
use crate::tensor::Tensor;
use crate::DType;

pub(crate) trait Node {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        vec![]
    }

    fn get_nodes(&self) -> HashSet<&dyn Node> where Self: Sized {
        let mut out = HashSet::new();
        let dyn_self: &dyn Node = self;
        out.insert(dyn_self);
        for input in self.get_input_tensors() {
            out.extend(input.get_nodes());
        }
        out
    }

    fn get_tensors(&self) -> HashSet<&dyn Tensor> {
        let mut out = HashSet::new();
        for input in self.get_input_tensors() {
            out.insert(input);
            out.extend(input.get_sub_tensors());
        }
        out
    }

    fn get_output_tensors(&self) -> Vec<&dyn Tensor>;

    fn get_name(&self) -> Option<&str> {
        None
    }


    fn get_onnx_type(&self) -> &str;
    fn get_onnx_domain(&self) -> &str {
        "ai.onnx"
    }

    fn get_onnx_attributes(&self) -> Vec<crate::onnx::AttributeProto> {
        vec![]
    }

    fn to_node_proto(&self, name: Option<String>, tensor_names: &HashMap<&dyn Tensor, String>) -> crate::onnx::NodeProto {
        crate::onnx::NodeProto {
            name: name.unwrap_or_default(),
            input: self.get_input_tensors().iter().map(|tensor| tensor_names[tensor].clone()).collect(),
            output: self.get_output_tensors().iter().map(|tensor| tensor_names[tensor].clone()).collect(),
            op_type: self.get_onnx_type().to_string(),
            domain: self.get_onnx_domain().to_string(),
            attribute: self.get_onnx_attributes(),
            .. Default::default()
        }
    }
}

impl<'a> PartialEq for &'a dyn Node{
    fn eq(&self, other:&Self) -> bool{
        std::ptr::addr_eq(*self, *other)
    }
}

impl<'a> Eq for &'a dyn Node{}

impl<'a> Hash for &'a dyn Node {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let a: *const _ = *self;
        let address: *const u8 = a.cast();
        state.write_usize(address.addr());
    }
}

/*
trait MultiOutputNode: Node {
    fn get_output_shape(&self, output_index: usize) -> Vec<usize>;

    fn get_output_dtype(&self, output_index: usize) -> DType;

    fn get_num_outputs(&self) -> usize;
}
*/

pub(crate) trait NodeElementwise: Node {}

pub(crate) trait SingleOutputNode: Node {
    fn get_output_shape(&self) -> &Shape;

    fn get_output_dtype(&self) -> DType;
    
    fn resolve_output_data(&self) -> Option<TensorData> {
        None
    }
}


impl <T: NodeElementwise> SingleOutputNode for T {
    fn get_output_shape(&self) -> &Shape {
        let inputs = self.get_input_tensors();
        for input in &inputs {
            assert_eq!(input.shape(), inputs[0].shape());
        }
        inputs[0].shape()
    }

    fn get_output_dtype(&self) -> DType {
        let inputs = self.get_input_tensors();
        for input in &inputs {
            assert_eq!(input.dtype(), inputs[0].dtype());
        }
        inputs[0].dtype()
    }
}