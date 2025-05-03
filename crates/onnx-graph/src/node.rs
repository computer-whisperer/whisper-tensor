use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::default::Default;
use crate::tensor::{Shape, TensorData};
use crate::tensor::Tensor;
use crate::DType;

pub trait Node {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        vec![]
    }

    fn get_nodes<'a>(&'a self, table: &mut HashSet<&'a dyn Node>) where Self: Sized {
        let dyn_self: &dyn Node = self;
        if !table.contains(&dyn_self) {
            self.get_sub_nodes(table);
            table.insert(dyn_self);
        }
    }

    fn get_sub_nodes<'a>(&'a self, table: &mut HashSet<&'a dyn Node>) {
        for input in self.get_input_tensors() {
            input.get_nodes(table);
        }
    }

    fn get_tensors<'a>(&'a self, table: &mut HashSet<&'a dyn Tensor>) {
        for input in self.get_input_tensors() {
            if !table.contains(&input) {
                table.insert(input);
                input.get_sub_tensors(table);
            }
        }
    }

    fn get_output_tensors(&self) -> Vec<&dyn Tensor>;

    fn get_name(&self) -> Option<&str> {
        None
    }


    fn get_onnx_type(&self) -> &str;
    fn get_onnx_domain(&self) -> &str {
        ""
    }

    fn get_onnx_attributes(&self) -> Vec<crate::onnx::AttributeProto>;

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

pub(crate) trait SingleOutputNode: Node {
    fn get_output_shape(&self) -> &Shape;

    fn get_output_dtype(&self) -> DType;
    
    fn resolve_output_data(&self) -> Option<TensorData> {
        None
    }
}