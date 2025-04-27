import onnx
import onnx.helper
import onnx.reference
import onnx.external_data_helper
import numpy as np

def main():
    model_fname = "gpt2-10.onnx"
    model = onnx.load(model_fname)
    onnx.external_data_helper.load_external_data_for_model(model, ".")
    onnx.checker.check_model(model)

    session = onnx.reference.ReferenceEvaluator(model)
    tensor_input = np.array([[[10]]])
    input_types = session.input_types
    input_shape = input_types[0].tensor_type.shape
    output = session.run(None, {"token_input": tensor_input})
    print(output)


if __name__ == "__main__":
    main()