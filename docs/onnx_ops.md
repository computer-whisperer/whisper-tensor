# Supported ONNX Operators

142 operators supported. Both the NDArray (CPU) and Vulkan backends execute these via the Symbolic Graph → Milli-Op Graph pipeline, with per-op fallback from Vulkan to NDArray when a GPU kernel is not yet available.

## Operator list

| # | Operator | Category |
|---|----------|----------|
| 1 | Abs | Unary |
| 2 | Acos | Unary |
| 3 | Acosh | Unary |
| 4 | Add | Binary |
| 5 | And | Binary (logical) |
| 6 | ArgMax | Reduce |
| 7 | ArgMin | Reduce |
| 8 | Asin | Unary |
| 9 | Asinh | Unary |
| 10 | Atan | Unary |
| 11 | Atanh | Unary |
| 12 | AveragePool | Pool |
| 13 | BatchNormalization | Normalization |
| 14 | BiasGelu | Unary (fused) |
| 15 | BitShift | Binary (bitwise) |
| 16 | BitwiseAnd | Binary (bitwise) |
| 17 | BitwiseNot | Unary (bitwise) |
| 18 | BitwiseOr | Binary (bitwise) |
| 19 | BitwiseXor | Binary (bitwise) |
| 20 | Cast | Type |
| 21 | CastLike | Type |
| 22 | Ceil | Unary |
| 23 | Celu | Unary |
| 24 | Clip | Unary |
| 25 | Compress | Indexing |
| 26 | Concat | Shape |
| 27 | Constant | Constant |
| 28 | ConstantOfShape | Constant |
| 29 | Conv | Convolution |
| 30 | ConvTranspose | Convolution |
| 31 | Cos | Unary |
| 32 | Cosh | Unary |
| 33 | CumSum | Reduce |
| 34 | DepthToSpace | Shape |
| 35 | Div | Binary |
| 36 | Dropout | Identity (inference) |
| 37 | Einsum | Tensor contraction |
| 38 | Elu | Unary |
| 39 | Equal | Binary (comparison) |
| 40 | Erf | Unary |
| 41 | Exp | Unary |
| 42 | Expand | Shape |
| 43 | EyeLike | Constant |
| 44 | Flatten | Shape |
| 45 | Floor | Unary |
| 46 | Gather | Indexing |
| 47 | GatherElements | Indexing |
| 48 | GatherND | Indexing |
| 49 | Gelu | Unary |
| 50 | Gemm | Linear algebra |
| 51 | GlobalAveragePool | Pool |
| 52 | GlobalMaxPool | Pool |
| 53 | Greater | Binary (comparison) |
| 54 | GreaterOrEqual | Binary (comparison) |
| 55 | GroupNormalization | Normalization |
| 56 | Hardmax | Unary |
| 57 | HardSigmoid | Unary |
| 58 | HardSwish | Unary |
| 59 | Identity | Identity |
| 60 | If | Control flow |
| 61 | InstanceNormalization | Normalization |
| 62 | IsInf | Unary |
| 63 | IsNaN | Unary |
| 64 | LayerNormalization | Normalization |
| 65 | LeakyRelu | Unary |
| 66 | Less | Binary (comparison) |
| 67 | LessOrEqual | Binary (comparison) |
| 68 | Log | Unary |
| 69 | LogSoftmax | Unary |
| 70 | LpNormalization | Normalization |
| 71 | LSTM | Recurrent |
| 72 | MatMul | Linear algebra |
| 73 | Max | Binary |
| 74 | MaxPool | Pool |
| 75 | Mean | Binary |
| 76 | MeanVarianceNormalization | Normalization |
| 77 | Min | Binary |
| 78 | Mish | Unary |
| 79 | Mod | Binary |
| 80 | Mul | Binary |
| 81 | Neg | Unary |
| 82 | NegativeLogLikelihoodLoss | Loss |
| 83 | NonZero | Indexing |
| 84 | Not | Unary (logical) |
| 85 | Or | Binary (logical) |
| 86 | Pad | Shape |
| 87 | Pow | Binary |
| 88 | PRelu | Binary |
| 89 | RandomNormalLike | Random |
| 90 | Range | Constant |
| 91 | Reciprocal | Unary |
| 92 | ReduceL1 | Reduce |
| 93 | ReduceL2 | Reduce |
| 94 | ReduceLogSum | Reduce |
| 95 | ReduceLogSumExp | Reduce |
| 96 | ReduceMax | Reduce |
| 97 | ReduceMean | Reduce |
| 98 | ReduceMin | Reduce |
| 99 | ReduceProd | Reduce |
| 100 | ReduceSum | Reduce |
| 101 | ReduceSumSquare | Reduce |
| 102 | Relu | Unary |
| 103 | Reshape | Shape |
| 104 | Resize | Shape |
| 105 | ReverseSequence | Sequence |
| 106 | RMSNormalization | Normalization |
| 107 | RotaryEmbedding | Custom |
| 108 | Round | Unary |
| 109 | Scan | Control flow |
| 110 | Scatter | Indexing (alias for ScatterElements) |
| 111 | ScatterElements | Indexing |
| 112 | ScatterND | Indexing |
| 113 | Selu | Unary |
| 114 | Shape | Shape |
| 115 | Shrink | Unary |
| 116 | Sigmoid | Unary |
| 117 | Sign | Unary |
| 118 | Sin | Unary |
| 119 | Sinh | Unary |
| 120 | Size | Shape |
| 121 | Slice | Indexing |
| 122 | Softmax | Unary |
| 123 | SoftmaxCrossEntropyLoss | Loss |
| 124 | Softplus | Unary |
| 125 | Softsign | Unary |
| 126 | SpaceToDepth | Shape |
| 127 | Split | Shape |
| 128 | Sqrt | Unary |
| 129 | Squeeze | Shape |
| 130 | STFT | Signal |
| 131 | Sub | Binary |
| 132 | Sum | Binary |
| 133 | Tan | Unary |
| 134 | Tanh | Unary |
| 135 | ThresholdedRelu | Unary |
| 136 | Tile | Shape |
| 137 | TopK | Reduce |
| 138 | Transpose | Shape |
| 139 | Trilu | Shape |
| 140 | Unsqueeze | Shape |
| 141 | Where | Conditional |
| 142 | Xor | Binary (logical) |

## Notes

- **Opset coverage**: Priority is opset 19+ semantics. Some ops support multiple opset versions (e.g., Split handles both opset-2 and opset-13 signatures). Very old pre-10 edge cases are low priority.
- **Custom ops**: `RotaryEmbedding` and `BiasGelu` are non-standard ops commonly found in Transformers models.
- **Float8 types**: Cast supports F8E4M3FN and F8E5M2 dtypes for quantized model import.
- **Decomposition**: Many ops (e.g., Einsum, NegativeLogLikelihoodLoss, SoftmaxCrossEntropyLoss, AveragePool, MaxPool) are decomposed into simpler Milli-Op primitives rather than having dedicated backend kernels.
