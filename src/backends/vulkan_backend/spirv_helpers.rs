use crate::backends::vulkan_backend::VulkanError;
use crate::dtype::DType;

pub fn get_spirv_datatype(
    b: &mut rspirv::dr::Builder,
    dtype: DType,
) -> Result<rspirv::spirv::Word, VulkanError> {
    Ok(match dtype {
        DType::F64 => b.type_float(64),
        DType::F32 => b.type_float(32),
        DType::BF16 => b.type_float(16),
        DType::F16 => b.type_float(16),
        DType::I64 => b.type_int(64, 1),
        DType::U64 => b.type_int(64, 0),
        DType::I32 => b.type_int(32, 1),
        DType::U32 => b.type_int(32, 0),
        DType::I16 => b.type_int(16, 1),
        DType::U16 => b.type_int(16, 0),
        DType::I8 => b.type_int(8, 1),
        DType::U8 => b.type_int(8, 0),
        DType::BOOL => b.type_bool(),
        DType::STRING => panic!(),
    })
}

pub fn cast_bf16_to_f32(
    b: &mut rspirv::dr::Builder,
    input: rspirv::spirv::Word,
) -> rspirv::spirv::Word {
    let f32_t = b.type_float(32);
    let u32_t = b.type_int(32, 0);

    let c16 = b.constant_bit32(u32_t, 16);

    let as32 = b.u_convert(u32_t, None, input).unwrap();
    let sh = b.shift_left_logical(u32_t, None, as32, c16).unwrap();
    b.bitcast(f32_t, None, sh).unwrap()
}

pub fn cast_f32_to_bf16(
    b: &mut rspirv::dr::Builder,
    input: rspirv::spirv::Word,
) -> rspirv::spirv::Word {
    let u32_t = b.type_int(32, 0);
    let u16_t = b.type_int(16, 0);

    let c16 = b.constant_bit32(u32_t, 16);
    let c8000 = b.constant_bit32(u32_t, 0x0000_8000);

    let bits32 = b.bitcast(u32_t, None, input).unwrap();
    let rounded = b.i_add(u32_t, None, bits32, c8000).unwrap();
    let sh = b.shift_right_logical(u32_t, None, rounded, c16).unwrap();
    b.u_convert(u16_t, None, sh).unwrap()
}

pub fn spirv_standard_cast(
    b: &mut rspirv::dr::Builder,
    input: rspirv::spirv::Word,
    input_dtype: DType,
    output_dtype: DType,
) -> Result<rspirv::spirv::Word, VulkanError> {
    if input_dtype == output_dtype {
        Ok(input)
    } else {
        let input_data_type = get_spirv_datatype(b, input_dtype)?;
        let output_data_type = get_spirv_datatype(b, output_dtype)?;
        match input_dtype {
            DType::BF16 | DType::F16 | DType::F32 | DType::F64 => match output_dtype {
                DType::BF16 | DType::F16 | DType::F32 | DType::F64 => {
                    Ok(b.f_convert(output_data_type, None, input).unwrap())
                }
                DType::I64 | DType::I32 | DType::I16 | DType::I8 => {
                    Ok(b.convert_f_to_s(output_data_type, None, input).unwrap())
                }
                DType::U64 | DType::U32 | DType::U16 | DType::U8 => {
                    Ok(b.convert_f_to_u(output_data_type, None, input).unwrap())
                }
                DType::BOOL => {
                    let f32_type = b.type_float(32);
                    let const_zero = b.constant_bit32(f32_type, 0.0f32.to_bits());
                    let const_zero = b.f_convert(input_data_type, None, const_zero).unwrap();
                    Ok(
                        b.f_unord_not_equal(output_data_type, None, input, const_zero)
                            .unwrap(),
                    )
                }
                _ => Err(VulkanError::UnsupportedByBackendError),
            },
            DType::I64 | DType::I32 | DType::I16 | DType::I8 => {
                match output_dtype {
                    DType::BF16 | DType::F16 | DType::F32 | DType::F64 => {
                        Ok(b.convert_s_to_f(output_data_type, None, input).unwrap())
                    }
                    DType::I64 | DType::I32 | DType::I16 | DType::I8 => {
                        Ok(b.s_convert(output_data_type, None, input).unwrap())
                    }
                    DType::U64 | DType::U32 | DType::U16 | DType::U8 => {
                        let i64_type = b.type_int(64, 1);
                        let u64_type = b.type_int(64, 0);
                        // up-convert to i64
                        let i = b.s_convert(i64_type, None, input).unwrap();
                        // convert to u64
                        let i = b.bitcast(u64_type, None, i).unwrap();
                        // down-convert to output type
                        Ok(b.u_convert(output_data_type, None, i).unwrap())
                    }
                    DType::BOOL => {
                        let u32_type = b.type_int(32, 1);
                        let const_zero = b.constant_bit32(u32_type, 0);
                        let const_zero = b.s_convert(input_data_type, None, const_zero).unwrap();
                        Ok(b.i_not_equal(output_data_type, None, input, const_zero)
                            .unwrap())
                    }
                    _ => Err(VulkanError::UnsupportedByBackendError),
                }
            }
            DType::U64 | DType::U32 | DType::U16 | DType::U8 => {
                match output_dtype {
                    DType::BF16 | DType::F16 | DType::F32 | DType::F64 => {
                        Ok(b.convert_u_to_f(output_data_type, None, input).unwrap())
                    }
                    DType::I64 | DType::I32 | DType::I16 | DType::I8 => {
                        let i64_type = b.type_int(64, 1);
                        let u64_type = b.type_int(64, 0);
                        // up-convert to u64
                        let i = b.u_convert(u64_type, None, input).unwrap();
                        // convert to i64
                        let i = b.bitcast(i64_type, None, i).unwrap();
                        // down-convert to output type
                        Ok(b.u_convert(output_data_type, None, i).unwrap())
                    }
                    DType::U64 | DType::U32 | DType::U16 | DType::U8 => {
                        Ok(b.u_convert(output_data_type, None, input).unwrap())
                    }
                    DType::BOOL => {
                        let u32_type = b.type_int(32, 0);
                        let const_zero = b.constant_bit32(u32_type, 0);
                        let const_zero = b.u_convert(input_data_type, None, const_zero).unwrap();
                        Ok(b.i_not_equal(output_data_type, None, input, const_zero)
                            .unwrap())
                    }
                    _ => Err(VulkanError::UnsupportedByBackendError),
                }
            }
            DType::BOOL => match output_dtype {
                DType::BF16 | DType::F16 | DType::F32 | DType::F64 => {
                    let f32_type = b.type_float(32);
                    let const_zero = b.constant_bit32(f32_type, 0.0f32.to_bits());
                    let const_one = b.constant_bit32(f32_type, 1.0f32.to_bits());
                    let tmp_out = b
                        .select(output_data_type, None, input, const_one, const_zero)
                        .unwrap();
                    Ok(b.f_convert(output_data_type, None, tmp_out).unwrap())
                }
                DType::I64 | DType::I32 | DType::I16 | DType::I8 => {
                    let i32_type = b.type_int(32, 1);
                    let const_zero = b.constant_bit32(i32_type, 0);
                    let const_one = b.constant_bit32(i32_type, 1);
                    let tmp_out = b
                        .select(output_data_type, None, input, const_one, const_zero)
                        .unwrap();
                    Ok(b.s_convert(output_data_type, None, tmp_out).unwrap())
                }
                DType::U64 | DType::U32 | DType::U16 | DType::U8 => {
                    let u32_type = b.type_int(32, 0);
                    let const_zero = b.constant_bit32(u32_type, 0);
                    let const_one = b.constant_bit32(u32_type, 1);
                    let tmp_out = b
                        .select(output_data_type, None, input, const_one, const_zero)
                        .unwrap();
                    Ok(b.u_convert(output_data_type, None, tmp_out).unwrap())
                }
                DType::BOOL => Ok(input),
                _ => Err(VulkanError::UnsupportedByBackendError),
            },
            _ => Err(VulkanError::UnsupportedByBackendError),
        }
    }
}
