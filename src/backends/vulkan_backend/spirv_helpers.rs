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
    let out = b.bitcast(f32_t, None, sh).unwrap();
    out
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
    let out = b.u_convert(u16_t, None, sh).unwrap();
    out
}
