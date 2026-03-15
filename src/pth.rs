use crate::dtype::DType;
use std::collections::HashMap;
use std::io::{BufRead, Read};
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum PthError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Zip(#[from] zip::result::ZipError),
    #[error("invalid pickle stream: {0}")]
    InvalidPickle(String),
    #[error("unsupported storage type {0}")]
    UnsupportedStorageType(String),
    #[error("unsupported dtype in pth parser: {0:?}")]
    UnsupportedDType(DType),
    #[error("missing key {0} in checkpoint object")]
    MissingKey(String),
}

type Result<T> = std::result::Result<T, PthError>;
type OResult<T> = std::result::Result<T, Object>;

#[repr(u8)]
#[derive(Debug, Eq, PartialEq, Clone)]
enum OpCode {
    Proto = 0x80,
    Global = b'c',
    BinPut = b'q',
    LongBinPut = b'r',
    EmptyTuple = b')',
    Reduce = b'R',
    Mark = b'(',
    BinUnicode = b'X',
    BinInt = b'J',
    Tuple = b't',
    BinPersId = b'Q',
    BinInt1 = b'K',
    BinInt2 = b'M',
    Tuple1 = 0x85,
    Tuple2 = 0x86,
    Tuple3 = 0x87,
    NewTrue = 0x88,
    NewFalse = 0x89,
    None = b'N',
    BinGet = b'h',
    LongBinGet = b'j',
    SetItem = b's',
    SetItems = b'u',
    EmptyDict = b'}',
    Dict = b'd',
    Build = b'b',
    Stop = b'.',
    NewObj = 0x81,
    EmptyList = b']',
    BinFloat = b'G',
    Append = b'a',
    Appends = b'e',
    Long1 = 0x8a,
}

impl TryFrom<u8> for OpCode {
    type Error = u8;

    fn try_from(value: u8) -> std::result::Result<Self, Self::Error> {
        match value {
            0x80 => Ok(Self::Proto),
            b'c' => Ok(Self::Global),
            b'q' => Ok(Self::BinPut),
            b'r' => Ok(Self::LongBinPut),
            b')' => Ok(Self::EmptyTuple),
            b'R' => Ok(Self::Reduce),
            b'(' => Ok(Self::Mark),
            b'X' => Ok(Self::BinUnicode),
            b'J' => Ok(Self::BinInt),
            b't' => Ok(Self::Tuple),
            b'Q' => Ok(Self::BinPersId),
            b'K' => Ok(Self::BinInt1),
            b'M' => Ok(Self::BinInt2),
            0x85 => Ok(Self::Tuple1),
            0x86 => Ok(Self::Tuple2),
            0x87 => Ok(Self::Tuple3),
            0x88 => Ok(Self::NewTrue),
            0x89 => Ok(Self::NewFalse),
            b'N' => Ok(Self::None),
            b'h' => Ok(Self::BinGet),
            b'j' => Ok(Self::LongBinGet),
            b's' => Ok(Self::SetItem),
            b'u' => Ok(Self::SetItems),
            b'}' => Ok(Self::EmptyDict),
            b'd' => Ok(Self::Dict),
            b'b' => Ok(Self::Build),
            b'.' => Ok(Self::Stop),
            0x81 => Ok(Self::NewObj),
            b']' => Ok(Self::EmptyList),
            b'G' => Ok(Self::BinFloat),
            b'a' => Ok(Self::Append),
            b'e' => Ok(Self::Appends),
            0x8a => Ok(Self::Long1),
            other => Err(other),
        }
    }
}

fn invalid_pickle(msg: impl Into<String>) -> PthError {
    PthError::InvalidPickle(msg.into())
}

fn read_u8<R: Read>(r: &mut R) -> Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

fn read_u16_le<R: Read>(r: &mut R) -> Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}

fn read_u32_le<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_i32_le<R: Read>(r: &mut R) -> Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}

fn read_f64_be<R: Read>(r: &mut R) -> Result<f64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_be_bytes(b))
}

fn read_to_newline<R: BufRead>(r: &mut R) -> Result<Vec<u8>> {
    let mut data = Vec::with_capacity(32);
    r.read_until(b'\n', &mut data)?;
    if data.is_empty() {
        return Err(invalid_pickle("expected newline-terminated string"));
    }
    data.pop();
    if data.last() == Some(&b'\r') {
        data.pop();
    }
    Ok(data)
}

#[derive(Debug, Clone, PartialEq)]
enum Object {
    Class {
        module_name: String,
        class_name: String,
    },
    Int(i32),
    Long(i64),
    Float(f64),
    Unicode(String),
    Bool(bool),
    None,
    Tuple(Vec<Object>),
    List(Vec<Object>),
    Mark,
    Dict(Vec<(Object, Object)>),
    Reduce {
        callable: Box<Object>,
        args: Box<Object>,
    },
    Build {
        callable: Box<Object>,
        args: Box<Object>,
    },
    PersistentLoad(Box<Object>),
}

impl Object {
    fn unicode(self) -> OResult<String> {
        match self {
            Self::Unicode(t) => Ok(t),
            other => Err(other),
        }
    }

    fn reduce(self) -> OResult<(Self, Self)> {
        match self {
            Self::Reduce { callable, args } => Ok((*callable, *args)),
            other => Err(other),
        }
    }

    fn int_or_long(self) -> OResult<i64> {
        match self {
            Self::Int(t) => Ok(t as i64),
            Self::Long(t) => Ok(t),
            other => Err(other),
        }
    }

    fn tuple(self) -> OResult<Vec<Self>> {
        match self {
            Self::Tuple(t) => Ok(t),
            other => Err(other),
        }
    }

    fn class(self) -> OResult<(String, String)> {
        match self {
            Self::Class {
                module_name,
                class_name,
            } => Ok((module_name, class_name)),
            other => Err(other),
        }
    }

    fn persistent_load(self) -> OResult<Self> {
        match self {
            Self::PersistentLoad(t) => Ok(*t),
            other => Err(other),
        }
    }

    fn into_tensor_info(self, name: Self, dir_name: &Path) -> Result<Option<TensorInfo>> {
        let name = match name.unicode() {
            Ok(name) => name,
            Err(_) => return Ok(None),
        };

        let (callable, args) = match self.reduce() {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };

        let (callable, args) = match callable {
            Object::Class {
                module_name,
                class_name,
            } if module_name == "torch._tensor" && class_name == "_rebuild_from_type_v2" => {
                let mut args = args
                    .tuple()
                    .map_err(|obj| invalid_pickle(format!("expected tuple args, got {obj:?}")))?;
                let callable = args.remove(0);
                let args = args.remove(1);
                (callable, args)
            }
            Object::Class {
                module_name,
                class_name,
            } if module_name == "torch._utils" && class_name == "_rebuild_parameter" => {
                let mut args = args
                    .tuple()
                    .map_err(|obj| invalid_pickle(format!("expected tuple args, got {obj:?}")))?;
                args.remove(0)
                    .reduce()
                    .map_err(|obj| invalid_pickle(format!("expected reduce args, got {obj:?}")))?
            }
            _ => (callable, args),
        };

        match callable {
            Object::Class {
                module_name,
                class_name,
            } if module_name == "torch._utils" && class_name == "_rebuild_tensor_v2" => {}
            _ => return Ok(None),
        }

        let (layout, dtype, file_path, storage_size) = rebuild_args(args)?;
        Ok(Some(TensorInfo {
            name,
            dtype,
            layout,
            path: format!("{}/{}", dir_name.to_string_lossy(), file_path),
            storage_size,
        }))
    }
}

impl TryFrom<Object> for String {
    type Error = Object;

    fn try_from(value: Object) -> std::result::Result<Self, Self::Error> {
        match value {
            Object::Unicode(s) => Ok(s),
            other => Err(other),
        }
    }
}

impl TryFrom<Object> for usize {
    type Error = Object;

    fn try_from(value: Object) -> std::result::Result<Self, Self::Error> {
        match value {
            Object::Int(s) if s >= 0 => Ok(s as usize),
            other => Err(other),
        }
    }
}

impl<T: TryFrom<Object, Error = Object>> TryFrom<Object> for Vec<T> {
    type Error = Object;

    fn try_from(value: Object) -> std::result::Result<Self, Self::Error> {
        match value {
            Object::Tuple(values) => values
                .into_iter()
                .map(T::try_from)
                .collect::<std::result::Result<Vec<T>, Self::Error>>(),
            other => Err(other),
        }
    }
}

#[derive(Debug)]
struct Stack {
    stack: Vec<Object>,
    memo: HashMap<u32, Object>,
}

impl Stack {
    fn empty() -> Self {
        Self {
            stack: Vec::with_capacity(512),
            memo: HashMap::new(),
        }
    }

    fn read_loop<R: BufRead>(&mut self, r: &mut R) -> Result<()> {
        loop {
            if self.read(r)? {
                break;
            }
        }
        Ok(())
    }

    fn finalize(mut self) -> Result<Object> {
        self.pop()
    }

    fn push(&mut self, obj: Object) {
        self.stack.push(obj);
    }

    fn pop(&mut self) -> Result<Object> {
        self.stack
            .pop()
            .ok_or_else(|| invalid_pickle("unexpected empty stack"))
    }

    fn last_mut(&mut self) -> Result<&mut Object> {
        self.stack
            .last_mut()
            .ok_or_else(|| invalid_pickle("unexpected empty stack"))
    }

    fn memo_get(&self, id: u32) -> Result<Object> {
        self.memo
            .get(&id)
            .cloned()
            .ok_or_else(|| invalid_pickle(format!("missing object in memo {id}")))
    }

    fn memo_put(&mut self, id: u32) -> Result<()> {
        let obj = self
            .stack
            .last()
            .cloned()
            .ok_or_else(|| invalid_pickle("unexpected empty stack"))?;
        self.memo.insert(id, obj);
        Ok(())
    }

    fn persistent_load(&self, id: Object) -> Object {
        Object::PersistentLoad(Box::new(id))
    }

    fn new_obj(&self, class: Object, args: Object) -> Object {
        Object::Reduce {
            callable: Box::new(class),
            args: Box::new(args),
        }
    }

    fn pop_to_marker(&mut self) -> Result<Vec<Object>> {
        let mut mark_idx = None;
        for (idx, obj) in self.stack.iter().enumerate().rev() {
            if obj == &Object::Mark {
                mark_idx = Some(idx);
                break;
            }
        }

        match mark_idx {
            Some(mark_idx) => {
                let objs = self.stack.split_off(mark_idx + 1);
                self.stack.pop();
                Ok(objs)
            }
            None => Err(invalid_pickle("marker object not found")),
        }
    }

    fn build(&mut self) -> Result<()> {
        let args = self.pop()?;
        let obj = self.pop()?;
        let obj = match (obj, args) {
            (Object::Dict(mut obj), Object::Dict(mut args)) => {
                obj.append(&mut args);
                Object::Dict(obj)
            }
            (obj, args) => Object::Build {
                callable: Box::new(obj),
                args: Box::new(args),
            },
        };
        self.push(obj);
        Ok(())
    }

    fn reduce(&mut self) -> Result<()> {
        let args = self.pop()?;
        let callable = self.pop()?;
        let reduced = match &callable {
            Object::Class {
                module_name,
                class_name,
            } if module_name == "collections"
                && (class_name == "OrderedDict" || class_name == "defaultdict") =>
            {
                Some(Object::Dict(vec![]))
            }
            _ => None,
        };

        self.push(reduced.unwrap_or_else(|| Object::Reduce {
            callable: Box::new(callable),
            args: Box::new(args),
        }));
        Ok(())
    }

    fn read<R: BufRead>(&mut self, r: &mut R) -> Result<bool> {
        let op_raw = read_u8(r)?;
        let op_code = OpCode::try_from(op_raw)
            .map_err(|_| invalid_pickle(format!("unknown op-code {op_raw}")))?;

        match op_code {
            OpCode::Proto => {
                let _version = read_u8(r)?;
            }
            OpCode::Global => {
                let module_name = String::from_utf8_lossy(&read_to_newline(r)?).to_string();
                let class_name = String::from_utf8_lossy(&read_to_newline(r)?).to_string();
                self.push(Object::Class {
                    module_name,
                    class_name,
                });
            }
            OpCode::BinInt1 => self.push(Object::Int(read_u8(r)? as i32)),
            OpCode::BinInt2 => self.push(Object::Int(read_u16_le(r)? as i32)),
            OpCode::BinInt => self.push(Object::Int(read_i32_le(r)?)),
            OpCode::BinFloat => self.push(Object::Float(read_f64_be(r)?)),
            OpCode::BinUnicode => {
                let len = read_u32_le(r)? as usize;
                let mut data = vec![0u8; len];
                r.read_exact(&mut data)?;
                let data = String::from_utf8(data)
                    .map_err(|e| invalid_pickle(format!("invalid utf8 in unicode: {e}")))?;
                self.push(Object::Unicode(data));
            }
            OpCode::BinPersId => {
                let id = self.pop()?;
                self.push(self.persistent_load(id));
            }
            OpCode::Tuple => {
                let objs = self.pop_to_marker()?;
                self.push(Object::Tuple(objs));
            }
            OpCode::Tuple1 => {
                let obj = self.pop()?;
                self.push(Object::Tuple(vec![obj]));
            }
            OpCode::Tuple2 => {
                let obj2 = self.pop()?;
                let obj1 = self.pop()?;
                self.push(Object::Tuple(vec![obj1, obj2]));
            }
            OpCode::Tuple3 => {
                let obj3 = self.pop()?;
                let obj2 = self.pop()?;
                let obj1 = self.pop()?;
                self.push(Object::Tuple(vec![obj1, obj2, obj3]));
            }
            OpCode::NewTrue => self.push(Object::Bool(true)),
            OpCode::NewFalse => self.push(Object::Bool(false)),
            OpCode::Append => {
                let value = self.pop()?;
                match self.last_mut()? {
                    Object::List(d) => d.push(value),
                    other => {
                        return Err(invalid_pickle(format!("expected list, got {other:?}")));
                    }
                }
            }
            OpCode::Appends => {
                let objs = self.pop_to_marker()?;
                match self.last_mut()? {
                    Object::List(d) => d.extend(objs),
                    other => {
                        return Err(invalid_pickle(format!("expected list, got {other:?}")));
                    }
                }
            }
            OpCode::SetItem => {
                let value = self.pop()?;
                let key = self.pop()?;
                match self.last_mut()? {
                    Object::Dict(d) => d.push((key, value)),
                    other => {
                        return Err(invalid_pickle(format!("expected dict, got {other:?}")));
                    }
                }
            }
            OpCode::SetItems => {
                let mut objs = self.pop_to_marker()?;
                match self.last_mut()? {
                    Object::Dict(d) => {
                        if objs.len() % 2 != 0 {
                            return Err(invalid_pickle("setitems: not an even number of objects"));
                        }
                        while let Some(value) = objs.pop() {
                            let key = objs
                                .pop()
                                .ok_or_else(|| invalid_pickle("setitems: missing key"))?;
                            d.push((key, value));
                        }
                    }
                    other => {
                        return Err(invalid_pickle(format!("expected dict, got {other:?}")));
                    }
                }
            }
            OpCode::None => self.push(Object::None),
            OpCode::Stop => return Ok(true),
            OpCode::Build => self.build()?,
            OpCode::EmptyDict => self.push(Object::Dict(vec![])),
            OpCode::Dict => {
                let mut objs = self.pop_to_marker()?;
                if objs.len() % 2 != 0 {
                    return Err(invalid_pickle("dict: not an even number of objects"));
                }
                let mut pydict = vec![];
                while let Some(value) = objs.pop() {
                    let key = objs
                        .pop()
                        .ok_or_else(|| invalid_pickle("dict: missing key"))?;
                    pydict.push((key, value));
                }
                self.push(Object::Dict(pydict));
            }
            OpCode::Mark => self.push(Object::Mark),
            OpCode::Reduce => self.reduce()?,
            OpCode::EmptyTuple => self.push(Object::Tuple(vec![])),
            OpCode::EmptyList => self.push(Object::List(vec![])),
            OpCode::BinGet => {
                let arg = read_u8(r)?;
                self.push(self.memo_get(arg as u32)?);
            }
            OpCode::LongBinGet => {
                let arg = read_u32_le(r)?;
                self.push(self.memo_get(arg)?);
            }
            OpCode::BinPut => self.memo_put(read_u8(r)? as u32)?,
            OpCode::LongBinPut => self.memo_put(read_u32_le(r)?)?,
            OpCode::NewObj => {
                let args = self.pop()?;
                let class = self.pop()?;
                self.push(self.new_obj(class, args));
            }
            OpCode::Long1 => {
                let n_bytes = read_u8(r)?;
                let mut v = 0i64;
                for i in 0..n_bytes {
                    v |= (read_u8(r)? as i64) << (i * 8);
                }
                self.push(Object::Long(v));
            }
        }

        Ok(false)
    }
}

#[derive(Debug, Clone)]
pub struct PthLayout {
    shape: Vec<usize>,
    stride: Vec<usize>,
    start_offset: usize,
}

impl PthLayout {
    pub fn new(shape: Vec<usize>, stride: Vec<usize>, start_offset: usize) -> Self {
        Self {
            shape,
            stride,
            start_offset,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn start_offset(&self) -> usize {
        self.start_offset
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_contiguous(&self) -> bool {
        if self.shape.len() != self.stride.len() {
            return false;
        }
        let mut expected = 1usize;
        for idx in (0..self.shape.len()).rev() {
            let dim = self.shape[idx];
            let stride = self.stride[idx];
            if dim > 1 && stride != expected {
                return false;
            }
            expected = expected.saturating_mul(dim.max(1));
        }
        true
    }

    pub fn is_fortran_contiguous(&self) -> bool {
        if self.shape.len() != self.stride.len() {
            return false;
        }
        let mut expected = 1usize;
        for idx in 0..self.shape.len() {
            let dim = self.shape[idx];
            let stride = self.stride[idx];
            if dim > 1 && stride != expected {
                return false;
            }
            expected = expected.saturating_mul(dim.max(1));
        }
        true
    }
}

fn rebuild_args(args: Object) -> Result<(PthLayout, DType, String, usize)> {
    let mut args = args
        .tuple()
        .map_err(|obj| invalid_pickle(format!("expected tuple args, got {obj:?}")))?;

    let stride = Vec::<usize>::try_from(args.remove(3))
        .map_err(|obj| invalid_pickle(format!("expected tuple stride, got {obj:?}")))?;
    let size = Vec::<usize>::try_from(args.remove(2))
        .map_err(|obj| invalid_pickle(format!("expected tuple size, got {obj:?}")))?;
    let offset = args
        .remove(1)
        .int_or_long()
        .map_err(|obj| invalid_pickle(format!("expected int offset, got {obj:?}")))?
        as usize;

    let storage = args
        .remove(0)
        .persistent_load()
        .map_err(|obj| invalid_pickle(format!("expected persistent load, got {obj:?}")))?;

    let mut storage = storage
        .tuple()
        .map_err(|obj| invalid_pickle(format!("expected storage tuple, got {obj:?}")))?;
    let storage_size = storage
        .remove(4)
        .int_or_long()
        .map_err(|obj| invalid_pickle(format!("expected storage size int, got {obj:?}")))?
        as usize;
    let path = storage
        .remove(2)
        .unicode()
        .map_err(|obj| invalid_pickle(format!("expected storage path, got {obj:?}")))?;

    let (_module_name, class_name) = storage
        .remove(1)
        .class()
        .map_err(|obj| invalid_pickle(format!("expected storage class, got {obj:?}")))?;

    let dtype = match class_name.as_str() {
        "FloatStorage" => DType::F32,
        "DoubleStorage" => DType::F64,
        "HalfStorage" => DType::F16,
        "BFloat16Storage" => DType::BF16,
        "ByteStorage" => DType::U8,
        "LongStorage" => DType::I64,
        other => return Err(PthError::UnsupportedStorageType(other.to_string())),
    };

    let item_size = dtype.size().ok_or(PthError::UnsupportedDType(dtype))?;
    let layout = PthLayout::new(size, stride, offset * item_size);

    Ok((layout, dtype, path, storage_size))
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    pub layout: PthLayout,
    pub path: String,
    pub storage_size: usize,
}

pub fn read_pth_tensor_info(path: &Path, key: Option<&str>) -> Result<Vec<TensorInfo>> {
    let file = std::fs::File::open(path)?;
    let zip_reader = std::io::BufReader::new(file);
    let mut zip = zip::ZipArchive::new(zip_reader)?;

    let file_names = zip
        .file_names()
        .map(|f| f.to_string())
        .collect::<Vec<String>>();

    let mut tensor_infos = vec![];

    for file_name in &file_names {
        if !file_name.ends_with("data.pkl") {
            continue;
        }

        let dir_name = PathBuf::from(
            file_name
                .strip_suffix(".pkl")
                .ok_or_else(|| invalid_pickle("missing .pkl suffix"))?,
        );

        let reader = zip.by_name(file_name)?;
        let mut reader = std::io::BufReader::new(reader);
        let mut stack = Stack::empty();
        stack.read_loop(&mut reader)?;
        let obj = stack.finalize()?;

        let obj = match obj {
            Object::Build { callable, args } => match *callable {
                Object::Reduce { callable, args: _ } => match *callable {
                    Object::Class {
                        module_name,
                        class_name,
                    } if module_name == "__torch__" && class_name == "Module" => *args,
                    _ => continue,
                },
                _ => continue,
            },
            obj => obj,
        };

        let obj = if let Some(key) = key {
            if let Object::Dict(key_values) = obj {
                key_values
                    .into_iter()
                    .find(|(k, _)| *k == Object::Unicode(key.to_owned()))
                    .map(|(_, v)| v)
                    .ok_or_else(|| PthError::MissingKey(key.to_string()))?
            } else {
                obj
            }
        } else {
            obj
        };

        if let Object::Dict(key_values) = obj {
            for (name, value) in key_values {
                if let Some(tensor_info) = value.into_tensor_info(name, &dir_name)? {
                    tensor_infos.push(tensor_info);
                }
            }
        }
    }

    Ok(tensor_infos)
}

pub struct PthTensors {
    tensor_infos: HashMap<String, TensorInfo>,
    path: PathBuf,
}

impl PthTensors {
    pub fn new(path: &Path, key: Option<&str>) -> Result<Self> {
        let tensor_infos = read_pth_tensor_info(path, key)?;
        let tensor_infos = tensor_infos
            .into_iter()
            .map(|ti| (ti.name.to_string(), ti))
            .collect();
        Ok(Self {
            tensor_infos,
            path: path.to_owned(),
        })
    }

    pub fn tensor_infos(&self) -> &HashMap<String, TensorInfo> {
        &self.tensor_infos
    }

    pub fn get_raw_bytes(&self, name: &str) -> Result<Option<Vec<u8>>> {
        let tensor_info = match self.tensor_infos.get(name) {
            Some(info) => info,
            None => return Ok(None),
        };

        if !tensor_info.layout.is_contiguous() && !tensor_info.layout.is_fortran_contiguous() {
            return Err(invalid_pickle(format!(
                "cannot retrieve non-contiguous tensor {}",
                tensor_info.name
            )));
        }

        let mut zip =
            zip::ZipArchive::new(std::io::BufReader::new(std::fs::File::open(&self.path)?))?;
        let mut reader = zip.by_name(&tensor_info.path)?;

        let start_offset = tensor_info.layout.start_offset();
        if start_offset > 0 {
            std::io::copy(
                &mut reader.by_ref().take(start_offset as u64),
                &mut std::io::sink(),
            )?;
        }

        let elem_size = tensor_info
            .dtype
            .size()
            .ok_or(PthError::UnsupportedDType(tensor_info.dtype))?;
        let numel = tensor_info.layout.num_elements();
        let byte_len = numel.saturating_mul(elem_size);

        let mut raw = vec![0u8; byte_len];
        reader.read_exact(&mut raw)?;

        if tensor_info.layout.rank() > 1 && tensor_info.layout.is_fortran_contiguous() {
            raw = fortran_to_c_bytes(&raw, tensor_info.layout.shape(), elem_size);
        }

        Ok(Some(raw))
    }
}

fn fortran_to_c_bytes(src: &[u8], shape: &[usize], elem_size: usize) -> Vec<u8> {
    if shape.is_empty() || shape.contains(&0) {
        return vec![];
    }

    let rank = shape.len();
    let numel: usize = shape.iter().product();
    let mut out = vec![0u8; src.len()];
    let mut coords = vec![0usize; rank];

    for out_idx in 0..numel {
        let mut tmp = out_idx;
        for dim in (0..rank).rev() {
            let d = shape[dim];
            coords[dim] = tmp % d;
            tmp /= d;
        }

        let mut in_idx = 0usize;
        let mut stride = 1usize;
        for (coord, dim) in coords.iter().zip(shape.iter()) {
            in_idx += coord * stride;
            stride *= *dim;
        }

        let src_start = in_idx * elem_size;
        let dst_start = out_idx * elem_size;
        out[dst_start..dst_start + elem_size]
            .copy_from_slice(&src[src_start..src_start + elem_size]);
    }

    out
}
