use whisper_tensor::tokenizer::AnyTokenizer;
use crate::LoadedModelId;

struct TextInferenceSession {
    model_id: LoadedModelId,
    tokenizer: AnyTokenizer,
}