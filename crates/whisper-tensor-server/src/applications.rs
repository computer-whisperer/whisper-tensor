use crate::LoadedModelId;
use whisper_tensor::tokenizer::AnyTokenizer;

struct TextInferenceSession {
    model_id: LoadedModelId,
    tokenizer: AnyTokenizer,
}
