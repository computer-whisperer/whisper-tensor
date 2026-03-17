use whisper_tensor::interfaces::KokoroVoiceEmbedding;

pub(super) fn ensure_kokoro_voice_selection(
    selected: &mut Option<String>,
    voices: &[KokoroVoiceEmbedding],
    default_voice: Option<&str>,
) {
    let selected_valid = selected
        .as_ref()
        .is_some_and(|name| voices.iter().any(|v| v.name == *name));
    if selected_valid {
        return;
    }
    if let Some(default_voice) = default_voice
        && voices.iter().any(|v| v.name == default_voice)
    {
        *selected = Some(default_voice.to_string());
        return;
    }
    *selected = voices.first().map(|v| v.name.clone());
}

pub(super) fn selected_kokoro_voice<'a>(
    selected: &Option<String>,
    voices: &'a [KokoroVoiceEmbedding],
    default_voice: Option<&str>,
) -> Option<&'a KokoroVoiceEmbedding> {
    if let Some(name) = selected
        && let Some(voice) = voices.iter().find(|v| v.name == *name)
    {
        return Some(voice);
    }
    if let Some(default_voice) = default_voice
        && let Some(voice) = voices.iter().find(|v| v.name == default_voice)
    {
        return Some(voice);
    }
    voices.first()
}
