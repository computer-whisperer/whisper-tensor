const PHONEMIZER_BACKEND_ENV: &str = "WHISPER_TENSOR_PHONEMIZER";

trait PhonemizerBackend {
    fn id(&self) -> &'static str;
    fn phonemize_to_ipa(&self, text: &str, voice: &str) -> Result<String, String>;
}

struct NativeRustPhonemizer;
static NATIVE_RUST_PHONEMIZER: NativeRustPhonemizer = NativeRustPhonemizer;

impl PhonemizerBackend for NativeRustPhonemizer {
    fn id(&self) -> &'static str {
        "rust-native"
    }

    fn phonemize_to_ipa(&self, text: &str, _voice: &str) -> Result<String, String> {
        Ok(native_text_to_ipa(text))
    }
}

#[cfg(feature = "espeak")]
struct EspeakPhonemizer;

#[cfg(feature = "espeak")]
static ESPEAK_PHONEMIZER: EspeakPhonemizer = EspeakPhonemizer;

#[cfg(feature = "espeak")]
impl PhonemizerBackend for EspeakPhonemizer {
    fn id(&self) -> &'static str {
        "espeak"
    }

    fn phonemize_to_ipa(&self, text: &str, voice: &str) -> Result<String, String> {
        let sentences = espeak_rs::text_to_phonemes(text, voice, None, true, false)
            .map_err(|e| format!("espeak-ng phonemization failed: {e}"))?;
        Ok(sentences.join(" "))
    }
}

pub fn text_to_piper_phonemes(text: &str, voice: &str) -> Result<String, String> {
    let backend = select_backend(voice)?;
    backend
        .phonemize_to_ipa(text, voice)
        .map_err(|e| format!("{} phonemization failed: {e}", backend.id()))
}

pub fn text_to_kokoro_phonemes(text: &str, voice: &str) -> Result<String, String> {
    let ipa = text_to_piper_phonemes(text, voice)?;
    Ok(espeak_to_misaki(&ipa))
}

fn select_backend(voice: &str) -> Result<&'static dyn PhonemizerBackend, String> {
    if let Ok(value) = std::env::var(PHONEMIZER_BACKEND_ENV) {
        let normalized = value.trim().to_ascii_lowercase();
        if !normalized.is_empty() && normalized != "auto" {
            return match normalized.as_str() {
                "rust" | "native" | "rust-native" => Ok(&NATIVE_RUST_PHONEMIZER),
                "espeak" => {
                    #[cfg(feature = "espeak")]
                    {
                        Ok(&ESPEAK_PHONEMIZER)
                    }
                    #[cfg(not(feature = "espeak"))]
                    {
                        Err(format!(
                            "{}=espeak requested, but the espeak feature is disabled",
                            PHONEMIZER_BACKEND_ENV
                        ))
                    }
                }
                other => Err(format!(
                    "invalid {} value '{other}', expected one of: auto, rust, native, espeak",
                    PHONEMIZER_BACKEND_ENV
                )),
            };
        }
    }

    if !voice.to_ascii_lowercase().starts_with("en") {
        #[cfg(feature = "espeak")]
        {
            return Ok(&ESPEAK_PHONEMIZER);
        }
    }

    Ok(&NATIVE_RUST_PHONEMIZER)
}

fn native_text_to_ipa(text: &str) -> String {
    let mut tokens = Vec::<String>::new();
    let mut current_word = String::new();

    let flush_word = |word: &mut String, out: &mut Vec<String>| {
        if word.is_empty() {
            return;
        }
        let phonemes = native_word_to_ipa(word);
        if !phonemes.is_empty() {
            out.push(phonemes);
        }
        word.clear();
    };

    for ch in text.chars() {
        if ch.is_ascii_alphabetic() || ch == '\'' {
            current_word.push(ch.to_ascii_lowercase());
            continue;
        }

        flush_word(&mut current_word, &mut tokens);

        if let Some(name) = digit_to_word(ch) {
            let digit_phonemes = native_word_to_ipa(name);
            if !digit_phonemes.is_empty() {
                tokens.push(digit_phonemes);
            }
            continue;
        }

        if is_supported_punctuation(ch) {
            tokens.push(ch.to_string());
        }
    }

    flush_word(&mut current_word, &mut tokens);

    normalize_ipa_output(&tokens.join(" "))
}

fn normalize_ipa_output(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut previous_was_space = true;

    for ch in raw.chars() {
        if ch.is_whitespace() {
            if !previous_was_space {
                out.push(' ');
                previous_was_space = true;
            }
        } else {
            out.push(ch);
            previous_was_space = false;
        }
    }

    out.trim().to_string()
}

fn native_word_to_ipa(word: &str) -> String {
    let trimmed = word.trim_matches('\'');
    if trimmed.is_empty() {
        return String::new();
    }

    if let Some(ipa) = native_lexicon_override(trimmed) {
        return ipa.to_string();
    }

    let mut out = String::new();
    let mut i = 0usize;
    while i < trimmed.len() {
        let rest = &trimmed[i..];

        if rest.starts_with("tion") {
            out.push_str("ʃən");
            i += 4;
            continue;
        }
        if rest.starts_with("sion") {
            out.push_str("ʒən");
            i += 4;
            continue;
        }
        if rest.starts_with("tch") {
            out.push('ʧ');
            i += 3;
            continue;
        }
        if rest.starts_with("dge") {
            out.push('ʤ');
            i += 3;
            continue;
        }
        if rest.starts_with("igh") {
            out.push_str("aɪ");
            i += 3;
            continue;
        }

        if rest.starts_with("ch") {
            out.push('ʧ');
            i += 2;
            continue;
        }
        if rest.starts_with("sh") {
            out.push('ʃ');
            i += 2;
            continue;
        }
        if rest.starts_with("zh") {
            out.push('ʒ');
            i += 2;
            continue;
        }
        if rest.starts_with("th") {
            if use_voiced_th(trimmed) {
                out.push('ð');
            } else {
                out.push('θ');
            }
            i += 2;
            continue;
        }
        if rest.starts_with("ph") {
            out.push('f');
            i += 2;
            continue;
        }
        if rest.starts_with("ng") {
            out.push('ŋ');
            i += 2;
            continue;
        }
        if rest.starts_with("qu") {
            out.push_str("kw");
            i += 2;
            continue;
        }
        if rest.starts_with("ck") {
            out.push('k');
            i += 2;
            continue;
        }
        if rest.starts_with("wr") {
            out.push('ɹ');
            i += 2;
            continue;
        }
        if rest.starts_with("wh") {
            out.push('w');
            i += 2;
            continue;
        }
        if rest.starts_with("kn") && i == 0 {
            out.push('n');
            i += 2;
            continue;
        }
        if rest.starts_with("gn") && i == 0 {
            out.push('n');
            i += 2;
            continue;
        }

        if rest.starts_with("oo") {
            out.push('u');
            i += 2;
            continue;
        }
        if rest.starts_with("ee") || rest.starts_with("ea") {
            out.push('i');
            i += 2;
            continue;
        }
        if rest.starts_with("oa") {
            out.push('o');
            i += 2;
            continue;
        }
        if rest.starts_with("ai")
            || rest.starts_with("ay")
            || rest.starts_with("ei")
            || rest.starts_with("ey")
        {
            out.push_str("eɪ");
            i += 2;
            continue;
        }
        if rest.starts_with("oi") || rest.starts_with("oy") {
            out.push_str("ɔɪ");
            i += 2;
            continue;
        }
        if rest.starts_with("ou") || rest.starts_with("ow") {
            out.push_str("aʊ");
            i += 2;
            continue;
        }
        if rest.starts_with("au") || rest.starts_with("aw") {
            out.push('ɔ');
            i += 2;
            continue;
        }
        if rest.starts_with("ar") {
            out.push_str("ɑɹ");
            i += 2;
            continue;
        }
        if rest.starts_with("or") {
            out.push_str("ɔɹ");
            i += 2;
            continue;
        }
        if rest.starts_with("er") || rest.starts_with("ir") || rest.starts_with("ur") {
            out.push_str("əɹ");
            i += 2;
            continue;
        }

        let ch = trimmed.as_bytes()[i] as char;
        if ch == 'e' && i + 1 == trimmed.len() && i > 0 {
            i += 1;
            continue;
        }

        match ch {
            'a' => out.push('æ'),
            'b' => out.push('b'),
            'c' => {
                let next = trimmed.as_bytes().get(i + 1).copied().map(char::from);
                if matches!(next, Some('e' | 'i' | 'y')) {
                    out.push('s');
                } else {
                    out.push('k');
                }
            }
            'd' => out.push('d'),
            'e' => out.push('ɛ'),
            'f' => out.push('f'),
            'g' => {
                let next = trimmed.as_bytes().get(i + 1).copied().map(char::from);
                if matches!(next, Some('e' | 'i' | 'y')) {
                    out.push('ʤ');
                } else {
                    out.push('g');
                }
            }
            'h' => out.push('h'),
            'i' => out.push('ɪ'),
            'j' => out.push('ʤ'),
            'k' => out.push('k'),
            'l' => out.push('l'),
            'm' => out.push('m'),
            'n' => out.push('n'),
            'o' => out.push('ɑ'),
            'p' => out.push('p'),
            'q' => out.push('k'),
            'r' => out.push('ɹ'),
            's' => out.push('s'),
            't' => out.push('t'),
            'u' => out.push('ʌ'),
            'v' => out.push('v'),
            'w' => out.push('w'),
            'x' => out.push_str("ks"),
            'y' => {
                if i == 0 {
                    out.push('j');
                } else {
                    out.push('i');
                }
            }
            'z' => out.push('z'),
            _ => {}
        }

        i += 1;
    }

    out
}

fn native_lexicon_override(word: &str) -> Option<&'static str> {
    match word {
        "a" => Some("ə"),
        "an" => Some("æn"),
        "and" => Some("ænd"),
        "are" => Some("ɑɹ"),
        "as" => Some("æz"),
        "be" => Some("bi"),
        "for" => Some("fɔɹ"),
        "from" => Some("fɹʌm"),
        "have" => Some("hæv"),
        "i" => Some("aɪ"),
        "is" => Some("ɪz"),
        "of" => Some("əv"),
        "the" => Some("ðə"),
        "their" => Some("ðɛɹ"),
        "them" => Some("ðɛm"),
        "then" => Some("ðɛn"),
        "there" => Some("ðɛɹ"),
        "these" => Some("ðiz"),
        "they" => Some("ðeɪ"),
        "this" => Some("ðɪs"),
        "those" => Some("ðoz"),
        "to" => Some("tu"),
        "zero" => Some("zɪɹo"),
        "one" => Some("wʌn"),
        "two" => Some("tu"),
        "three" => Some("θɹi"),
        "four" => Some("fɔɹ"),
        "five" => Some("faɪv"),
        "six" => Some("sɪks"),
        "seven" => Some("sɛvən"),
        "eight" => Some("eɪt"),
        "nine" => Some("naɪn"),
        "was" => Some("wʌz"),
        "we" => Some("wi"),
        "were" => Some("wəɹ"),
        "with" => Some("wɪθ"),
        "you" => Some("ju"),
        "your" => Some("jɔɹ"),
        _ => None,
    }
}

fn use_voiced_th(word: &str) -> bool {
    matches!(
        word,
        "the"
            | "this"
            | "that"
            | "these"
            | "those"
            | "them"
            | "they"
            | "there"
            | "then"
            | "than"
            | "though"
            | "thus"
    )
}

fn digit_to_word(ch: char) -> Option<&'static str> {
    match ch {
        '0' => Some("zero"),
        '1' => Some("one"),
        '2' => Some("two"),
        '3' => Some("three"),
        '4' => Some("four"),
        '5' => Some("five"),
        '6' => Some("six"),
        '7' => Some("seven"),
        '8' => Some("eight"),
        '9' => Some("nine"),
        _ => None,
    }
}

fn is_supported_punctuation(ch: char) -> bool {
    matches!(ch, '.' | ',' | '!' | '?' | ';' | ':')
}

fn espeak_to_misaki(ipa: &str) -> String {
    // E2M replacements, applied longest-first.
    static E2M: &[(&str, &str)] = &[
        ("a\u{0361}\u{026a}", "I"),
        ("a\u{0361}\u{028a}", "W"),
        ("d\u{0361}\u{0292}", "\u{02A4}"),
        ("e\u{0361}\u{026a}", "A"),
        ("t\u{0361}\u{0283}", "\u{02A7}"),
        ("\u{0254}\u{0361}\u{026a}", "Y"),
        ("o\u{0361}\u{028a}", "O"),
        ("a\u{026a}", "I"),
        ("a\u{028a}", "W"),
        ("d\u{0292}", "\u{02A4}"),
        ("e\u{026a}", "A"),
        ("t\u{0283}", "\u{02A7}"),
        ("\u{0254}\u{026a}", "Y"),
        ("o\u{028a}", "O"),
        ("\u{0294}\u{02cc}n\u{0329}", "t\u{1d4a}n"),
        ("\u{0294}n", "t\u{1d4a}n"),
        ("\u{0259}\u{0361}l", "\u{1d4a}l"),
        ("\u{0259}l", "\u{1d4a}l"),
        ("\u{025a}", "\u{0259}\u{0279}"),
        ("\u{025c}\u{02d0}\u{0279}", "\u{025c}\u{0279}"),
        ("\u{025c}\u{02d0}", "\u{025c}\u{0279}"),
        ("\u{026a}\u{0259}", "i\u{0259}"),
        ("e", "A"),
        ("r", "\u{0279}"),
        ("x", "k"),
        ("\u{00e7}", "k"),
        ("\u{0250}", "\u{0259}"),
        ("\u{026c}", "l"),
        ("\u{0294}", "t"),
        ("o", "\u{0254}"),
        ("\u{027e}", "T"),
    ];

    let mut result = ipa.to_string();
    result = result.replace('\u{0303}', "");
    result = result.replace('\u{02b2}', "");
    for &(from, to) in E2M {
        result = result.replace(from, to);
    }
    result = result.replace('\u{02d0}', "");
    result = result.replace('\u{0329}', "");
    result
}

#[cfg(test)]
mod tests {
    use super::{
        espeak_to_misaki, native_text_to_ipa, native_word_to_ipa, select_backend,
        text_to_piper_phonemes,
    };

    #[test]
    fn native_word_golden_cases() {
        let cases = [
            ("the", "ðə"),
            ("ship", "ʃɪp"),
            ("phone", "fɑn"),
            ("thing", "θɪŋ"),
            ("queen", "kwin"),
            ("can't", "kænt"),
            ("knock", "nɑk"),
        ];
        for (word, expected) in cases {
            assert_eq!(native_word_to_ipa(word), expected, "{word}");
        }
    }

    #[test]
    fn native_sentence_golden_case() {
        let out = native_text_to_ipa("The quick brown fox jumps over 2 lazy dogs!");
        assert_eq!(out, "ðə kwɪk bɹaʊn fɑks ʤʌmps ɑvəɹ tu læzi dɑgs !");
    }

    #[test]
    fn native_text_normalizes_whitespace_and_drops_unknown_punctuation() {
        let out = native_text_to_ipa("  Hello   (world)\n");
        assert_eq!(out, "hɛllɑ wɔɹld");
    }

    #[test]
    fn piper_phonemes_use_default_backend() {
        let out = text_to_piper_phonemes("Hello", "en-us").unwrap();
        assert!(!out.is_empty());
    }

    #[test]
    fn kokoro_transform_maps_common_symbols() {
        assert_eq!(espeak_to_misaki("aɪ oʊ ɾ"), "I O T");
    }

    #[test]
    fn backend_selection_defaults_to_native_for_english() {
        let backend = select_backend("en-us").unwrap();
        assert_eq!(backend.id(), "rust-native");
    }

    #[cfg(not(feature = "espeak"))]
    #[test]
    fn backend_selection_non_english_without_espeak_uses_native() {
        let backend = select_backend("ja").unwrap();
        assert_eq!(backend.id(), "rust-native");
    }

    #[cfg(feature = "espeak")]
    #[test]
    fn backend_selection_non_english_with_espeak_prefers_espeak() {
        let backend = select_backend("ja").unwrap();
        assert_eq!(backend.id(), "espeak");
    }
}
