pub fn text_to_piper_phonemes(text: &str, voice: &str) -> Result<String, String> {
    #[cfg(feature = "espeak")]
    {
        let sentences = espeak_rs::text_to_phonemes(text, voice, None, true, false)
            .map_err(|e| format!("espeak-ng phonemization failed: {e}"))?;
        Ok(sentences.join(" "))
    }
    #[cfg(not(feature = "espeak"))]
    {
        let _ = (text, voice);
        Err("espeak feature is not enabled".to_string())
    }
}

pub fn text_to_kokoro_phonemes(text: &str, voice: &str) -> Result<String, String> {
    let ipa = text_to_piper_phonemes(text, voice)?;
    Ok(espeak_to_misaki(&ipa))
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
