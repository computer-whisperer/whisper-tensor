use whisper_tensor::DynRank;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::dtype::DType;

#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;
#[cfg(target_arch = "wasm32")]
use tokio::sync::mpsc;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::spawn_local;

#[derive(Clone, Debug)]
pub(crate) struct PickedAudioFile {
    pub name: String,
    pub bytes: Vec<u8>,
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn pick_audio_file_native() -> Result<Option<PickedAudioFile>, String> {
    let Some(path) = rfd::FileDialog::new()
        .add_filter("Waveform audio", &["wav"])
        .pick_file()
    else {
        return Ok(None);
    };

    let bytes =
        std::fs::read(&path).map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let name = path
        .file_name()
        .map(|x| x.to_string_lossy().to_string())
        .unwrap_or_else(|| path.display().to_string());
    Ok(Some(PickedAudioFile { name, bytes }))
}

#[cfg(target_arch = "wasm32")]
pub(crate) type WebAudioFilePickReceiver = mpsc::UnboundedReceiver<Result<PickedAudioFile, String>>;

#[cfg(target_arch = "wasm32")]
pub(crate) fn start_audio_file_pick_web() -> WebAudioFilePickReceiver {
    let (tx, rx) = mpsc::unbounded_channel();
    spawn_local(async move {
        let result = match rfd::AsyncFileDialog::new()
            .add_filter("Waveform audio", &["wav"])
            .pick_file()
            .await
        {
            Some(file) => {
                let name = file.file_name();
                let bytes = file.read().await;
                Ok(PickedAudioFile { name, bytes })
            }
            None => Err("file selection canceled".to_string()),
        };
        let _ = tx.send(result);
    });
    rx
}

pub(crate) fn decode_wav_bytes_to_mono_f32(
    bytes: &[u8],
    target_sample_rate_hz: u32,
) -> Result<Vec<f32>, String> {
    if target_sample_rate_hz == 0 {
        return Err("target sample rate must be greater than zero".to_string());
    }

    let cursor = std::io::Cursor::new(bytes);
    let mut reader = hound::WavReader::new(cursor)
        .map_err(|err| format!("failed to open wav payload: {err}"))?;
    let spec = reader.spec();

    let samples = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample.saturating_sub(1))) as f32;
            if max_val <= 0.0 {
                return Err(format!(
                    "unsupported integer bit depth: {}",
                    spec.bits_per_sample
                ));
            }
            reader
                .samples::<i32>()
                .map(|s| {
                    s.map(|x| x as f32 / max_val)
                        .map_err(|err| format!("wav decode failed: {err}"))
                })
                .collect::<Result<Vec<_>, _>>()?
        }
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.map_err(|err| format!("wav decode failed: {err}")))
            .collect::<Result<Vec<_>, _>>()?,
    };

    let mono: Vec<f32> = if spec.channels > 1 {
        samples
            .chunks(spec.channels as usize)
            .filter_map(|chunk| chunk.first().copied())
            .collect()
    } else {
        samples
    };

    if mono.is_empty() {
        return Err("wav contains no audio samples".to_string());
    }

    if spec.sample_rate == target_sample_rate_hz {
        Ok(mono)
    } else {
        Ok(linear_resample_mono(
            &mono,
            spec.sample_rate,
            target_sample_rate_hz,
        ))
    }
}

fn linear_resample_mono(
    samples: &[f32],
    src_sample_rate_hz: u32,
    dst_sample_rate_hz: u32,
) -> Vec<f32> {
    if samples.is_empty() || src_sample_rate_hz == 0 || dst_sample_rate_hz == 0 {
        return Vec::new();
    }
    if src_sample_rate_hz == dst_sample_rate_hz {
        return samples.to_vec();
    }

    let ratio = src_sample_rate_hz as f64 / dst_sample_rate_hz as f64;
    let out_len = (samples.len() as f64 / ratio).round().max(1.0) as usize;

    (0..out_len)
        .map(|i| {
            let pos = i as f64 * ratio;
            let idx = pos.floor() as usize;
            let frac = pos - idx as f64;
            let a = samples[idx.min(samples.len() - 1)];
            let b = samples[(idx + 1).min(samples.len() - 1)];
            a + (b - a) * frac as f32
        })
        .collect()
}

pub(crate) fn tensor_to_audio_samples(
    audio_tensor: &NDArrayNumericTensor<DynRank>,
) -> Result<Vec<f32>, String> {
    let f32_tensor = audio_tensor
        .cast(DType::F32)
        .map_err(|err| format!("failed to cast audio tensor to f32: {err}"))?;
    let flat = f32_tensor.flatten();
    flat.try_to_vec()
        .map_err(|err| format!("failed to flatten audio tensor: {err}"))
}

pub(crate) fn play_audio_samples(samples: &[f32], sample_rate_hz: u32) -> Result<(), String> {
    if sample_rate_hz == 0 {
        return Err("sample rate must be greater than zero".to_string());
    }

    #[cfg(target_arch = "wasm32")]
    {
        let wav_data = encode_wav_i16_mono(samples, sample_rate_hz);
        js_play_wav_bytes(&wav_data);
        Ok(())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        play_audio_samples_native(samples, sample_rate_hz)
    }
}

pub(crate) fn stop_audio_playback() {
    #[cfg(target_arch = "wasm32")]
    {
        js_stop_audio_playback();
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        stop_audio_playback_native();
    }
}

pub(crate) fn download_audio_wav(samples: &[f32], sample_rate_hz: u32) -> Result<(), String> {
    let wav_data = encode_wav_i16_mono(samples, sample_rate_hz);
    #[cfg(target_arch = "wasm32")]
    {
        trigger_browser_download("generated_audio.wav", &wav_data, "audio/wav")
            .map_err(|err| format!("browser download failed: {err:?}"))?;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::fs::write("generated_audio.wav", &wav_data)
            .map_err(|err| format!("failed to write generated_audio.wav: {err}"))?;
    }
    Ok(())
}

fn encode_wav_i16_mono(samples: &[f32], sample_rate_hz: u32) -> Vec<u8> {
    let bytes_per_sample = 2u32;
    let data_size = (samples.len() as u32) * bytes_per_sample;
    let chunk_size = 36u32 + data_size;
    let byte_rate = sample_rate_hz * bytes_per_sample;
    let block_align = bytes_per_sample as u16;
    let bits_per_sample = 16u16;

    let mut out = Vec::with_capacity((44 + data_size as usize).max(44));

    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&chunk_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");

    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate_hz.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());

    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());

    for &sample in samples {
        let sample = sample.clamp(-1.0, 1.0);
        let pcm = (sample * i16::MAX as f32).round() as i16;
        out.extend_from_slice(&pcm.to_le_bytes());
    }

    out
}

#[cfg(not(target_arch = "wasm32"))]
enum NativeAudioCommand {
    Play {
        samples: Vec<f32>,
        sample_rate_hz: u32,
    },
    Stop,
}

#[cfg(not(target_arch = "wasm32"))]
static NATIVE_AUDIO_TX: OnceLock<std::sync::mpsc::Sender<NativeAudioCommand>> = OnceLock::new();

#[cfg(not(target_arch = "wasm32"))]
fn ensure_native_audio_thread()
-> Result<&'static std::sync::mpsc::Sender<NativeAudioCommand>, String> {
    use cpal::traits::{DeviceTrait, HostTrait};
    use std::sync::mpsc;
    use std::time::Duration;

    if let Some(tx) = NATIVE_AUDIO_TX.get() {
        return Ok(tx);
    }

    let (cmd_tx, cmd_rx) = mpsc::channel::<NativeAudioCommand>();
    let (init_tx, init_rx) = mpsc::channel::<Result<(), String>>();

    std::thread::Builder::new()
        .name("wt-audio-playback".to_string())
        .spawn(move || {
            use rodio::buffer::SamplesBuffer;
            use rodio::{OutputStream, Sink};

            let stream_result = {
                let host = cpal::default_host();
                let default_device = host.default_output_device();
                let default_name = default_device
                    .as_ref()
                    .and_then(|d| d.name().ok())
                    .unwrap_or_else(|| "<none>".to_string());
                let env_preference = std::env::var("WT_AUDIO_DEVICE")
                    .ok()
                    .map(|v| v.trim().to_lowercase())
                    .filter(|v| !v.is_empty());

                let mut named_devices: Vec<(String, cpal::Device)> = host
                    .output_devices()
                    .map(|iter| {
                        iter.filter_map(|d| {
                            d.name().ok().map(|name| (name.to_lowercase(), name, d))
                        })
                        .map(|(_key, display_name, d)| (display_name, d))
                        .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();

                if named_devices.is_empty() {
                    OutputStream::try_default()
                } else {
                    fn rank_device(
                        name: &str,
                        env_preference: Option<&str>,
                        default_name: &str,
                    ) -> u8 {
                        let name_l = name.to_lowercase();
                        let default_l = default_name.to_lowercase();
                        if let Some(pref) = env_preference
                            && name_l.contains(pref)
                        {
                            return 0;
                        }
                        if default_l != "<none>" && default_l != "default" && name_l == default_l {
                            return 1;
                        }
                        if name_l.contains("pipewire") {
                            return 2;
                        }
                        if name_l.contains("pulse") {
                            return 3;
                        }
                        if name_l.contains("default:card=generic") {
                            return 4;
                        }
                        if name_l.contains("front:card=generic") {
                            return 5;
                        }
                        if name_l == "default" {
                            return 6;
                        }
                        if name_l == "sysdefault" {
                            return 7;
                        }
                        8
                    }

                    named_devices.sort_by_key(|(name, _)| {
                        (
                            rank_device(name, env_preference.as_deref(), &default_name),
                            name.clone(),
                        )
                    });

                    let mut opened_stream = None;
                    for (name, device) in named_devices {
                        match OutputStream::try_from_device(&device) {
                            Ok(stream) => {
                                log::info!("Selected audio output device: {name}");
                                opened_stream = Some(stream);
                                break;
                            }
                            Err(err) => {
                                log::debug!("Audio device open failed ({name}): {err}");
                            }
                        }
                    }

                    if let Some(stream) = opened_stream {
                        Ok(stream)
                    } else {
                        OutputStream::try_default()
                    }
                }
            };

            let (stream, stream_handle) = match stream_result {
                Ok(v) => v,
                Err(err) => {
                    let _ = init_tx.send(Err(format!("no audio output device: {err}")));
                    return;
                }
            };

            let mut sink = match Sink::try_new(&stream_handle) {
                Ok(sink) => sink,
                Err(err) => {
                    let _ = init_tx.send(Err(format!("failed to create sink: {err}")));
                    return;
                }
            };

            let _ = init_tx.send(Ok(()));

            while let Ok(command) = cmd_rx.recv() {
                match command {
                    NativeAudioCommand::Play {
                        samples,
                        sample_rate_hz,
                    } => {
                        sink.stop();
                        match Sink::try_new(&stream_handle) {
                            Ok(new_sink) => {
                                new_sink.append(SamplesBuffer::new(1, sample_rate_hz, samples));
                                new_sink.play();
                                sink = new_sink;
                            }
                            Err(err) => {
                                log::error!("failed to create sink: {err}");
                            }
                        }
                    }
                    NativeAudioCommand::Stop => {
                        sink.stop();
                    }
                }
            }

            let _stream = stream;
        })
        .map_err(|err| format!("failed to start audio thread: {err}"))?;

    match init_rx.recv_timeout(Duration::from_secs(2)) {
        Ok(Ok(())) => {}
        Ok(Err(err)) => return Err(err),
        Err(err) => return Err(format!("audio thread init timeout: {err}")),
    }

    let _ = NATIVE_AUDIO_TX.set(cmd_tx);
    NATIVE_AUDIO_TX
        .get()
        .ok_or_else(|| "audio thread failed to initialize sender".to_string())
}

#[cfg(not(target_arch = "wasm32"))]
fn play_audio_samples_native(samples: &[f32], sample_rate_hz: u32) -> Result<(), String> {
    if samples.is_empty() {
        return Err("audio buffer is empty".to_string());
    }

    let tx = ensure_native_audio_thread()?;
    tx.send(NativeAudioCommand::Play {
        samples: samples.to_vec(),
        sample_rate_hz,
    })
    .map_err(|err| format!("failed to queue audio playback: {err}"))
}

#[cfg(not(target_arch = "wasm32"))]
fn stop_audio_playback_native() {
    if let Some(tx) = NATIVE_AUDIO_TX.get() {
        let _ = tx.send(NativeAudioCommand::Stop);
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(inline_js = "
let __wtAudioPlayback = { audio: null, url: null };

export function js_play_wav_bytes(bytes) {
    if (__wtAudioPlayback.audio) {
        __wtAudioPlayback.audio.pause();
    }
    if (__wtAudioPlayback.url) {
        URL.revokeObjectURL(__wtAudioPlayback.url);
    }

    const blob = new Blob([bytes], { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    __wtAudioPlayback = { audio, url };
    audio.onended = () => {
        if (__wtAudioPlayback.url === url) {
            URL.revokeObjectURL(url);
            __wtAudioPlayback = { audio: null, url: null };
        }
    };
    const promise = audio.play();
    if (promise && promise.catch) {
        promise.catch((err) => console.error('Audio playback failed', err));
    }
}

export function js_stop_audio_playback() {
    if (__wtAudioPlayback.audio) {
        __wtAudioPlayback.audio.pause();
        __wtAudioPlayback.audio.currentTime = 0.0;
    }
    if (__wtAudioPlayback.url) {
        URL.revokeObjectURL(__wtAudioPlayback.url);
    }
    __wtAudioPlayback = { audio: null, url: null };
}
")]
extern "C" {
    fn js_play_wav_bytes(bytes: &[u8]);
    fn js_stop_audio_playback();
}

#[cfg(target_arch = "wasm32")]
fn trigger_browser_download(
    filename: &str,
    data: &[u8],
    mime_type: &str,
) -> Result<(), wasm_bindgen::JsValue> {
    use wasm_bindgen::JsCast;

    let uint8_array = js_sys::Uint8Array::from(data);
    let array = js_sys::Array::new();
    array.push(&uint8_array.buffer());

    let options = web_sys::BlobPropertyBag::new();
    options.set_type(mime_type);
    let blob = web_sys::Blob::new_with_u8_array_sequence_and_options(&array, &options)?;

    let url = web_sys::Url::create_object_url_with_blob(&blob)?;

    let window = web_sys::window().ok_or("no window")?;
    let document = window.document().ok_or("no document")?;
    let a = document
        .create_element("a")?
        .dyn_into::<web_sys::HtmlAnchorElement>()?;
    a.set_href(&url);
    a.set_download(filename);
    a.style().set_property("display", "none")?;
    document.body().ok_or("no body")?.append_child(&a)?;
    a.click();
    a.remove();
    web_sys::Url::revoke_object_url(&url)?;
    Ok(())
}
