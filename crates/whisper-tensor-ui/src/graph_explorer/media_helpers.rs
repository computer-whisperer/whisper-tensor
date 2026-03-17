use egui::{Color32, ColorImage};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

pub(super) fn save_image_to_download(color_image: &ColorImage) {
    let [w, h] = color_image.size;
    let bmp_data = encode_bmp(w, h, &color_image.pixels);
    #[cfg(target_arch = "wasm32")]
    {
        let _ = trigger_browser_download("generated_image.bmp", &bmp_data, "image/bmp");
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = std::fs::write("generated_image.bmp", &bmp_data);
    }
}

pub(super) fn copy_image_to_clipboard(color_image: &ColorImage) {
    let [w, h] = color_image.size;
    let mut rgba = Vec::with_capacity(w * h * 4);
    for pixel in &color_image.pixels {
        rgba.push(pixel.r());
        rgba.push(pixel.g());
        rgba.push(pixel.b());
        rgba.push(pixel.a());
    }
    #[cfg(target_arch = "wasm32")]
    super::js_copy_image_to_clipboard(&rgba, w as u32, h as u32);
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (&rgba, w, h); // TODO: native clipboard support
    }
}

fn encode_bmp(w: usize, h: usize, pixels: &[Color32]) -> Vec<u8> {
    let row_size = w * 3;
    let row_padding = (4 - (row_size % 4)) % 4;
    let padded_row = row_size + row_padding;
    let pixel_data_size = padded_row * h;
    let file_size = 54 + pixel_data_size;

    let mut data = Vec::with_capacity(file_size);

    // BMP file header (14 bytes)
    data.extend_from_slice(b"BM");
    data.extend_from_slice(&(file_size as u32).to_le_bytes());
    data.extend_from_slice(&0u16.to_le_bytes()); // reserved
    data.extend_from_slice(&0u16.to_le_bytes()); // reserved
    data.extend_from_slice(&54u32.to_le_bytes()); // pixel data offset

    // DIB header (40 bytes)
    data.extend_from_slice(&40u32.to_le_bytes()); // header size
    data.extend_from_slice(&(w as i32).to_le_bytes());
    data.extend_from_slice(&(h as i32).to_le_bytes());
    data.extend_from_slice(&1u16.to_le_bytes()); // planes
    data.extend_from_slice(&24u16.to_le_bytes()); // bits per pixel
    data.extend_from_slice(&0u32.to_le_bytes()); // no compression
    data.extend_from_slice(&(pixel_data_size as u32).to_le_bytes());
    data.extend_from_slice(&2835u32.to_le_bytes()); // h resolution (72 dpi)
    data.extend_from_slice(&2835u32.to_le_bytes()); // v resolution
    data.extend_from_slice(&0u32.to_le_bytes()); // colors in palette
    data.extend_from_slice(&0u32.to_le_bytes()); // important colors

    // Pixel data (bottom-up, BGR)
    for y in (0..h).rev() {
        for x in 0..w {
            let c = pixels[y * w + x];
            data.push(c.b());
            data.push(c.g());
            data.push(c.r());
        }
        data.resize(data.len() + row_padding, 0);
    }

    data
}

#[cfg(target_arch = "wasm32")]
pub(super) fn trigger_browser_download(
    filename: &str,
    data: &[u8],
    mime_type: &str,
) -> Result<(), wasm_bindgen::JsValue> {
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
