use egui::Color32;
use whisper_tensor_server::{AbbreviatedTensorValue, ScaleParams};

pub(crate) fn build_tensor_swatch(
    abbreviated_tensor: &AbbreviatedTensorValue,
    width: usize,
    height: usize,
) -> Option<Vec<Color32>> {
    if let Some((a, _b)) = &abbreviated_tensor.value {
        if a.len() == width * height {
            Some(colors_from_quantized(a, Colormap::AuroraMuted))
        } else if a.len() < width {
            None
        } else {
            let mut values = vec![];
            let scale = (width * height) / (a.len());
            for i in 0..width * height {
                values.push(a[(i / scale) % a.len()]);
            }
            Some(colors_from_quantized(&values, Colormap::AuroraMuted))
        }
    } else {
        None
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Colormap {
    Diverging,
    Gray,
    Magma,
    AuroraMuted,
}

// -------- public entry points --------

/// Map raw f32 values -> Color32 using provided scaling parameters.
pub fn colors_from_values(vals: &[f32], params: ScaleParams, cmap: Colormap) -> Vec<Color32> {
    let denom = (params.vmax - params.vmin).max(1e-12);
    let mut out = Vec::with_capacity(vals.len());
    for &v in vals {
        let mut t = (v - params.vmin) / denom;
        if !t.is_finite() {
            t = 0.5;
        } // neutral if NaN/Inf slips through
        t = clamp01(t);
        out.push(map_colormap(t, cmap));
    }
    out
}

/// If the server already quantized to u8 (0..255), use this.
pub fn colors_from_quantized(q: &[u8], cmap: Colormap) -> Vec<Color32> {
    let mut out = Vec::with_capacity(q.len());
    for &b in q {
        let t = (b as f32) * (1.0 / 255.0);
        out.push(map_colormap(t, cmap));
    }
    out
}

// -------- colormaps --------

#[inline]
fn map_colormap(t: f32, cmap: Colormap) -> Color32 {
    match cmap {
        Colormap::Gray => cmap_gray(t),
        Colormap::Magma => cmap_magma_like(t),
        Colormap::Diverging => cmap_diverging_blue_white_red(t),
        Colormap::AuroraMuted => cmap_aurora_muted(t),
    }
}

#[inline]
fn cmap_gray(t: f32) -> Color32 {
    let g = to_byte(t);
    Color32::from_rgb(g, g, g)
}

/// Simple diverging map: blue -> white -> red, centered at 0.5
fn cmap_diverging_blue_white_red(t: f32) -> Color32 {
    let u = clamp01(t);
    let (r, g, b) = if u < 0.5 {
        // low side: bluish
        let k = u / 0.5; // 0..1
        (
            lerp_byte(230, 40, k),
            lerp_byte(230, 110, k),
            lerp_byte(230, 250, k),
        )
    } else {
        // high side: reddish
        let k = (u - 0.5) / 0.5;
        (
            lerp_byte(230, 250, k),
            lerp_byte(230, 80, k),
            lerp_byte(230, 60, k),
        )
    };
    Color32::from_rgb(r, g, b)
}

/// Lightweight “magma-ish” sequential map for 0..1
fn cmap_magma_like(t: f32) -> Color32 {
    let u = clamp01(t);
    let r = (255.0 * u.powf(0.5)) as f32;
    let g = (255.0 * u.powf(1.2) * 0.6) as f32;
    let b = (255.0 * u.powf(2.0) * 0.2) as f32;
    Color32::from_rgb(sat_u8(r), sat_u8(g), sat_u8(b))
}

fn cmap_aurora_muted(t: f32) -> Color32 {
    // Control points in sRGB 0..255; deliberately muted & dark-mode friendly
    const STOPS: &[(f32, [u8; 3])] = &[
        (0.00, [0x0E, 0x14, 0x20]), // deep indigo
        (0.42, [0x22, 0x40, 0x4A]), // slate-teal
        (0.72, [0x50, 0x80, 0x80]), // muted teal
        (1.00, [0xA6, 0xBA, 0xA0]), // sage (not too bright)
    ];
    let u = clamp01(t);

    // Find segment
    let mut i = 0;
    while i + 1 < STOPS.len() && u > STOPS[i + 1].0 {
        i += 1;
    }
    if i + 1 == STOPS.len() {
        i -= 1;
    } // guard

    let (t0, c0) = STOPS[i];
    let (t1, c1) = STOPS[i + 1];
    let w = if t1 > t0 { (u - t0) / (t1 - t0) } else { 0.0 };

    // Interpolate in **linear sRGB** then convert back to sRGB for display
    let a = srgb8_to_linear3(c0);
    let b = srgb8_to_linear3(c1);
    let rgb_lin = [
        lerp(a[0], b[0], w),
        lerp(a[1], b[1], w),
        lerp(a[2], b[2], w),
    ];
    let rgb8 = linear_to_srgb8(rgb_lin);
    Color32::from_rgb(rgb8[0], rgb8[1], rgb8[2])
}

// -------- helpers --------

#[inline]
fn clamp01(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

#[inline]
fn to_byte(t01: f32) -> u8 {
    // round-to-nearest-even for determinism
    let x = (t01 * 255.0).clamp(0.0, 255.0);
    let f = x.floor();
    let r = x - f;
    let mut n = if r > 0.5 {
        f as i32 + 1
    } else if r < 0.5 {
        f as i32
    } else {
        let fi = f as i32;
        if fi & 1 == 0 { fi } else { fi + 1 }
    };
    if n < 0 {
        n = 0
    } else if n > 255 {
        n = 255
    }
    n as u8
}

#[inline]
fn sat_u8(x: f32) -> u8 {
    let y = x.round().clamp(0.0, 255.0);
    y as u8
}

#[inline]
fn lerp_byte(a: u8, b: u8, t: f32) -> u8 {
    let t = clamp01(t);
    let v = (a as f32) + (b as f32 - a as f32) * t;
    sat_u8(v)
}
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline]
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}
#[inline]
fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}
#[inline]
fn srgb8_to_linear3(rgb: [u8; 3]) -> [f32; 3] {
    [
        srgb_to_linear(rgb[0] as f32 / 255.0),
        srgb_to_linear(rgb[1] as f32 / 255.0),
        srgb_to_linear(rgb[2] as f32 / 255.0),
    ]
}
#[inline]
fn linear_to_srgb8(rgb_lin: [f32; 3]) -> [u8; 3] {
    let r = (linear_to_srgb(rgb_lin[0]) * 255.0)
        .round()
        .clamp(0.0, 255.0) as u8;
    let g = (linear_to_srgb(rgb_lin[1]) * 255.0)
        .round()
        .clamp(0.0, 255.0) as u8;
    let b = (linear_to_srgb(rgb_lin[2]) * 255.0)
        .round()
        .clamp(0.0, 255.0) as u8;
    [r, g, b]
}
