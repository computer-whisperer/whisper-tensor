use whisper_tensor::DynRank;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use egui::{self, Align, FontId, Id, Response, Ui};
use egui_extras::{Column, TableBuilder};
use whisper_tensor::numeric_scalar::NumericScalarType;

pub fn tensor_view_old(ui: &mut Ui, value: &NDArrayNumericTensor<DynRank>) -> Response {
    let frame = egui::Frame::default()
        .inner_margin(2.0)
        .stroke(ui.visuals().window_stroke);
    frame.show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(format!("{:}, {:?}", value.dtype(), value.shape()));
            if ui.button("Copy Full").clicked() {
                let text = format!("{:#}", value);
                ui.output_mut(|o| o.copied_text = text);
            }
        });
        ui.label(value.to_string());
    }).response
}

/// State you should persist in your egui app.
#[derive(Clone, Debug)]
pub struct TensorViewState {
    /// Which axes are displayed as rows/cols.
    pub row_axis: usize,
    pub col_axis: usize,
    /// For every axis, a fixed index (ignored for row_axis/col_axis).
    pub fixed_idx: Vec<u64>,

    /// Top-left of the visible page for row/col axes.
    pub row_offset: u64,
    pub col_offset: u64,

    /// Desired page size (number of rows/cols shown at once).
    pub rows_per_page: u64,
    pub cols_per_page: u64,

    /// Cell appearance / formatting
    pub row_height: f32,
    pub col_width: f32,
    pub precision: usize,
    pub scientific: bool,
    pub monospace: bool,
    pub auto_axes: bool,
    pub last_shape: Vec<u64>
}

impl TensorViewState {
    pub fn new(shape: &[u64]) -> Self {
        let ndim = shape.len().max(1);
        let row_axis = if ndim >= 2 { ndim - 2 } else { 0 };
        let col_axis = if ndim >= 2 { ndim - 1 } else { 0 };
        let mut fixed_idx = vec![0; ndim];
        Self {
            row_axis,
            col_axis,
            fixed_idx,
            row_offset: 0,
            col_offset: 0,
            rows_per_page: 8,  // good default; TableBuilder virtualizes rows
            cols_per_page: 8,  // we page horizontally
            row_height: 20.0,
            col_width: 60.0,
            precision: 4,
            scientific: false,
            monospace: true,
            auto_axes: true,
            last_shape: shape.to_vec()
        }
    }

    fn clamp_to(&mut self, shape: &[u64]) {
        let n = shape.len();

        // Unknown or scalar handled specially
        if n == 0 {
            self.fixed_idx.clear();
            self.row_axis = 0;
            self.col_axis = 0;
            self.row_offset = 0;
            self.col_offset = 0;
            self.rows_per_page = 1;
            self.cols_per_page = 1;
            return;
        }

        // Keep indices sized to rank
        if self.fixed_idx.len() != n {
            self.fixed_idx.resize(n, 0);
        }

        // Distinct axes for rank >= 2
        self.row_axis = self.row_axis.min(n - 1);
        self.col_axis = self.col_axis.min(n - 1);
        if n >= 2 && self.row_axis == self.col_axis {
            self.col_axis = if self.row_axis + 1 < n { self.row_axis + 1 } else { self.row_axis - 1 };
        }

        // Vector specialization
        if n == 1 {
            let len = shape[0];
            self.row_offset = 0;
            self.rows_per_page = 1;
            self.col_offset = self.col_offset.min(len.saturating_sub(1));
            self.cols_per_page = self.cols_per_page.max(1).min(len);
            return;
        }

        // Normal 2D case
        for (i, fi) in self.fixed_idx.iter_mut().enumerate() {
            *fi = (*fi).min(shape[i].saturating_sub(1));
        }
        let rows_total = shape[self.row_axis];
        let cols_total = shape[self.col_axis];
        self.row_offset = self.row_offset.min(rows_total.saturating_sub(1));
        self.col_offset = self.col_offset.min(cols_total.saturating_sub(1));
        self.rows_per_page = self.rows_per_page.max(1).min(rows_total.max(1));
        self.cols_per_page = self.cols_per_page.max(1).min(cols_total.max(1));
    }
}

impl Default for TensorViewState {
    fn default() -> Self {
        Self {
            row_axis: 0,
            col_axis: 0,
            fixed_idx: Vec::new(),
            row_offset: 0,
            col_offset: 0,
            rows_per_page: 8,
            cols_per_page: 8,
            row_height: 20.0,
            col_width: 60.0,
            precision: 4,
            scientific: false,
            monospace: true,
            auto_axes: true,
            last_shape: Vec::new(),
        }
    }
}

fn non_singleton_axes(shape: &[u64]) -> Vec<usize> {
    shape.iter()
        .enumerate()
        .filter_map(|(i, &d)| if d > 1 { Some(i) } else { None })
        .collect()
}


pub fn tensor_view(
    ui: &mut Ui,
    tensor: &NDArrayNumericTensor<DynRank>,
    state: &mut TensorViewState,
) -> Response {
    let shape = tensor.shape().to_vec();
    let ndim = shape.len();
    state.clamp_to(&shape);
    let is_vector = ndim==1;

    let frame = egui::Frame::default()
        .inner_margin(4.0)
        .stroke(ui.visuals().window_stroke);

    frame.show(ui, |ui| {
        // Top bar: dtype, shape, copy buttons, format options.
        ui.horizontal(|ui| {
            ui.label(format!("{}  {:?}", tensor.dtype(), shape));
            if ui.button("Copy Full").clicked() {
                let text = format!("{:#}", tensor);
                ui.output_mut(|o| o.copied_text = text);
            }
            if ui.button("Copy Visible Slice").clicked() {
                let txt = slice_to_string(tensor, state);
                ui.output_mut(|o| o.copied_text = txt);
            }
            ui.separator();

            ui.label("fmt:");
            ui.checkbox(&mut state.monospace, "mono");
            ui.checkbox(&mut state.scientific, "sci");
            ui.add(
                egui::DragValue::new(&mut state.precision)
                    .clamp_range(0..=12)
                    .speed(0.1)
                    .prefix("prec "),
            );
        });

        // Axis selection & index pickers for fixed axes
        ui.horizontal_wrapped(|ui| {
            ui.label("row axis:");
            axis_combo(ui, Id::new("row_axis"), &mut state.row_axis, ndim);
            ui.label("col axis:");
            axis_combo(ui, Id::new("col_axis"), &mut state.col_axis, ndim);

            // Keep distinct if user picked same axis twice
            if state.row_axis == state.col_axis && ndim > 1 {
                state.col_axis = (state.row_axis + 1) % ndim;
            }

            ui.separator();
            // Fixed axes controls
            for ax in 0..ndim {
                if ax != state.row_axis && ax != state.col_axis {
                    let max = shape[ax].saturating_sub(1);
                    ui.add(
                        egui::DragValue::new(&mut state.fixed_idx[ax])
                            .clamp_range(0..=max as i64)
                            .prefix(format!("axis{ax}=")),
                    );
                }
            }
        });

        ui.horizontal(|ui| {
            // Paging / offsets
            let rmax = shape[state.row_axis].saturating_sub(1);
            let cmax = shape[state.col_axis].saturating_sub(1);
            if ui.button("⬆ prev rows").clicked() {
                if is_vector {
                    state.row_offset = 0;
                    state.rows_per_page = 1;
                } else {
                    state.row_offset = state.row_offset.saturating_sub(state.rows_per_page);
                }
            }
            if ui.button("⬇ next rows").clicked() {
                if is_vector {
                    state.row_offset = 0;
                    state.rows_per_page = 1;
                } else {
                    state.row_offset = (state.row_offset + state.rows_per_page).min(rmax);
                }
            }
            ui.add(
                egui::DragValue::new(&mut state.row_offset)
                    .clamp_range(0..=rmax as i64)
                    .prefix("row@"),
            );
            ui.add(
                egui::DragValue::new(&mut state.rows_per_page)
                    .clamp_range(1..=shape[state.row_axis] as i64)
                    .prefix("rows/page "),
            );

            ui.separator();

            if ui.button("⬅ prev cols").clicked() {
                state.col_offset = state.col_offset.saturating_sub(state.cols_per_page);
            }
            if ui.button("➡ next cols").clicked() {
                state.col_offset = (state.col_offset + state.cols_per_page).min(cmax);
            }
            ui.add(
                egui::DragValue::new(&mut state.col_offset)
                    .clamp_range(0..=cmax as i64)
                    .prefix("col@"),
            );
            ui.add(
                egui::DragValue::new(&mut state.cols_per_page)
                    .clamp_range(1..=shape[state.col_axis] as i64)
                    .prefix("cols/page "),
            );

            ui.separator();

            // Quick "fit" based on current available size and rough cell metrics
            if ui.button("Fit to view").clicked() {
                let avail = ui.available_size();
                if is_vector {
                    let approx_cols = ((avail.x - 80.0) / state.col_width).floor().max(1.0) as u64;
                    state.cols_per_page = approx_cols.min(shape[state.col_axis].max(1));
                    state.rows_per_page = 1; // vector
                } else {
                    let row_h = state.row_height.max(12.0);
                    let approx_cols = ((avail.x - 80.0) / state.col_width).floor().max(1.0) as u64;
                    let approx_rows = (avail.y / row_h).floor().max(1.0) as u64;
                    state.cols_per_page = approx_cols.min(shape[state.col_axis].max(1));
                    state.rows_per_page = approx_rows.min(shape[state.row_axis].max(1));
                }
            }
        });

        ui.separator();

        let is_vector = ndim==1;
        // Table: 1 index column + visible data columns
        let (rows_total, cols_total) = if is_vector {
            (1, shape[0])
        } else {
            (shape[state.row_axis], shape[state.col_axis])
        };

        let rows_vis = rows_total.saturating_sub(state.row_offset).min(state.rows_per_page);
        let cols_vis = cols_total.saturating_sub(state.col_offset).min(state.cols_per_page);

        let mut table = TableBuilder::new(ui)
            .striped(true)
            .vscroll(true)
            .resizable(true)
            .cell_layout(egui::Layout::left_to_right(Align::Center));

        // Row-index column
        table = table.column(Column::initial(72.0).resizable(true));

        // Data columns (paged)
        for _ in 0..cols_vis {
            table = table.column(Column::initial(state.col_width).resizable(true));
        }

        let font = if state.monospace {
            FontId::monospace(13.0)
        } else {
            FontId::proportional(13.0)
        };

        // Header with column indices
        table
            .header(22.0, |mut header| {
                let hdr = if is_vector {
                    "axis0".to_string()
                } else {
                    format!("axis{} \\ axis{}", state.row_axis, state.col_axis)
                };
                header.col(|ui| {ui.strong(hdr);});

                for j in 0..cols_vis {
                    let col_idx = state.col_offset + j;
                    header.col(|ui| {
                        ui.monospace(format!("{col_idx}"));
                    });
                }
            })
            .body(|mut body| {
                body.rows(state.row_height, rows_vis as usize, |mut row| {
                    let row_idx = row.index();
                    let r = state.row_offset + row_idx as u64;
                    row.col(|ui| {
                        ui.monospace(format!("{r}"));
                    });

                    for j in 0..cols_vis {
                        let c = state.col_offset + j;

                        let mut idx = vec![0u64; ndim];
                        if is_vector {
                            // Only one axis
                            idx[0] = c;
                        } else {
                            // Distinct axes, no overwrite
                            idx[state.row_axis] = r;
                            idx[state.col_axis] = c;
                            // Other axes come from state.fixed_idx (already clamped)
                            for ax in 0..ndim {
                                if ax != state.row_axis && ax != state.col_axis {
                                    idx[ax] = state.fixed_idx[ax];
                                }
                            }
                        }

                        let txt = match tensor.get(&idx) {
                            Some(v) => format_number(f64::cast_from_numeric_scalar(&v), state.precision, state.scientific),
                            None => String::from("—"),
                        };

                        row.col(|ui| {
                            let mut job = egui::text::LayoutJob::default();
                            job.append(
                                &txt,
                                0.0,
                                egui::TextFormat {
                                    font_id: font.clone(),
                                    color: ui.visuals().text_color(),
                                    ..Default::default()
                                },
                            );
                            ui.label(job);
                        });
                    }
                });
            });
    }).response
}

fn axis_combo(ui: &mut Ui, id: Id, axis: &mut usize, ndim: usize) {
    egui::ComboBox::from_id_source(id)
        .selected_text(format!("{axis}"))
        .show_ui(ui, |ui| {
            for a in 0..ndim {
                ui.selectable_value(axis, a, format!("{a}"));
            }
        });
}

fn format_number(v: f64, precision: usize, scientific: bool) -> String {
    if !v.is_finite() {
        if v.is_nan() { "NaN".to_string() }
        else if v.is_sign_negative() { "-∞".to_string() }
        else { "∞".to_string() }
    } else if scientific {
        format!("{:.*e}", precision, v)
    } else {
        format!("{:.*}", precision, v)
    }
}

/// Produce a textual dump of the currently visible 2D slice (for Copy).
fn slice_to_string(t: &NDArrayNumericTensor<DynRank>, s: &TensorViewState) -> String {
    let ndim = t.rank();
    let shape = t.shape();
    if ndim == 0 {
        return t.first_element().to_string();
    }


    let is_vector = ndim == 1;
    if is_vector {
        let total = shape[0];
        let cols = total.saturating_sub(s.col_offset).min(s.cols_per_page.max(1));
        let mut out = String::from("vector slice axis0\n[");
        for j in 0..cols {
            let c = s.col_offset + j;
            let val = t.get(&vec![c]).map(|x| f64::cast_from_numeric_scalar(&x)).unwrap_or(f64::NAN);
            if j > 0 { out.push_str(", "); }
            out.push_str(&format!("{}", val));
        }
        out.push(']');
        out
    } else {

        let rows_total = shape[s.row_axis];
        let cols_total = shape[s.col_axis];
        let rows_vis = rows_total.saturating_sub(s.row_offset).min(s.rows_per_page);
        let cols_vis = cols_total.saturating_sub(s.col_offset).min(s.cols_per_page);

        let mut out = String::new();
        out.push_str(&format!(
            "slice axes (row={}, col={}), offsets (row@{}, col@{}), size {}x{}\n",
            s.row_axis, s.col_axis, s.row_offset, s.col_offset, rows_vis, cols_vis
        ));

        for i in 0..rows_vis {
            let r = s.row_offset + i;
            out.push('[');
            for j in 0..cols_vis {
                let c = s.col_offset + j;
                let mut idx = s.fixed_idx.clone();
                if idx.len() != ndim { idx.resize(ndim, 0); }
                idx[s.row_axis] = r;
                idx[s.col_axis] = c;
                let val = t.get(&idx).map(|x| f64::cast_from_numeric_scalar(&x)).unwrap_or(f64::NAN);
                if j > 0 { out.push_str(", "); }
                out.push_str(&format!("{}", val));
            }
            out.push_str("]\n");
        }
        out
    }
}
