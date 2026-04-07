use whisper_tensor::super_graph::{ModelEvalMode, SuperGraphEvalOptions};

/// Default inline constant threshold for LoweredEval mode.
const DEFAULT_INLINE_THRESHOLD: u64 = 1024;

/// Render an editor for `SuperGraphEvalOptions`.
///
/// Shows a combo box for selecting the model eval mode, and additional
/// controls when LoweredEval or CompiledEval is selected (inline constant
/// threshold).
pub(crate) fn eval_options_ui(ui: &mut egui::Ui, options: &mut SuperGraphEvalOptions) {
    ui.horizontal(|ui| {
        ui.label("Eval");

        let mode_label = match &options.model_eval_mode {
            ModelEvalMode::SymbolicEval => "Symbolic",
            ModelEvalMode::LoweredEval { .. } => "Lowered",
            ModelEvalMode::CompiledEval { .. } => "Compiled",
        };

        egui::ComboBox::from_id_salt(ui.id().with("eval_mode"))
            .selected_text(mode_label)
            .width(80.0)
            .show_ui(ui, |ui| {
                let is_symbolic = matches!(options.model_eval_mode, ModelEvalMode::SymbolicEval);
                if ui.selectable_label(is_symbolic, "Symbolic").clicked() {
                    options.model_eval_mode = ModelEvalMode::SymbolicEval;
                }
                let is_lowered =
                    matches!(options.model_eval_mode, ModelEvalMode::LoweredEval { .. });
                if ui.selectable_label(is_lowered, "Lowered").clicked() && !is_lowered {
                    options.model_eval_mode = ModelEvalMode::LoweredEval {
                        inline_constant_threshold: DEFAULT_INLINE_THRESHOLD,
                    };
                }
                let is_compiled =
                    matches!(options.model_eval_mode, ModelEvalMode::CompiledEval { .. });
                if ui.selectable_label(is_compiled, "Compiled").clicked() && !is_compiled {
                    options.model_eval_mode = ModelEvalMode::CompiledEval {
                        inline_constant_threshold: DEFAULT_INLINE_THRESHOLD,
                        compile_options: whisper_tensor::compiler::CompileOptions::default(),
                    };
                }
            });

        match &mut options.model_eval_mode {
            ModelEvalMode::LoweredEval {
                inline_constant_threshold,
            }
            | ModelEvalMode::CompiledEval {
                inline_constant_threshold,
                ..
            } => {
                ui.label("threshold");
                ui.add(
                    egui::DragValue::new(inline_constant_threshold)
                        .range(0..=1_000_000)
                        .speed(64),
                );
            }
            _ => {}
        }
    });
}
