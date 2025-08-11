use egui::{Color32, CursorIcon, Event, EventFilter, Label, RichText, Sense, Ui, Vec2, Widget};
use log::info;
use whisper_tensor::tokenizer::{Tokenizer, TokenizerError};

fn escape_token_text(input: &str) -> String {
    use std::fmt::Write;

    let mut out = String::with_capacity(input.len());

    for c in input.chars() {
        match c {
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\\' => out.push_str("\\\\"),
            // `is_control` is true for all C0 controls (0x00–0x1F) and DEL (0x7F)
            c if c.is_control() => {
                // \u{XXXX} where XXXX is at least 4 hex digits
                write!(out, "\\u{{{:04X}}}", c as u32).unwrap();
            }
            // Printable, keep as‑is
            c => out.push(c),
        }
    }

    out
}

pub struct TokenizedRichText {
    logits_to_render: usize,
}

impl TokenizedRichText {
    pub fn new() -> Self {
        Self {
            logits_to_render: 4,
        }
    }

    pub fn ui<T: Tokenizer>(
        &self,
        ui: &mut Ui,
        tokenizer: &T,
        tokens: &mut Vec<u32>,
        logits: Option<&[&[(u32, f32)]]>,
    ) {
        let frame = egui::Frame::default()
            .inner_margin(2.0)
            .stroke(ui.visuals().window_stroke);
        frame.show(ui, |ui| {
            let id = egui::Id::new("llm box 2");

            // Display

            let color_wheel = [
                Color32::from_rgb(50, 0, 0),
                Color32::from_rgb(0, 50, 0),
                Color32::from_rgb(0, 0, 50),
            ];

            ui.horizontal_wrapped(|ui| {
                {
                    let spacing_mut = ui.spacing_mut();
                    spacing_mut.item_spacing.x = 0.0;
                    spacing_mut.item_spacing.y = 2.0;
                }
                let mut idx = 0;
                let mut is_good = true;
                let mut last_first_block_height = 0.0;
                loop {
                    let color = color_wheel[idx % color_wheel.len()];
                    if idx >= tokens.len() && !is_good {
                        break;
                    }

                    let chosen_token_id = tokens.get(idx);

                    ui.vertical(|ui| {
                        let mut next_is_good = is_good;
                        if let Some(chosen_token) = chosen_token_id {
                            let label_text = match tokenizer.decode(&[*chosen_token]) {
                                Ok(token_str) => escape_token_text(&token_str),
                                Err(_) => "(?)".to_string(),
                            };

                            // Token string
                            let text = RichText::new(label_text).background_color(color).size(16.0);
                            Label::new(text).ui(ui);

                            // Token id
                            let text = RichText::new(chosen_token.to_string())
                                .background_color(color)
                                .size(6.0);
                            Label::new(text).ui(ui);

                            last_first_block_height = ui.min_rect().height();
                        } else {
                            next_is_good = false;
                            ui.allocate_space(Vec2::new(5.0, last_first_block_height));
                        }
                        {
                            let spacing_mut = ui.spacing_mut();
                            spacing_mut.item_spacing.x = 5.0;
                            spacing_mut.item_spacing.y = 3.0;
                        }
                        if is_good {
                            if let Some(logits) = logits {
                                if idx - 1 < logits.len() {
                                    let logits = logits[idx - 1];
                                    for j in 0..self.logits_to_render {
                                        if j >= logits.len() {
                                            break;
                                        }
                                        let (token_id, value) = &logits[j];
                                        let token_str = tokenizer
                                            .decode(&[*token_id])
                                            .unwrap_or("?".to_string());
                                        let token_str = escape_token_text(&token_str);
                                        //ui.label(format!("{token_id}, {value}, {token_str}"));
                                        let text = RichText::new(token_str)
                                            .background_color(Color32::from_rgb(20, 20, 20))
                                            .size(11.0);
                                        ui.label(text);
                                        //let text = RichText::new(token_id.to_string()).size(8.0);
                                        //ui.label(text);
                                        let text = RichText::new(format!("{value:.02}")).size(8.0);
                                        ui.label(text);
                                    }
                                }
                            }
                        }
                        is_good = next_is_good;
                    });

                    idx += 1;
                }

                ui.allocate_exact_size(ui.available_size_before_wrap(), Sense::click());
            });

            let mut response = ui.interact(ui.min_rect(), id, Sense::click());

            if response.hovered() {
                ui.ctx().set_cursor_icon(CursorIcon::Text);
            }
            let event_filter = EventFilter {
                // moving the cursor is really important
                horizontal_arrows: true,
                vertical_arrows: true,
                tab: false, // tab is used to change focus, not to insert a tab character
                ..Default::default()
            };
            if ui.memory(|mem| mem.has_focus(id)) {
                ui.memory_mut(|mem| mem.set_focus_lock_filter(id, event_filter));

                let events = ui.input(|i| i.filtered_events(&event_filter));
                for event in events {
                    match event {
                        Event::Text(x) => {
                            if let Ok(mut text) = tokenizer.decode(tokens) {
                                text.push_str(&x);
                                let new_tokens = tokenizer.encode(&text);
                                *tokens = new_tokens;
                            }
                        }
                        Event::Key {
                            key: egui::Key::Backspace,
                            pressed: true,
                            ..
                        } => {
                            if let Ok(mut text) = tokenizer.decode(tokens) {
                                if !text.is_empty() {
                                    text.remove(text.len() - 1);
                                    let new_tokens = tokenizer.encode(&text);
                                    *tokens = new_tokens;
                                }
                            }
                        }
                        Event::PointerMoved(_) | Event::PointerGone | Event::Key { .. } => {
                            // Don't care
                        }
                        _ => {
                            info!("Unhandled event: {event:?}")
                        }
                    }
                }
            }
            if response.clicked() {
                ui.memory_mut(|mem| mem.request_focus(response.id));
            }
        });
    }
}
