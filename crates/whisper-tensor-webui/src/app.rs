use strum::IntoEnumIterator;
use web_sys::WebSocket;
use tokio::sync::mpsc;
use futures::SinkExt;

enum WebsocketClientServerMessage {
    Ping
}

enum WebsocketServerClientMessage {
    Pong
}

async fn websocket_task(server_client_sender: mpsc::Sender<WebsocketServerClientMessage>, client_server_receiver: mpsc::UnboundedReceiver<WebsocketClientServerMessage>) {
    let ws = WebSocket::new("/ws").unwrap();
    
}

/// We derive Deserialize/Serialize so we can persist app state on shutdown.
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)] // if we add new fields, give them default values when deserializing old state
pub struct TemplateApp {
    is_load_model_dialog_open: bool,
    model_to_load_path_text: String,
    model_type_hint_selected: Option<onnx_import::ModelTypeHint>,
    websocket_server_client_receiver: mpsc::Receiver<WebsocketServerClientMessage>,
    websocket_client_server_message: mpsc::UnboundedSender<WebsocketClientServerMessage>
}

impl TemplateApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>, 
               websocket_server_client_receiver: mpsc::Receiver<WebsocketServerClientMessage>,
               websocket_client_server_message: mpsc::UnboundedSender<WebsocketClientServerMessage>) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.

        // Load previous app state (if any).
        // Note that you must enable the `persistence` feature for this to work.
        /*
        if let Some(storage) = cc.storage {
            return eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default();
        }*/

        Self {
            is_load_model_dialog_open: false,
            model_to_load_path_text: String::new(),
            model_type_hint_selected: None,
            websocket_server_client_receiver,
            websocket_client_server_message
        }
    }
}

impl eframe::App for TemplateApp {
    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Put your widgets into a `SidePanel`, `TopBottomPanel`, `CentralPanel`, `Window` or `Area`.
        // For inspiration and more examples, go to https://emilk.github.io/egui

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            // The top panel is often a good place for a menu bar:
            egui::menu::bar(ui, |ui| {
                egui::widgets::global_theme_preference_buttons(ui);
            });
        });

        if self.is_load_model_dialog_open {
            egui::Modal::new(egui::Id::new("Load Model")).show(ctx, |ui| {
                ui.text_edit_singleline(&mut self.model_to_load_path_text);
                egui::ComboBox::from_id_salt(1245).selected_text(if let Some(which) = &self.model_type_hint_selected {which.to_string()} else {String::from("No Hint")})
                    .show_ui(ui, |ui|{
                        ui.selectable_value(&mut self.model_type_hint_selected, None, "No Hint");
                        for which in onnx_import::ModelTypeHint::iter() {
                            ui.selectable_value(&mut self.model_type_hint_selected, Some(which.clone()), which.to_string());
                        }
                    });
                ui.button("Load");
                if ui.button("Cancel").clicked() {
                    self.is_load_model_dialog_open = false;
                }
            });
        }


        egui::CentralPanel::default().show(ctx, |ui| {
            // The central panel the region left after adding TopPanel's and SidePanel's
            ui.heading("Whisper Tensor");
            if ui.button("Load model").clicked() {
                self.is_load_model_dialog_open = true;
            };

            if ui.button("Ping").clicked() {
                self.websocket_client_server_message.send(WebsocketClientServerMessage::Ping);
            };
        });
    }
}
