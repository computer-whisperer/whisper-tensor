#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use std::collections::HashMap;
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::sync::Mutex as StdMutex;
    use whisper_tensor_server::ServerConfigReport;
    use whisper_tensor_server::handler::handle_client_session;
    use whisper_tensor_server::model_server::{ModelServer, default_loaders};
    use whisper_tensor_server::scheduler::scheduler;

    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let rt = tokio::runtime::Runtime::new().unwrap();

    let (server_client_sender, server_client_receiver) = tokio::sync::mpsc::unbounded_channel();
    let (client_server_sender, client_server_receiver) = tokio::sync::mpsc::unbounded_channel();

    // Start the in-process server on the tokio runtime
    let ctx_sender = std::sync::Arc::new(std::sync::Mutex::new(None::<egui::Context>));
    let ctx_for_handler = ctx_sender.clone();

    rt.spawn(async move {
        let loaders = default_loaders();
        let model_server = Arc::new(ModelServer::new(loaders));

        let (scheduler_tx, scheduler_rx) = tokio::sync::mpsc::channel(100);
        let cancellation_registry = Arc::new(StdMutex::new(HashSet::<u64>::new()));
        let observer_settings_registry = Arc::new(std::sync::Mutex::new(HashMap::new()));

        #[cfg(feature = "vulkan")]
        let vulkan_available = true;
        #[cfg(not(feature = "vulkan"))]
        let vulkan_available = false;

        let server_config_report = ServerConfigReport { vulkan_available };

        tokio::spawn(scheduler(
            scheduler_rx,
            model_server.clone(),
            cancellation_registry.clone(),
            observer_settings_registry.clone(),
        ));

        handle_client_session(
            client_server_receiver,
            server_client_sender,
            scheduler_tx,
            cancellation_registry,
            observer_settings_registry,
            model_server,
            server_config_report,
            move || {
                if let Some(ctx) = ctx_for_handler.lock().unwrap().as_ref() {
                    ctx.request_repaint();
                }
            },
        )
        .await;
    });

    let native_options = eframe::NativeOptions::default();

    let ctx_sender_for_app = ctx_sender.clone();
    eframe::run_native(
        "Whisper Tensor",
        native_options,
        Box::new(move |cc| {
            // Give the handler access to the egui context for repaint triggers
            *ctx_sender_for_app.lock().unwrap() = Some(cc.egui_ctx.clone());

            Ok(Box::new(whisper_tensor_ui::WebUIApp::new(
                cc,
                server_client_receiver,
                client_server_sender,
            )))
        }),
    )
    .unwrap();
}

#[cfg(target_arch = "wasm32")]
fn main() {
    // WASM entry point is in lib.rs via wasm_bindgen
}
