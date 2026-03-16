use std::collections::{HashMap, HashSet};
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::SendError;
#[cfg(target_arch = "wasm32")]
use whisper_tensor_server::WebsocketServerClientMessage;
use whisper_tensor_server::{
    AbbreviatedTensorReportSettings, SuperGraphExecutionReport, SuperGraphObserverSettingsUpdate,
    SuperGraphRequest, SuperGraphResponse, WebsocketClientServerMessage,
};

#[derive(Clone, Debug, PartialEq)]
struct ObserverSettingsSnapshot {
    subscribed_tensors: Vec<Vec<whisper_tensor::graph::GlobalId>>,
    do_node_execution_reports: bool,
    abbreviated_tensor_report_settings: Option<AbbreviatedTensorReportSettings>,
}

impl ObserverSettingsSnapshot {
    fn new(
        mut subscribed_tensors: Vec<Vec<whisper_tensor::graph::GlobalId>>,
        do_node_execution_reports: bool,
        mut abbreviated_tensor_report_settings: Option<AbbreviatedTensorReportSettings>,
    ) -> Self {
        for path in &mut subscribed_tensors {
            path.shrink_to_fit();
        }
        subscribed_tensors.sort();
        subscribed_tensors.dedup();

        if let Some(settings) = &mut abbreviated_tensor_report_settings {
            settings.subscribed_tensors.sort();
            settings.subscribed_tensors.dedup();
        }

        Self {
            subscribed_tensors,
            do_node_execution_reports,
            abbreviated_tensor_report_settings,
        }
    }
}

pub struct ServerRequestManager {
    client_server_sender: mpsc::UnboundedSender<WebsocketClientServerMessage>,
    incoming_responses: HashMap<u64, SuperGraphResponse>,
    incoming_reports: HashMap<u64, Vec<SuperGraphExecutionReport>>,
    active_requests: HashSet<u64>,
    observer_settings_by_request: HashMap<u64, ObserverSettingsSnapshot>,
    next_attention_token: u64,
}

impl ServerRequestManager {
    pub fn new(client_server_sender: mpsc::UnboundedSender<WebsocketClientServerMessage>) -> Self {
        Self {
            client_server_sender,
            incoming_reports: HashMap::new(),
            incoming_responses: HashMap::new(),
            active_requests: HashSet::new(),
            observer_settings_by_request: HashMap::new(),
            next_attention_token: 0,
        }
    }

    pub fn new_response(&mut self, resp: SuperGraphResponse) {
        if let Some(attention_token) = resp.attention_token
            && self.active_requests.contains(&attention_token)
        {
            self.incoming_responses.insert(attention_token, resp);
        }
    }

    pub fn new_execution_report(&mut self, report: SuperGraphExecutionReport) {
        if let Some(attention_token) = report.attention
            && self.active_requests.contains(&attention_token)
        {
            self.incoming_reports
                .entry(attention_token)
                .or_default()
                .push(report);
        }
    }

    #[allow(dead_code)]
    pub fn cancel_request(&mut self, attention_token: u64) {
        self.incoming_reports.remove(&attention_token);
        self.incoming_responses.remove(&attention_token);
        self.active_requests.remove(&attention_token);
        self.observer_settings_by_request.remove(&attention_token);
        let _ =
            self.client_server_sender
                .send(WebsocketClientServerMessage::CancelSuperGraphRequest(
                    attention_token,
                ));
    }

    pub fn update_observer_settings(
        &mut self,
        attention_token: u64,
        subscribed_tensors: Vec<Vec<whisper_tensor::graph::GlobalId>>,
        do_node_execution_reports: bool,
        abbreviated_tensor_report_settings: Option<AbbreviatedTensorReportSettings>,
    ) {
        if !self.active_requests.contains(&attention_token) {
            return;
        }
        let new_settings = ObserverSettingsSnapshot::new(
            subscribed_tensors,
            do_node_execution_reports,
            abbreviated_tensor_report_settings,
        );
        if self.observer_settings_by_request.get(&attention_token) == Some(&new_settings) {
            return;
        }
        self.observer_settings_by_request
            .insert(attention_token, new_settings.clone());
        let _ = self.client_server_sender.send(
            WebsocketClientServerMessage::UpdateSuperGraphObserverSettings(
                SuperGraphObserverSettingsUpdate {
                    attention_token,
                    subscribed_tensors: new_settings.subscribed_tensors,
                    do_node_execution_reports: new_settings.do_node_execution_reports,
                    abbreviated_tensor_report_settings: new_settings
                        .abbreviated_tensor_report_settings,
                },
            ),
        );
    }

    pub fn submit_supergraph_request(&mut self, mut req: SuperGraphRequest) -> u64 {
        if req.attention_token.is_none() {
            req.attention_token = Some(self.next_attention_token);
            self.next_attention_token += 1;
        };
        let ret = req.attention_token.unwrap();
        self.active_requests.insert(ret);
        self.observer_settings_by_request.insert(
            ret,
            ObserverSettingsSnapshot::new(
                req.subscribed_tensors.clone(),
                req.do_node_execution_reports,
                req.abbreviated_tensor_report_settings.clone(),
            ),
        );
        self.client_server_sender
            .send(WebsocketClientServerMessage::SuperGraphRequest(req))
            .unwrap();
        ret
    }

    pub fn get_response(&mut self, attention_token: u64) -> Option<SuperGraphResponse> {
        if let Some(x) = self.incoming_responses.remove(&attention_token) {
            self.active_requests.remove(&attention_token);
            self.incoming_reports.remove(&attention_token);
            self.observer_settings_by_request.remove(&attention_token);
            Some(x)
        } else {
            None
        }
    }

    pub fn get_reports(&mut self, attention_token: u64) -> Option<Vec<SuperGraphExecutionReport>> {
        self.incoming_reports.remove(&attention_token)
    }

    #[allow(clippy::result_large_err)]
    pub fn send(
        &mut self,
        message: WebsocketClientServerMessage,
    ) -> Result<(), SendError<WebsocketClientServerMessage>> {
        self.client_server_sender.send(message)
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) async fn websocket_task(
    server_client_sender: mpsc::UnboundedSender<WebsocketServerClientMessage>,
    mut client_server_receiver: mpsc::UnboundedReceiver<WebsocketClientServerMessage>,
    context: egui::Context,
) {
    use std::time::Duration;
    use wasm_bindgen::JsCast;
    use wasm_bindgen::prelude::Closure;
    use web_sys::{WebSocket, js_sys};

    let ws = WebSocket::new("/ws").unwrap();

    // Set up event handlers
    let onopen_callback = Closure::wrap(Box::new(move || {
        log::debug!("WebSocket connection opened");
    }) as Box<dyn FnMut()>);
    ws.set_onopen(Some(onopen_callback.as_ref().unchecked_ref()));
    onopen_callback.forget();

    // Handle messages coming from the server
    let server_client_sender_clone = server_client_sender.clone();
    let onmessage_callback = Closure::wrap(Box::new(move |e: web_sys::MessageEvent| {
        let context = context.clone();
        match e.data().dyn_into::<web_sys::Blob>() {
            Ok(blob) => {
                let fr = web_sys::FileReader::new().unwrap();
                let fr_c = fr.clone();
                // create onLoadEnd callback
                let sender_clone = server_client_sender_clone.clone();
                let onloadend_cb =
                    Closure::<dyn FnMut(_)>::new(move |_e: web_sys::ProgressEvent| {
                        let array = js_sys::Uint8Array::new(&fr_c.result().unwrap());
                        let vec = array.to_vec();

                        match ciborium::from_reader::<WebsocketServerClientMessage, _>(
                            vec.as_slice(),
                        ) {
                            Ok(msg) => {
                                sender_clone.send(msg).unwrap();
                                context.request_repaint_after(Duration::from_millis(20));
                            }
                            Err(err) => {
                                log::warn!("Failed to decode message: {:?}", err);
                            }
                        }
                    });
                fr.set_onloadend(Some(onloadend_cb.as_ref().unchecked_ref()));
                fr.read_as_array_buffer(&blob).expect("blob not readable");
                onloadend_cb.forget();
            }
            Err(err) => {
                log::warn!("Failed to decode message: {:?}", err);
            }
        }
    }) as Box<dyn FnMut(web_sys::MessageEvent)>);
    ws.set_onmessage(Some(onmessage_callback.as_ref().unchecked_ref()));
    onmessage_callback.forget();

    // Handle errors
    let onerror_callback = Closure::wrap(Box::new(move |e: web_sys::ErrorEvent| {
        log::error!("WebSocket error: {}", e.message());
    }) as Box<dyn FnMut(web_sys::ErrorEvent)>);
    ws.set_onerror(Some(onerror_callback.as_ref().unchecked_ref()));
    onerror_callback.forget();

    // Handle closing
    let onclose_callback = Closure::wrap(Box::new(move |e: web_sys::CloseEvent| {
        log::debug!("WebSocket closed: {} - {}", e.code(), e.reason());
    }) as Box<dyn FnMut(web_sys::CloseEvent)>);
    ws.set_onclose(Some(onclose_callback.as_ref().unchecked_ref()));
    onclose_callback.forget();

    // Process messages from the client to send to the server
    let ws_clone = ws.clone();
    while let Some(message) = client_server_receiver.recv().await {
        let mut data = Vec::<u8>::new();
        ciborium::into_writer(&message, &mut data).unwrap();
        ws_clone.send_with_u8_array(&data).unwrap();
    }
}
