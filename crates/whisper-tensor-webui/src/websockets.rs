use std::collections::HashMap;
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::SendError;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::Closure;
use web_sys::{WebSocket, js_sys};
use whisper_tensor_server::{
    SuperGraphRequest, SuperGraphResponse, WebsocketClientServerMessage,
    WebsocketServerClientMessage,
};

pub struct ServerRequestManager {
    client_server_sender: mpsc::UnboundedSender<WebsocketClientServerMessage>,
    incoming_responses: HashMap<u64, SuperGraphResponse>,
    cancelled_responses: Vec<u64>,
    next_attention_token: u64,
}

impl ServerRequestManager {
    pub fn new(client_server_sender: mpsc::UnboundedSender<WebsocketClientServerMessage>) -> Self {
        Self {
            client_server_sender,
            incoming_responses: HashMap::new(),
            cancelled_responses: Vec::new(),
            next_attention_token: 0,
        }
    }

    pub fn new_response(&mut self, resp: SuperGraphResponse) {
        if let Some(attention_token) = resp.attention_token.clone() {
            if self.cancelled_responses.contains(&attention_token) {
                self.cancelled_responses.retain(|&x| x != attention_token);
            } else {
                self.incoming_responses.insert(attention_token, resp);
            }
        }
    }

    pub fn cancel_request(&mut self, attention_token: u64) {
        self.cancelled_responses.push(attention_token);
    }

    pub fn submit_supergraph_request(&mut self, mut req: SuperGraphRequest) -> u64 {
        if req.attention_token.is_none() {
            req.attention_token = Some(self.next_attention_token);
            self.next_attention_token += 1;
        };
        let ret = req.attention_token.unwrap();
        self.client_server_sender
            .send(WebsocketClientServerMessage::SuperGraphRequest(req))
            .unwrap();
        ret
    }

    pub fn get_response(&mut self, attention_token: u64) -> Option<SuperGraphResponse> {
        self.incoming_responses.remove(&attention_token)
    }

    pub fn send(
        &mut self,
        message: WebsocketClientServerMessage,
    ) -> Result<(), SendError<WebsocketClientServerMessage>> {
        self.client_server_sender.send(message)
    }
}

pub(crate) async fn websocket_task(
    server_client_sender: mpsc::UnboundedSender<WebsocketServerClientMessage>,
    mut client_server_receiver: mpsc::UnboundedReceiver<WebsocketClientServerMessage>,
) {
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
                        log::info!("Blob received {} bytes: {:?}", vec.len(), vec);
                        // here you can for example use the received image/png data

                        match ciborium::from_reader::<WebsocketServerClientMessage, _>(
                            vec.as_slice(),
                        ) {
                            Ok(msg) => {
                                log::debug!("Decoded message: {:?}", msg);
                                sender_clone.send(msg).unwrap();
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
        log::debug!("Sending message to server");
        ws_clone.send_with_u8_array(&data).unwrap();
    }
}
