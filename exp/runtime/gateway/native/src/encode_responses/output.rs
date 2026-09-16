//! Final public envelope and SSE framing for the Responses encoder.

use serde_json::{json, Value};

use super::{aggregate, OutputSlot, ResponsesSseEncoder};
use crate::encode::compact_json;
use crate::errors::Failure;
use crate::events::ProviderOutputItemStatus;

impl ResponsesSseEncoder {
    /// Build one SDK-readable Responses envelope for the current lifecycle state.
    pub(super) fn response(&self, status: &str, failure: Option<&Failure>) -> Value {
        let include_content = status != "in_progress";
        let fallback_status = match status {
            "in_progress" => ProviderOutputItemStatus::InProgress,
            "completed" => ProviderOutputItemStatus::Completed,
            _ => ProviderOutputItemStatus::Incomplete,
        };
        let output: Vec<Value> = self
            .output_order
            .iter()
            .map(|slot| match slot {
                OutputSlot::Message(key) => {
                    self.messages[key].item(include_content, fallback_status)
                }
                OutputSlot::Tool(index) => self.tools[index].item(fallback_status),
                // Hosted tool items re-serve the provider's verbatim JSON.
                OutputSlot::HostedTool(index) => self.hosted[index].item.clone(),
                OutputSlot::Reasoning(index) => self.reasoning[index].item(
                    include_content,
                    fallback_status,
                    self.envelope.include_encrypted_reasoning,
                ),
                OutputSlot::FireworksReasoning => self
                    .fireworks_reasoning
                    .as_ref()
                    .expect("Fireworks output slot has state")
                    .item(
                        include_content,
                        fallback_status,
                        self.envelope.include_encrypted_reasoning,
                    ),
            })
            .collect();
        let error = if status == "failed" {
            json!({
                "code": "server_error",
                "message": failure
                    .map(|failure| failure.safe_message.clone())
                    .unwrap_or_else(|| "Gateway stream failed.".to_string()),
            })
        } else {
            Value::Null
        };
        let mut response = json!({
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "completed_at": if status == "completed" { json!(self.created_at) } else { Value::Null },
            "status": status,
            "error": error,
            "incomplete_details": if status == "incomplete" {
                json!({"reason": "max_output_tokens"})
            } else {
                Value::Null
            },
            "instructions": Value::Null,
            "metadata": self.envelope.metadata,
            "model": self.model,
            "output": output,
            "parallel_tool_calls": self.envelope.parallel_tool_calls,
            "temperature": self.envelope.temperature,
            "top_p": self.envelope.top_p,
            "reasoning": self.envelope.reasoning,
            "tool_choice": self.envelope.tool_choice,
            "tools": self.envelope.tools,
            "max_output_tokens": self.envelope.max_output_tokens,
            "previous_response_id": self.envelope.previous_response_id,
            "usage": if include_content {
                aggregate::responses_usage(self.usage.as_ref())
            } else {
                Value::Null
            },
        });
        if !self.envelope.ignored_parameters.is_empty() {
            response
                .as_object_mut()
                .expect("response envelope is an object")
                .insert(
                    "x-experiential-ignored-parameters".to_string(),
                    json!(self.envelope.ignored_parameters),
                );
        }
        response
    }

    /// Assign one monotonic sequence number and frame a named SSE event.
    pub(super) fn event(&mut self, event_type: &str, fields: Value) -> String {
        let mut payload = serde_json::Map::new();
        payload.insert("type".to_string(), Value::String(event_type.to_string()));
        payload.insert("sequence_number".to_string(), json!(self.sequence));
        if let Value::Object(entries) = fields {
            for (key, value) in entries {
                payload.insert(key, value);
            }
        }
        self.sequence += 1;
        let encoded = compact_json(&Value::Object(payload));
        format!("event: {event_type}\ndata: {encoded}\n\n")
    }

    fn close_message(
        &mut self,
        key: MessageKey,
        fallback_status: ProviderOutputItemStatus,
    ) -> Vec<String> {
        let (
            item_id,
            output_index,
            text,
            refusal,
            annotations,
            text_started,
            refusal_started,
            text_done_logprobs,
            content_part_done_logprobs,
            item,
        ) = {
            let state = match self.messages.get_mut(&key) {
                Some(state) => state,
                None => return Vec::new(),
            };
            if state.done {
                return Vec::new();
            }
            state.done = true;
            if matches!(
                state.status,
                None | Some(ProviderOutputItemStatus::InProgress)
            ) {
                state.status = Some(fallback_status);
            }
            (
                state.item_id.clone(),
                state.output_index,
                state.text.clone(),
                state.refusal.clone(),
                state.annotations.clone(),
                state.text_started,
                state.refusal_started,
                state
                    .logprobs
                    .get(&0)
                    .and_then(|phases| phases.get("text_done"))
                    .cloned(),
                state
                    .logprobs
                    .get(&0)
                    .and_then(|phases| phases.get("content_part_done"))
                    .cloned(),
                state.item(true, fallback_status),
            )
        };
        let mut frames: Vec<String> = Vec::new();
        let mut content_index = 0;
        if text_started {
            frames.push(self.event(
                "response.output_text.done",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "text": text,
                    "logprobs": text_done_logprobs.unwrap_or_else(|| json!([])),
                }),
            ));
            let mut part = json!({"type": "output_text", "text": text, "annotations": annotations});
            if let Some(records) = content_part_done_logprobs {
                part["logprobs"] = records;
            }
            frames.push(self.event(
                "response.content_part.done",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "part": part,
                }),
            ));
            content_index += 1;
        }
        if refusal_started {
            frames.push(self.event(
                "response.refusal.done",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "refusal": refusal,
                }),
            ));
            let part = json!({"type": "refusal", "refusal": refusal});
            frames.push(self.event(
                "response.content_part.done",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "part": part,
                }),
            ));
        }
        frames.push(self.event(
            "response.output_item.done",
            json!({
                "output_index": output_index,
                "item": item,
            }),
        ));
        frames
    }
}
