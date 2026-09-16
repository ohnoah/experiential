//! OpenAI Responses probability observations.

use serde_json::{Map, Value};

use crate::errors::Failure;
use crate::events::Event;

/// Extract probability records from a Responses output text event.
pub(crate) fn payload_records(payload: &Map<String, Value>) -> Option<Value> {
    payload
        .get("logprobs")
        .or_else(|| {
            payload
                .get("part")
                .and_then(Value::as_object)?
                .get("logprobs")
        })
        .cloned()
}

/// Extract item completion probability observations, retaining each content
/// part identity and the provider's exact record values.
pub(crate) fn item_done_events(
    output_index: u32,
    item: &Map<String, Value>,
) -> Result<Vec<Event>, Failure> {
    let Some(item_id) = item.get("id").and_then(Value::as_str) else {
        return Ok(Vec::new());
    };
    let Some(content) = item.get("content").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    Ok(content
        .iter()
        .enumerate()
        .filter_map(|(content_index, part)| {
            let records = part.as_object()?.get("logprobs")?.clone();
            Some(Event::ProviderResponsesLogprobs {
                output_index,
                item_id: item_id.to_string(),
                content_index: content_index as u32,
                phase: "item_done".to_string(),
                records,
            })
        })
        .collect())
}

/// Extract terminal probability observations from the final response object.
pub(crate) fn terminal_events(response: &Map<String, Value>) -> Result<Vec<Event>, Failure> {
    let mut events = Vec::new();
    let Some(output) = response.get("output").and_then(Value::as_array) else {
        return Ok(events);
    };
    for (output_position, item) in output.iter().enumerate() {
        let Some(item_object) = item.as_object() else {
            continue;
        };
        if item_object.get("type").and_then(Value::as_str) != Some("message") {
            continue;
        }
        let Some(item_id) = item_object.get("id").and_then(Value::as_str) else {
            continue;
        };
        let output_index = item_object
            .get("output_index")
            .and_then(Value::as_u64)
            .unwrap_or(output_position as u64);
        let Some(content) = item_object.get("content").and_then(Value::as_array) else {
            continue;
        };
        for (content_index, part) in content.iter().enumerate() {
            let Some(records) = part.as_object().and_then(|part| part.get("logprobs")) else {
                continue;
            };
            events.push(Event::ProviderResponsesLogprobs {
                output_index: output_index as u32,
                item_id: item_id.to_string(),
                content_index: content_index as u32,
                phase: "terminal".to_string(),
                records: records.clone(),
            });
        }
    }
    Ok(events)
}
