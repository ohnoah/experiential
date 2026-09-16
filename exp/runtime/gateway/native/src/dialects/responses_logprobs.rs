//! OpenAI Responses probability observations.

use serde_json::{Map, Value};

use crate::errors::{Failure, FailureClass};
use crate::events::Event;

const MAX_TOKEN_CHARS: usize = 256;
const MAX_BYTES: usize = 4096;
const MAX_RECORDS_BYTES: usize = 1_048_576;

/// Check the bounded JSON shape accepted for one provider probability phase.
pub(crate) fn records_are_bounded(records: &Value) -> bool {
    let Some(records) = records.as_array() else {
        return false;
    };
    json_size(records, MAX_RECORDS_BYTES).is_some() && records.iter().all(valid_record)
}

/// Count JSON bytes without allocating a serialized copy.
pub(crate) fn json_size(value: &Value, limit: usize) -> Option<usize> {
    fn add(total: &mut usize, amount: usize, limit: usize) -> Option<()> {
        *total = total.checked_add(amount)?;
        (*total <= limit).then_some(())
    }
    fn walk(value: &Value, limit: usize) -> Option<usize> {
        let mut total = 0;
        match value {
            Value::Null => add(&mut total, 4, limit)?,
            Value::Bool(value) => add(&mut total, if *value { 4 } else { 5 }, limit)?,
            Value::Number(value) => add(&mut total, value.to_string().len(), limit)?,
            Value::String(value) => add(&mut total, value.len() + 2, limit)?,
            Value::Array(values) => {
                add(&mut total, 2, limit)?;
                for value in values {
                    total = total.checked_add(walk(value, limit - total)?)?;
                    if total > limit {
                        return None;
                    }
                }
            }
            Value::Object(values) => {
                add(&mut total, 2, limit)?;
                for (key, value) in values {
                    total = total.checked_add(key.len() + 3)?;
                    total = total.checked_add(walk(value, limit - total)?)?;
                    if total > limit {
                        return None;
                    }
                }
            }
        }
        Some(total)
    }
    walk(value, limit)
}

fn valid_record(record: &Value) -> bool {
    let Some(record) = record.as_object() else {
        return false;
    };
    let token_ok = record.get("token").is_some_and(|token| {
        token
            .as_str()
            .is_some_and(|token| token.chars().count() <= MAX_TOKEN_CHARS)
    });
    let logprob_ok = record
        .get("logprob")
        .is_some_and(|logprob| logprob.as_f64().is_some_and(f64::is_finite));
    let bytes_ok = record.get("bytes").is_none_or(|bytes| {
        bytes.as_array().is_some_and(|bytes| {
            bytes.len() <= MAX_BYTES
                && bytes
                    .iter()
                    .all(|byte| byte.as_u64().is_some_and(|byte| byte <= u8::MAX as u64))
        })
    });
    let alternatives_ok = record.get("top_logprobs").is_none_or(|alternatives| {
        alternatives.as_array().is_some_and(|alternatives| {
            alternatives.len() <= 20 && alternatives.iter().all(valid_record)
        })
    });
    token_ok && logprob_ok && bytes_ok && alternatives_ok
}

fn validate(records: &Value) -> Result<(), Failure> {
    records_are_bounded(records).then_some(()).ok_or_else(|| {
        Failure::new(
            FailureClass::MalformedResponse,
            "Responses probability records exceeded the supported shape",
        )
    })
}

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
    let mut events = Vec::new();
    for (content_index, part) in content.iter().enumerate() {
        let Some(records) = part.as_object().and_then(|part| part.get("logprobs")) else {
            continue;
        };
        validate(records)?;
        events.push(Event::ProviderResponsesLogprobs {
            output_index,
            item_id: item_id.to_string(),
            content_index: content_index as u32,
            phase: "item_done".to_string(),
            records: records.clone(),
        });
    }
    Ok(events)
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
            validate(records)?;
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
