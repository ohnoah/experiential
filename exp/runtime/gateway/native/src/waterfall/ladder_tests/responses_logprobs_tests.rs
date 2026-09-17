//! Responses probability commitment and retention regressions.
use super::*;

const DELTA: &str = r#"{"type":"response.output_text.delta","output_index":0,"item_id":"msg-a","content_index":0,"delta":"","logprobs":[{"token":"OK","logprob":-0.125,"bytes":[79,75]}]}"#;
const TERMINAL: &str = r#"{"type":"response.completed","response":{"status":"completed","output":[{"id":"msg-a","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"OK","logprobs":[{"token":"OK","logprob":-0.125,"bytes":[79,75]}]}]}],"usage":{"input_tokens":1,"output_tokens":1}}}"#;

fn responses_probability_wire(id: &str, url: &str) -> DeploymentWire {
    let mut route = responses_wire(id, url, &[]);
    route.upstream_payload["include"] = json!(["message.output_text.logprobs"]);
    route.upstream_payload["top_logprobs"] = json!(0);
    route
}

#[test]
fn responses_probability_commits_without_ttft_and_prevents_late_fallback() {
    block_on(async {
        let harness = Harness::new();
        let first = spawn_rung(vec![Answer::Stream(&[DELTA, FAILED])]).await;
        let second = spawn_rung(vec![Answer::Stream(&[DELTA, TERMINAL])]).await;
        let route = [
            responses_probability_wire("a", &first.url),
            responses_probability_wire("b", &second.url),
        ];
        let mut guard = AttemptGuard::new(
            harness.bridge.clone(),
            Arc::new(AtomicUsize::new(0)),
            "responses-probability-commit".to_string(),
            Instant::now(),
        );
        let context = WaterfallContext {
            bridge: &harness.bridge,
            http: &harness.http,
            request_id: "responses-probability-commit",
            raw_key: "key",
            caller_scope: None,
            route: &route,
            policy: RoutePolicy {
                maximum_total_attempts: 2,
                maximum_same_deployment_attempts: 1,
                refusal_failover: true,
                throttle_redial: None,
            },
            deadline: Instant::now() + Duration::from_secs(10),
            time_to_first_byte: Duration::from_secs(2),
            time_to_first_byte_slope_seconds_per_million_input_tokens: 0.0,
            approximate_input_tokens: 1.0,
            chat_logprobs: false,
            output_less_retention: None,
            output_token_cap: None,
        };
        let Won::Committed(committed) = acquire_attempt(&context, &mut guard).await else {
            panic!("Responses probabilities must commit");
        };
        assert_eq!(committed.depth, 0);
        assert!(committed.relay.first_token_at().is_none());
        assert!(committed.prefix.iter().any(|event| matches!(
            event,
            Event::ProviderResponsesLogprobs { records, .. } if records.as_array().is_some_and(|items| !items.is_empty())
        )));
        let terminal = committed
            .relay
            .next_event(
                Instant::now() + Duration::from_secs(2),
                Duration::from_secs(2),
                Instant::now(),
            )
            .await
            .expect("failed Responses terminal")
            .expect("terminal event");
        assert!(matches!(terminal, Event::Failed(_)));
        assert!(second.accepted.lock().unwrap().is_empty());
    });
}

#[test]
fn responses_probability_records_count_toward_retained_bytes() {
    let records = json!([{"token":"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx","logprob":-0.1,"bytes":[120],"top_logprobs":[]}]);
    let event = Event::ProviderResponsesLogprobs {
        output_index: 0,
        item_id: "msg".into(),
        content_index: 0,
        phase: "delta".into(),
        records,
    };
    assert!(crate::relay::event_retained_bytes(&event) > 64);
}

const EMPTY_DELTA: &str = r#"{"type":"response.output_text.delta","output_index":0,"item_id":"msg-empty","content_index":0,"delta":"","logprobs":[]}"#;
const FAILED: &str =
    r#"{"type":"error","code":"rate_limit_exceeded","message":"try another rung"}"#;
const SECOND_TERMINAL: &str = r#"{"type":"response.completed","response":{"status":"completed","output":[{"id":"msg-b","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"B","logprobs":[{"token":"B","logprob":-0.25,"bytes":[66]}]}]}],"usage":{"input_tokens":1,"output_tokens":1}}}"#;
const SECOND_DELTA: &str = r#"{"type":"response.output_text.delta","output_index":0,"item_id":"msg-b","content_index":0,"delta":"","logprobs":[{"token":"B","logprob":-0.25,"bytes":[66]}]}"#;

#[test]
fn responses_empty_probability_scaffolding_does_not_escape_failed_attempt() {
    block_on(async {
        let harness = Harness::new();
        let first = spawn_rung(vec![Answer::Stream(&[EMPTY_DELTA, FAILED])]).await;
        let second = spawn_rung(vec![Answer::Stream(&[SECOND_DELTA, SECOND_TERMINAL])]).await;
        let route = [
            responses_probability_wire("a", &first.url),
            responses_probability_wire("b", &second.url),
        ];
        let mut guard = AttemptGuard::new(
            harness.bridge.clone(),
            Arc::new(AtomicUsize::new(0)),
            "responses-empty-retry".into(),
            Instant::now(),
        );
        let context = WaterfallContext {
            bridge: &harness.bridge,
            http: &harness.http,
            request_id: "responses-empty-retry",
            raw_key: "key",
            caller_scope: None,
            route: &route,
            policy: RoutePolicy {
                maximum_total_attempts: 2,
                maximum_same_deployment_attempts: 1,
                refusal_failover: true,
                throttle_redial: None,
            },
            deadline: Instant::now() + Duration::from_secs(10),
            time_to_first_byte: Duration::from_secs(2),
            time_to_first_byte_slope_seconds_per_million_input_tokens: 0.0,
            approximate_input_tokens: 1.0,
            chat_logprobs: false,
            output_less_retention: None,
            output_token_cap: None,
        };
        let Won::Committed(committed) = acquire_attempt(&context, &mut guard).await else {
            panic!("fallback must commit");
        };
        assert_eq!(committed.depth, 1);
        assert_eq!(
            committed
                .prefix
                .iter()
                .filter_map(|event| match event {
                    Event::ProviderResponsesLogprobs {
                        item_id, records, ..
                    } => Some((item_id, records)),
                    _ => None,
                })
                .map(|(item_id, records)| (item_id.as_str(), records.clone()))
                .collect::<Vec<_>>(),
            vec![("msg-b", json!([{"token":"B","logprob":-0.25,"bytes":[66]}]))],
        );
        assert!(!committed.prefix.iter().any(|event| matches!(event, Event::ProviderOutputItemStarted { item_id: Some(item_id), .. } if item_id == "msg-empty")));
    });
}
