use crate::events::Event;

pub(super) fn is_semantic(event: &Event) -> bool {
    matches!(
        event,
        Event::TextDelta(_)
            | Event::RefusalDelta(_)
            | Event::ProviderTextDelta { .. }
            | Event::ProviderRefusalDelta { .. }
            | Event::ChoiceLogprobsDelta { .. }
            | Event::ProviderOutputItemCompleted { .. }
            | Event::ReasoningSummaryDelta { .. }
            | Event::ThinkingDelta { .. }
            | Event::ThinkingSignature { .. }
            | Event::RedactedThinking { .. }
            | Event::EncryptedReasoning { .. }
            | Event::ReasoningContentDelta { .. }
            | Event::ToolCallStarted { .. }
            | Event::ToolArgumentsDelta { .. }
            | Event::ToolCallCompleted { .. }
            | Event::TextBlockStarted { .. }
            | Event::CitationDelta { .. }
            | Event::ServerToolUseStarted { .. }
            | Event::ServerToolArgumentsDelta { .. }
            | Event::ServerToolUseCompleted { .. }
            | Event::ServerToolResult { .. }
            | Event::HostedToolItemStarted { .. }
            | Event::HostedToolItemProgress { .. }
            | Event::HostedToolItemCompleted { .. }
            | Event::ProviderTextAnnotation { .. }
    ) || matches!(
        event,
        Event::ProviderResponsesLogprobs { records, .. }
            if records.as_array().is_some_and(|items| !items.is_empty())
    )
}
