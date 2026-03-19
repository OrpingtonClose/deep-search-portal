import Foundation

/// Parses streaming content to separate `<think>` reasoning blocks from final answer content.
///
/// The Deep Search Portal proxies wrap LLM reasoning in `<think>...</think>` tags:
/// - **Thinking Proxy**: Converts `<THINKING>/<ANSWER>` tags to `<think>` format
/// - **MiroFlow**: Wraps the entire multi-turn research process in `<think>` tags
/// - **Persistent Research**: Wraps subagent research and synthesis in `<think>` tags
///
/// This parser processes the stream token-by-token and routes content to either
/// the thinking callback or the answer callback.
final class ThinkingParser {
    enum Phase {
        case preThink     // Haven't seen <think> yet
        case thinking     // Inside <think> block
        case answer       // After </think>, receiving final answer
    }

    private(set) var phase: Phase = .preThink
    private var buffer = ""
    private(set) var thinkingContent = ""
    private(set) var answerContent = ""

    /// Process an incoming content token from the SSE stream.
    /// Returns a tuple of (thinkingDelta, answerDelta) indicating what new content
    /// was added to each section.
    func process(_ token: String) -> (thinkingDelta: String, answerDelta: String) {
        buffer += token
        var thinkingDelta = ""
        var answerDelta = ""

        switch phase {
        case .preThink:
            if let range = buffer.range(of: "<think>", options: .caseInsensitive) {
                // Everything before <think> is discarded (usually empty)
                let afterTag = String(buffer[range.upperBound...])
                buffer = ""
                phase = .thinking

                // Check if there's already a </think> in the remaining content
                if let closeRange = afterTag.range(of: "</think>", options: .caseInsensitive) {
                    let thinking = String(afterTag[..<closeRange.lowerBound])
                    let answer = String(afterTag[closeRange.upperBound...])
                    thinkingContent += thinking
                    thinkingDelta = thinking
                    answerContent += answer
                    answerDelta = answer
                    phase = .answer
                } else {
                    thinkingContent += afterTag
                    thinkingDelta = afterTag
                }
            } else if buffer.count > 200 && !buffer.lowercased().contains("<think") {
                // Model didn't use think tags - treat everything as answer
                answerContent += buffer
                answerDelta = buffer
                buffer = ""
                phase = .answer
            }

        case .thinking:
            if let range = buffer.range(of: "</think>", options: .caseInsensitive) {
                let thinking = String(buffer[..<range.lowerBound])
                let answer = String(buffer[range.upperBound...])
                buffer = ""
                thinkingContent += thinking
                thinkingDelta = thinking
                answerContent += answer
                answerDelta = answer
                phase = .answer
            } else if buffer.count > 20 {
                // Emit all but the last 20 chars (in case </think> spans chunks)
                let emit = String(buffer.dropLast(20))
                buffer = String(buffer.suffix(20))
                thinkingContent += emit
                thinkingDelta = emit
            }

        case .answer:
            answerContent += buffer
            answerDelta = buffer
            buffer = ""
        }

        return (thinkingDelta, answerDelta)
    }

    /// Flush any remaining buffered content.
    func flush() -> (thinkingDelta: String, answerDelta: String) {
        var thinkingDelta = ""
        var answerDelta = ""

        if !buffer.isEmpty {
            switch phase {
            case .preThink, .answer:
                answerContent += buffer
                answerDelta = buffer
            case .thinking:
                thinkingContent += buffer
                thinkingDelta = buffer
                phase = .answer
            }
            buffer = ""
        }

        return (thinkingDelta, answerDelta)
    }

    /// Reset the parser for a new message.
    func reset() {
        phase = .preThink
        buffer = ""
        thinkingContent = ""
        answerContent = ""
    }
}
