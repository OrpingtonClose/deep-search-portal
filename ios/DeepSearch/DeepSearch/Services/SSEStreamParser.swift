import Foundation

/// Parses Server-Sent Events (SSE) from a streaming HTTP response.
///
/// The Deep Search Portal proxies stream responses in OpenAI-compatible
/// SSE format: each line starts with "data: " followed by a JSON chunk,
/// and the stream terminates with "data: [DONE]".
final class SSEStreamParser {
    private var buffer = ""

    /// Parse incoming data bytes and yield complete SSE events.
    func parse(_ data: Data) -> [SSEEvent] {
        guard let text = String(data: data, encoding: .utf8) else { return [] }
        buffer += text

        var events: [SSEEvent] = []
        let lines = buffer.components(separatedBy: "\n")

        // Keep the last incomplete line in the buffer
        buffer = lines.last ?? ""

        for line in lines.dropLast() {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }

            if trimmed.hasPrefix("data: ") {
                let payload = String(trimmed.dropFirst(6))
                if payload == "[DONE]" {
                    events.append(.done)
                } else if let data = payload.data(using: .utf8) {
                    do {
                        let chunk = try JSONDecoder().decode(ChatCompletionChunk.self, from: data)
                        events.append(.chunk(chunk))
                    } catch {
                        // Skip malformed chunks
                    }
                }
            }
        }

        return events
    }

    /// Reset the parser state for a new stream.
    func reset() {
        buffer = ""
    }
}

/// An event parsed from the SSE stream.
enum SSEEvent {
    case chunk(ChatCompletionChunk)
    case done
}
