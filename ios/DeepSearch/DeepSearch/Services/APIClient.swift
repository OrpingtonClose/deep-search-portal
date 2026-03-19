import Foundation

/// Async client for the Deep Search Portal's OpenAI-compatible API.
///
/// Supports both regular and streaming chat completions, model listing,
/// and health checks. All proxies (Thinking, Deep Research, Persistent
/// Research) share the same OpenAI-compatible protocol.
final class APIClient: @unchecked Sendable {
    private let session: URLSession

    init() {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 300 // LLM calls can take minutes
        config.timeoutIntervalForResource = 600
        self.session = URLSession(configuration: config)
    }

    // MARK: - Streaming Chat Completions

    /// Send a streaming chat completion request and yield content tokens as they arrive.
    ///
    /// The stream includes `<think>` tags for reasoning content (from both the
    /// Thinking Proxy and MiroFlow Deep Research). The caller is responsible for
    /// parsing these tags to separate thinking from answer content.
    func streamChatCompletion(
        baseURL: String,
        apiKey: String,
        messages: [ChatMessage],
        model: String,
        temperature: Double = 0.3
    ) -> AsyncThrowingStream<StreamToken, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    guard let url = URL(string: "\(baseURL)/v1/chat/completions") else {
                        continuation.finish(throwing: APIError.invalidURL("\(baseURL)/v1/chat/completions"))
                        return
                    }
                    var request = URLRequest(url: url)
                    request.httpMethod = "POST"
                    request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
                    request.setValue("application/json", forHTTPHeaderField: "Content-Type")

                    let body = ChatCompletionRequest(
                        model: model,
                        messages: messages,
                        stream: true,
                        temperature: temperature
                    )
                    request.httpBody = try JSONEncoder().encode(body)

                    let (bytes, response) = try await self.session.bytes(for: request)

                    guard let httpResponse = response as? HTTPURLResponse else {
                        continuation.finish(throwing: APIError.invalidResponse)
                        return
                    }

                    guard httpResponse.statusCode == 200 else {
                        var errorBody = ""
                        for try await line in bytes.lines {
                            errorBody += line
                            if errorBody.count > 1000 { break }
                        }
                        continuation.finish(throwing: APIError.httpError(
                            statusCode: httpResponse.statusCode,
                            body: errorBody
                        ))
                        return
                    }

                    let parser = SSEStreamParser()

                    for try await line in bytes.lines {
                        if Task.isCancelled { break }
                        let lineData = Data((line + "\n").utf8)
                        let events = parser.parse(lineData)

                        for event in events {
                            switch event {
                            case .chunk(let chunk):
                                if let choice = chunk.choices.first {
                                    if let content = choice.delta.content, !content.isEmpty {
                                        continuation.yield(.content(content))
                                    }
                                    if let finishReason = choice.finishReason {
                                        continuation.yield(.finished(reason: finishReason))
                                    }
                                }
                            case .done:
                                continuation.finish()
                                return
                            }
                        }
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    // MARK: - Models

    /// Fetch available models from a proxy endpoint.
    func fetchModels(baseURL: String, apiKey: String) async throws -> [ModelInfo] {
        guard let url = URL(string: "\(baseURL)/v1/models") else {
            throw APIError.invalidURL("\(baseURL)/v1/models")
        }
        var request = URLRequest(url: url)
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw APIError.invalidResponse
        }

        let modelsResponse = try JSONDecoder().decode(ModelsResponse.self, from: data)
        return modelsResponse.data
    }

    // MARK: - Health

    /// Check the health of a proxy endpoint.
    func checkHealth(baseURL: String) async throws -> HealthResponse {
        guard let url = URL(string: "\(baseURL)/health") else {
            throw APIError.invalidURL("\(baseURL)/health")
        }
        let (data, response) = try await session.data(for: URLRequest(url: url))

        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw APIError.invalidResponse
        }

        return try JSONDecoder().decode(HealthResponse.self, from: data)
    }
}

// MARK: - Stream Token

/// A token received from the streaming chat completion.
enum StreamToken {
    case content(String)
    case finished(reason: String)
}

// MARK: - API Errors

enum APIError: LocalizedError {
    case invalidResponse
    case invalidURL(String)
    case httpError(statusCode: Int, body: String)
    case noActiveServer

    var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "Invalid response from server"
        case .invalidURL(let url):
            return "Invalid URL: \(url)"
        case .httpError(let code, let body):
            return "HTTP \(code): \(body.prefix(200))"
        case .noActiveServer:
            return "No server configured. Go to Settings to add your Deep Search Portal URL."
        }
    }
}
