import Foundation

// MARK: - OpenAI-Compatible API Request/Response Models

/// Request body for the chat completions endpoint.
struct ChatCompletionRequest: Codable {
    let model: String
    let messages: [ChatMessage]
    let stream: Bool
    var temperature: Double?
    var maxTokens: Int?

    enum CodingKeys: String, CodingKey {
        case model, messages, stream, temperature
        case maxTokens = "max_tokens"
    }
}

/// A message in the OpenAI chat format.
struct ChatMessage: Codable {
    let role: String
    let content: String
}

/// A single SSE chunk from the streaming chat completions response.
struct ChatCompletionChunk: Codable {
    let id: String
    let object: String
    let created: Int
    let model: String
    let choices: [ChunkChoice]
}

struct ChunkChoice: Codable {
    let index: Int
    let delta: ChunkDelta
    let finishReason: String?

    enum CodingKeys: String, CodingKey {
        case index, delta
        case finishReason = "finish_reason"
    }
}

struct ChunkDelta: Codable {
    let role: String?
    let content: String?
}

/// Response from the /v1/models endpoint.
struct ModelsResponse: Codable {
    let data: [ModelInfo]
}

struct ModelInfo: Codable, Identifiable {
    let id: String
    let object: String?
    let created: Int?
    let ownedBy: String?

    enum CodingKeys: String, CodingKey {
        case id, object, created
        case ownedBy = "owned_by"
    }
}

/// Health check response.
struct HealthResponse: Codable {
    let status: String
    let service: String?
    let activeRequests: Int?

    enum CodingKeys: String, CodingKey {
        case status, service
        case activeRequests = "active_requests"
    }
}
