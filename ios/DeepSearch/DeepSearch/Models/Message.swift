import Foundation

/// A single chat message in a conversation.
struct Message: Identifiable, Codable, Equatable {
    let id: UUID
    let role: Role
    var content: String
    let timestamp: Date
    var thinkingContent: String
    var isStreaming: Bool
    var model: String?
    var toolCalls: [ToolCallInfo]

    init(
        id: UUID = UUID(),
        role: Role,
        content: String,
        timestamp: Date = Date(),
        thinkingContent: String = "",
        isStreaming: Bool = false,
        model: String? = nil,
        toolCalls: [ToolCallInfo] = []
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.timestamp = timestamp
        self.thinkingContent = thinkingContent
        self.isStreaming = isStreaming
        self.model = model
        self.toolCalls = toolCalls
    }

    enum Role: String, Codable {
        case system
        case user
        case assistant
    }

    /// Whether this message contains thinking/reasoning content.
    var hasThinking: Bool {
        !thinkingContent.isEmpty
    }
}

/// Information about a tool call made during research.
struct ToolCallInfo: Codable, Equatable, Identifiable {
    let id: UUID
    let name: String
    let arguments: String
    var result: String
    let timestamp: Date

    init(
        id: UUID = UUID(),
        name: String,
        arguments: String,
        result: String = "",
        timestamp: Date = Date()
    ) {
        self.id = id
        self.name = name
        self.arguments = arguments
        self.result = result
        self.timestamp = timestamp
    }

    var icon: String {
        switch name {
        case "searxng_search", "web_search":
            return "magnifyingglass"
        case "fetch_webpage":
            return "doc.text"
        case "python_exec":
            return "terminal"
        case "knowledge_graph_search":
            return "brain"
        case "knowledge_discover":
            return "sparkles"
        case "arxiv_search":
            return "book"
        case "wayback_fetch":
            return "clock.arrow.circlepath"
        case "wikidata_query":
            return "w.circle"
        case "news_search":
            return "newspaper"
        default:
            return "wrench"
        }
    }

    var displayName: String {
        switch name {
        case "searxng_search":
            return "Web Search"
        case "fetch_webpage":
            return "Fetch Page"
        case "python_exec":
            return "Python"
        case "knowledge_graph_search":
            return "Knowledge Search"
        case "knowledge_discover":
            return "Discovery"
        case "arxiv_search":
            return "ArXiv Search"
        case "wayback_fetch":
            return "Wayback Machine"
        case "wikidata_query":
            return "Wikidata"
        case "web_search":
            return "Web Search"
        case "news_search":
            return "News Search"
        default:
            return name
        }
    }
}
