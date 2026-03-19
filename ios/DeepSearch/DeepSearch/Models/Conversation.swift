import Foundation

/// A conversation containing a series of messages.
struct Conversation: Identifiable, Codable {
    let id: UUID
    var title: String
    var messages: [Message]
    var model: String
    var createdAt: Date
    var updatedAt: Date

    init(
        id: UUID = UUID(),
        title: String = "New Conversation",
        messages: [Message] = [],
        model: String = "miroflow",
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.title = title
        self.messages = messages
        self.model = model
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    /// The last user message in the conversation, used for generating titles.
    var lastUserMessage: String? {
        messages.last(where: { $0.role == .user })?.content
    }

    /// A preview string for the conversation list.
    var preview: String {
        if let lastMessage = messages.last {
            let content = lastMessage.content.isEmpty ? lastMessage.thinkingContent : lastMessage.content
            return String(content.prefix(120))
        }
        return "No messages"
    }
}
