import Foundation
import SwiftUI

/// Manages the chat interaction for a single conversation, including
/// streaming responses and parsing thinking/answer content.
@MainActor
final class ChatViewModel: ObservableObject {
    @Published var inputText = ""
    @Published var isStreaming = false
    @Published var streamingThinking = ""
    @Published var streamingAnswer = ""
    @Published var error: String?
    @Published var researchProgress: ResearchProgress?

    private let apiClient = APIClient()
    private var streamTask: Task<Void, Never>?

    // MARK: - Send Message

    func sendMessage(appState: AppState) {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        guard !isStreaming else { return }

        guard let server = appState.activeServer,
              !server.baseURL.isEmpty else {
            error = "No server configured. Go to Settings to add your Deep Search Portal URL."
            return
        }

        // Get or create conversation
        let conversationID: UUID
        if let activeID = appState.activeConversationID {
            conversationID = activeID
        } else {
            let conversation = appState.createConversation()
            conversationID = conversation.id
        }

        // Add user message
        let userMessage = Message(role: .user, content: text)
        appendMessage(userMessage, to: conversationID, appState: appState)
        inputText = ""
        error = nil

        // Start streaming response
        streamTask = Task {
            await streamResponse(
                conversationID: conversationID,
                server: server,
                model: appState.selectedModel,
                appState: appState
            )
        }
    }

    func cancelStream() {
        streamTask?.cancel()
        streamTask = nil
        isStreaming = false
    }

    // MARK: - Streaming

    private func streamResponse(
        conversationID: UUID,
        server: ServerConfig,
        model: String,
        appState: AppState
    ) async {
        isStreaming = true
        streamingThinking = ""
        streamingAnswer = ""
        researchProgress = ResearchProgress()

        // Build messages for API call
        guard let conversation = appState.conversations.first(where: { $0.id == conversationID }) else {
            isStreaming = false
            return
        }

        let chatMessages = conversation.messages.map { msg in
            ChatMessage(role: msg.role.rawValue, content: msg.content)
        }

        // Add a placeholder assistant message for streaming
        let assistantMessage = Message(
            role: .assistant,
            content: "",
            isStreaming: true,
            model: model
        )
        appendMessage(assistantMessage, to: conversationID, appState: appState)

        let parser = ThinkingParser()

        do {
            let stream = apiClient.streamChatCompletion(
                baseURL: server.normalizedBaseURL,
                apiKey: server.apiKey,
                messages: chatMessages,
                model: model
            )

            for try await token in stream {
                if Task.isCancelled { break }

                switch token {
                case .content(let text):
                    let (thinkDelta, answerDelta) = parser.process(text)

                    if !thinkDelta.isEmpty {
                        streamingThinking += thinkDelta
                        parseResearchProgress(thinkDelta)
                    }
                    if !answerDelta.isEmpty {
                        streamingAnswer += answerDelta
                    }

                    // Update the message in the conversation
                    updateStreamingMessage(
                        conversationID: conversationID,
                        thinking: parser.thinkingContent,
                        answer: parser.answerContent,
                        appState: appState
                    )

                case .finished:
                    break
                }
            }

        } catch {
            self.error = error.localizedDescription
        }

        // Flush any remaining buffered content (runs even after stream errors)
        let (thinkDelta, answerDelta) = parser.flush()
        if !thinkDelta.isEmpty { streamingThinking += thinkDelta }
        if !answerDelta.isEmpty { streamingAnswer += answerDelta }

        // Finalize the message
        finalizeStreamingMessage(
            conversationID: conversationID,
            thinking: parser.thinkingContent,
            answer: parser.answerContent,
            model: model,
            appState: appState
        )

        isStreaming = false
        researchProgress = nil
        appState.saveConversations()

        // Generate title if this is the first exchange
        if let conv = appState.conversations.first(where: { $0.id == conversationID }),
           conv.title == "New Conversation",
           let firstUserMsg = conv.messages.first(where: { $0.role == .user }) {
            await generateTitle(
                for: conversationID,
                firstMessage: firstUserMsg.content,
                appState: appState
            )
        }
    }

    // MARK: - Research Progress Parsing

    private func parseResearchProgress(_ text: String) {
        // Parse turn indicators like "[Turn 3/15]"
        if let turnMatch = text.range(of: #"\[Turn (\d+)/(\d+)\]"#, options: .regularExpression) {
            let turnText = String(text[turnMatch])
            let numbers = turnText.components(separatedBy: CharacterSet.decimalDigits.inverted)
                .filter { !$0.isEmpty }
                .compactMap { Int($0) }
            if numbers.count >= 2 {
                researchProgress?.currentTurn = numbers[0]
                researchProgress?.maxTurns = numbers[1]
            }
        }

        // Parse tool calls
        if text.contains("Searching:") || text.contains("magnifyingglass") {
            researchProgress?.toolCallCount += 1
            researchProgress?.lastTool = "Web Search"
        } else if text.contains("Reading:") || text.contains("Fetching") {
            researchProgress?.toolCallCount += 1
            researchProgress?.lastTool = "Fetch Page"
        } else if text.contains("Running code") || text.contains("python") {
            researchProgress?.toolCallCount += 1
            researchProgress?.lastTool = "Python"
        }

        // Parse completion
        if text.contains("Research complete") || text.contains("Generating answer") {
            researchProgress?.isComplete = true
        }
    }

    // MARK: - Message Management

    private func appendMessage(_ message: Message, to conversationID: UUID, appState: AppState) {
        guard let index = appState.conversations.firstIndex(where: { $0.id == conversationID }) else { return }
        appState.conversations[index].messages.append(message)
        appState.conversations[index].updatedAt = Date()
    }

    private func updateStreamingMessage(
        conversationID: UUID,
        thinking: String,
        answer: String,
        appState: AppState
    ) {
        guard let convIndex = appState.conversations.firstIndex(where: { $0.id == conversationID }),
              let msgIndex = appState.conversations[convIndex].messages.lastIndex(where: { $0.isStreaming })
        else { return }

        appState.conversations[convIndex].messages[msgIndex].thinkingContent = thinking
        appState.conversations[convIndex].messages[msgIndex].content = answer
    }

    private func finalizeStreamingMessage(
        conversationID: UUID,
        thinking: String,
        answer: String,
        model: String,
        appState: AppState
    ) {
        guard let convIndex = appState.conversations.firstIndex(where: { $0.id == conversationID }),
              let msgIndex = appState.conversations[convIndex].messages.lastIndex(where: { $0.isStreaming })
        else { return }

        appState.conversations[convIndex].messages[msgIndex].thinkingContent = thinking
        appState.conversations[convIndex].messages[msgIndex].content = answer
        appState.conversations[convIndex].messages[msgIndex].isStreaming = false
        appState.conversations[convIndex].messages[msgIndex].model = model
    }

    // MARK: - Title Generation

    private func generateTitle(
        for conversationID: UUID,
        firstMessage: String,
        appState: AppState
    ) async {
        // Generate a short title from the first message
        let title: String
        let words = firstMessage.components(separatedBy: .whitespacesAndNewlines)
        if words.count <= 6 {
            title = firstMessage
        } else {
            title = words.prefix(6).joined(separator: " ") + "..."
        }

        guard let index = appState.conversations.firstIndex(where: { $0.id == conversationID }) else { return }
        appState.conversations[index].title = title
        appState.saveConversations()
    }
}

/// Tracks the progress of a deep research session.
struct ResearchProgress {
    var currentTurn: Int = 0
    var maxTurns: Int = 15
    var toolCallCount: Int = 0
    var lastTool: String = ""
    var isComplete: Bool = false

    var progressFraction: Double {
        guard maxTurns > 0 else { return 0 }
        return Double(currentTurn) / Double(maxTurns)
    }
}
