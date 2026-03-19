import Foundation
import SwiftUI

/// Global application state managing conversations, server config, and model selection.
@MainActor
final class AppState: ObservableObject {
    @Published var conversations: [Conversation] = []
    @Published var activeConversationID: UUID?
    @Published var servers: [ServerConfig] = []
    @Published var selectedModel: String = "miroflow"
    @Published var isLoading = false
    @Published var errorMessage: String?

    let apiClient = APIClient()

    var activeServer: ServerConfig? {
        servers.first(where: { $0.isActive })
    }

    var activeConversation: Conversation? {
        get {
            guard let id = activeConversationID else { return nil }
            return conversations.first(where: { $0.id == id })
        }
        set {
            guard let id = activeConversationID,
                  let index = conversations.firstIndex(where: { $0.id == id }),
                  let newValue else { return }
            conversations[index] = newValue
        }
    }

    init() {
        Task {
            await loadData()
        }
    }

    // MARK: - Data Loading

    func loadData() async {
        do {
            let loadedServers = try await PersistenceService.shared.loadServers()
            let loadedConversations = try await PersistenceService.shared.loadConversations()

            servers = loadedServers
            conversations = loadedConversations.sorted { $0.updatedAt > $1.updatedAt }

            if servers.isEmpty {
                // Add a default empty server config to prompt setup
                servers = [ServerConfig()]
            }
        } catch {
            // First launch — start with defaults
            servers = [ServerConfig()]
            conversations = []
        }
    }

    // MARK: - Persistence

    func saveConversations() {
        Task {
            try? await PersistenceService.shared.saveConversations(conversations)
        }
    }

    func saveServers() {
        Task {
            try? await PersistenceService.shared.saveServers(servers)
        }
    }

    // MARK: - Conversation Management

    func createConversation() -> Conversation {
        let conversation = Conversation(model: selectedModel)
        conversations.insert(conversation, at: 0)
        activeConversationID = conversation.id
        saveConversations()
        return conversation
    }

    func deleteConversation(_ id: UUID) {
        conversations.removeAll { $0.id == id }
        if activeConversationID == id {
            activeConversationID = conversations.first?.id
        }
        saveConversations()
    }

    func selectConversation(_ id: UUID) {
        activeConversationID = id
    }

    // MARK: - Server Management

    func addServer(_ server: ServerConfig) {
        // Deactivate other servers if this one is active
        if server.isActive {
            for i in servers.indices {
                servers[i].isActive = false
            }
        }
        servers.append(server)
        saveServers()
    }

    func updateServer(_ server: ServerConfig) {
        guard let index = servers.firstIndex(where: { $0.id == server.id }) else { return }
        if server.isActive {
            for i in servers.indices {
                servers[i].isActive = false
            }
        }
        servers[index] = server
        saveServers()
    }

    func deleteServer(_ id: UUID) {
        servers.removeAll { $0.id == id }
        saveServers()
    }

    func setActiveServer(_ id: UUID) {
        for i in servers.indices {
            servers[i].isActive = servers[i].id == id
        }
        saveServers()
    }
}
