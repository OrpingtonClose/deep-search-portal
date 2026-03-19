import Foundation

/// Handles local persistence of conversations and server configurations.
///
/// Uses JSON files in the app's documents directory for simple, reliable storage.
actor PersistenceService {
    static let shared = PersistenceService()

    private let fileManager = FileManager.default

    private var documentsDirectory: URL {
        fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    private var conversationsURL: URL {
        documentsDirectory.appendingPathComponent("conversations.json")
    }

    private var serversURL: URL {
        documentsDirectory.appendingPathComponent("servers.json")
    }

    // MARK: - Conversations

    func saveConversations(_ conversations: [Conversation]) throws {
        let data = try JSONEncoder().encode(conversations)
        try data.write(to: conversationsURL, options: .atomic)
    }

    func loadConversations() throws -> [Conversation] {
        guard fileManager.fileExists(atPath: conversationsURL.path) else {
            return []
        }
        let data = try Data(contentsOf: conversationsURL)
        return try JSONDecoder().decode([Conversation].self, from: data)
    }

    // MARK: - Server Configs

    func saveServers(_ servers: [ServerConfig]) throws {
        let data = try JSONEncoder().encode(servers)
        try data.write(to: serversURL, options: .atomic)
    }

    func loadServers() throws -> [ServerConfig] {
        guard fileManager.fileExists(atPath: serversURL.path) else {
            return []
        }
        let data = try Data(contentsOf: serversURL)
        return try JSONDecoder().decode([ServerConfig].self, from: data)
    }
}
