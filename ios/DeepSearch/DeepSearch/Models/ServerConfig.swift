import Foundation

/// Configuration for connecting to a Deep Search Portal backend instance.
struct ServerConfig: Codable, Identifiable, Hashable {
    var id: UUID
    var name: String
    var baseURL: String
    var apiKey: String
    var isActive: Bool

    init(
        id: UUID = UUID(),
        name: String = "Deep Search Portal",
        baseURL: String = "",
        apiKey: String = "",
        isActive: Bool = true
    ) {
        self.id = id
        self.name = name
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.isActive = isActive
    }

    /// The base URL with trailing slash removed.
    var normalizedBaseURL: String {
        baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    }
}
