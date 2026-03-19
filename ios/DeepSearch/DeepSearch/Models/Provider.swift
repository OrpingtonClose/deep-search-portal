import Foundation

/// A model provider available through the Deep Search Portal.
struct Provider: Identifiable, Hashable {
    let id: String
    let name: String
    let port: Int
    let description: String
    let models: [Model]

    /// Pre-configured providers matching the Deep Search Portal architecture.
    static let defaults: [Provider] = [
        Provider(
            id: "thinking-proxy",
            name: "Thinking Proxy",
            port: 9100,
            description: "Wraps Mistral for step-by-step reasoning with <think> tag support",
            models: [Model(id: "mistral-large-thinking", name: "Mistral Large (Thinking)", provider: "thinking-proxy")]
        ),
        Provider(
            id: "deep-research",
            name: "Deep Research (MiroFlow)",
            port: 9200,
            description: "Agentic deep research - up to 15 rounds of search/read/analyze",
            models: [Model(id: "miroflow", name: "MiroFlow", provider: "deep-research")]
        ),
        Provider(
            id: "persistent-research",
            name: "Persistent Research",
            port: 9300,
            description: "Multi-session research with knowledge accumulation and subagent map-reduce",
            models: [Model(id: "persistent-miroflow", name: "Persistent MiroFlow", provider: "persistent-research")]
        ),
        Provider(
            id: "mistral-direct",
            name: "Mistral Direct",
            port: 0,
            description: "Direct access to Mistral models",
            models: [
                Model(id: "mistral-large-latest", name: "Mistral Large", provider: "mistral-direct"),
                Model(id: "mistral-medium-latest", name: "Mistral Medium", provider: "mistral-direct"),
            ]
        ),
    ]
}

/// A single LLM model available through a provider.
struct Model: Identifiable, Hashable, Codable {
    let id: String
    let name: String
    let provider: String
}
