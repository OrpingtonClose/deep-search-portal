import SwiftUI

/// Settings view for managing server connections and app preferences.
struct SettingsView: View {
    @EnvironmentObject var appState: AppState
    @Environment(\.dismiss) private var dismiss
    @State private var showAddServer = false
    @State private var editingServer: ServerConfig?
    @State private var healthStatus: [UUID: ServerHealthStatus] = [:]

    var body: some View {
        NavigationStack {
            List {
                // Server configurations
                Section {
                    ForEach(appState.servers) { server in
                        ServerRow(
                            server: server,
                            healthStatus: healthStatus[server.id],
                            onEdit: { editingServer = server },
                            onSetActive: { appState.setActiveServer(server.id) }
                        )
                    }
                    .onDelete(perform: deleteServers)

                    Button {
                        showAddServer = true
                    } label: {
                        Label("Add Server", systemImage: "plus.circle")
                    }
                } header: {
                    Text("Servers")
                } footer: {
                    Text("Connect to your Deep Search Portal instance. The active server is used for all research requests.")
                }

                // About section
                Section {
                    HStack {
                        Text("Version")
                        Spacer()
                        Text("1.0.0")
                            .foregroundStyle(.secondary)
                    }

                    Link(destination: URL(string: "https://github.com/OrpingtonClose/deep-search-portal")!) {
                        HStack {
                            Label("GitHub Repository", systemImage: "link")
                            Spacer()
                            Image(systemName: "arrow.up.right.square")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                } header: {
                    Text("About")
                } footer: {
                    Text("Deep Search Portal - Anti-censorship research with MiroFlow deep research agent.")
                }

                // Architecture info
                Section {
                    ArchitectureInfoView()
                } header: {
                    Text("Architecture")
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
            .sheet(isPresented: $showAddServer) {
                ServerEditorView(server: ServerConfig())
            }
            .sheet(item: $editingServer) { server in
                ServerEditorView(server: server)
            }
            .task {
                await checkAllServerHealth()
            }
        }
    }

    private func deleteServers(at offsets: IndexSet) {
        for index in offsets {
            let server = appState.servers[index]
            appState.deleteServer(server.id)
        }
    }

    private func checkAllServerHealth() async {
        for server in appState.servers {
            guard !server.baseURL.isEmpty else { continue }
            do {
                let response = try await appState.apiClient.checkHealth(baseURL: server.normalizedBaseURL)
                healthStatus[server.id] = .healthy(service: response.service ?? "unknown")
            } catch {
                healthStatus[server.id] = .unhealthy(error: error.localizedDescription)
            }
        }
    }
}

// MARK: - Server Health Status

enum ServerHealthStatus {
    case healthy(service: String)
    case unhealthy(error: String)

    var isHealthy: Bool {
        if case .healthy = self { return true }
        return false
    }
}

// MARK: - Server Row

struct ServerRow: View {
    let server: ServerConfig
    let healthStatus: ServerHealthStatus?
    let onEdit: () -> Void
    let onSetActive: () -> Void

    var body: some View {
        Button(action: onSetActive) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 6) {
                        Text(server.name)
                            .font(.body)
                            .foregroundStyle(.primary)

                        if server.isActive {
                            Text("Active")
                                .font(.caption2)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.green.opacity(0.2), in: Capsule())
                                .foregroundStyle(.green)
                        }
                    }

                    if server.baseURL.isEmpty {
                        Text("Not configured")
                            .font(.caption)
                            .foregroundStyle(.orange)
                    } else {
                        Text(server.baseURL)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    // Health indicator
                    if let status = healthStatus {
                        HStack(spacing: 4) {
                            Circle()
                                .fill(status.isHealthy ? .green : .red)
                                .frame(width: 6, height: 6)
                            Text(status.isHealthy ? "Connected" : "Unreachable")
                                .font(.caption2)
                                .foregroundStyle(status.isHealthy ? .green : .red)
                        }
                    }
                }

                Spacer()

                Button(action: onEdit) {
                    Image(systemName: "pencil.circle")
                        .foregroundStyle(.blue)
                }
                .buttonStyle(.plain)

                if server.isActive {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                }
            }
        }
    }
}

// MARK: - Architecture Info

struct ArchitectureInfoView: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            architectureRow(
                icon: "brain.head.profile",
                title: "Thinking Proxy (9100)",
                description: "Step-by-step reasoning with <think> tags"
            )
            architectureRow(
                icon: "magnifyingglass",
                title: "MiroFlow (9200)",
                description: "15-round deep research with web search, page reading, Python"
            )
            architectureRow(
                icon: "arrow.triangle.2.circlepath",
                title: "Persistent Research (9300)",
                description: "Multi-session with subagent map-reduce and knowledge graph"
            )
            architectureRow(
                icon: "server.rack",
                title: "Knowledge Engine (9400)",
                description: "Neo4j graph with ETL pipeline and discovery algorithms"
            )
            architectureRow(
                icon: "globe",
                title: "SearXNG (8888)",
                description: "Self-hosted meta-search (Brave, Bing, Wikipedia)"
            )
        }
        .font(.caption)
    }

    private func architectureRow(icon: String, title: String, description: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: icon)
                .foregroundStyle(.blue)
                .frame(width: 20)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.caption.bold())
                Text(description)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }
}
