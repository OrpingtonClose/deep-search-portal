import SwiftUI

/// Editor view for adding or editing a server configuration.
struct ServerEditorView: View {
    @EnvironmentObject var appState: AppState
    @Environment(\.dismiss) private var dismiss

    @State var server: ServerConfig
    @State private var isTesting = false
    @State private var testResult: TestResult?

    private var isNewServer: Bool {
        !appState.servers.contains(where: { $0.id == server.id })
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("Server Name", text: $server.name)
                        .textContentType(.name)

                    TextField("Base URL", text: $server.baseURL)
                        .textContentType(.URL)
                        .keyboardType(.URL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()

                    SecureField("API Key", text: $server.apiKey)
                        .textContentType(.password)
                        .textInputAutocapitalization(.never)

                    Toggle("Active", isOn: $server.isActive)
                } header: {
                    Text("Connection")
                } footer: {
                    Text("Enter the URL of your Deep Search Portal instance (e.g., https://deep-search.uk or http://192.168.1.100:9200). The API key is your Mistral API key or proxy authentication token.")
                }

                Section {
                    Button {
                        Task {
                            await testConnection()
                        }
                    } label: {
                        HStack {
                            Label("Test Connection", systemImage: "antenna.radiowaves.left.and.right")
                            Spacer()
                            if isTesting {
                                ProgressView()
                            }
                        }
                    }
                    .disabled(server.baseURL.isEmpty || isTesting)

                    if let result = testResult {
                        HStack(spacing: 8) {
                            Image(systemName: result.success ? "checkmark.circle.fill" : "xmark.circle.fill")
                                .foregroundStyle(result.success ? .green : .red)
                            Text(result.message)
                                .font(.caption)
                                .foregroundStyle(result.success ? .green : .red)
                        }
                    }
                } header: {
                    Text("Connection Test")
                }

                Section {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Proxy Endpoints")
                            .font(.caption.bold())

                        ForEach(endpointExamples, id: \.path) { endpoint in
                            HStack {
                                Text(endpoint.path)
                                    .font(.caption.monospaced())
                                Spacer()
                                Text(endpoint.description)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                } header: {
                    Text("API Reference")
                }
            }
            .navigationTitle(isNewServer ? "Add Server" : "Edit Server")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Save") {
                        saveServer()
                    }
                    .bold()
                    .disabled(server.baseURL.isEmpty)
                }
            }
        }
    }

    private func saveServer() {
        if isNewServer {
            appState.addServer(server)
        } else {
            appState.updateServer(server)
        }
        dismiss()
    }

    private func testConnection() async {
        isTesting = true
        testResult = nil

        do {
            let health = try await appState.apiClient.checkHealth(
                baseURL: server.normalizedBaseURL
            )
            testResult = TestResult(
                success: true,
                message: "Connected to \(health.service ?? "server") - \(health.status)"
            )
        } catch {
            testResult = TestResult(
                success: false,
                message: error.localizedDescription
            )
        }

        isTesting = false
    }

    private var endpointExamples: [(path: String, description: String)] {
        [
            ("/v1/chat/completions", "Chat (streaming SSE)"),
            ("/v1/models", "List models"),
            ("/health", "Health check"),
            ("/logs", "View proxy logs"),
        ]
    }
}

struct TestResult {
    let success: Bool
    let message: String
}
