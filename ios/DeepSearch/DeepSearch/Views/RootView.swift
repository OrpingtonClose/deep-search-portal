import SwiftUI

/// The root navigation view with a sidebar for conversations and a main chat area.
struct RootView: View {
    @EnvironmentObject var appState: AppState
    @State private var showSettings = false
    @State private var columnVisibility: NavigationSplitViewVisibility = .automatic

    var body: some View {
        NavigationSplitView(columnVisibility: $columnVisibility) {
            ConversationListView()
                .toolbar {
                    ToolbarItem(placement: .topBarLeading) {
                        Button {
                            showSettings = true
                        } label: {
                            Image(systemName: "gear")
                        }
                    }
                    ToolbarItem(placement: .topBarTrailing) {
                        Button {
                            let _ = appState.createConversation()
                        } label: {
                            Image(systemName: "square.and.pencil")
                        }
                    }
                }
                .navigationTitle("Conversations")
        } detail: {
            if appState.activeConversationID != nil {
                ChatView()
            } else {
                WelcomeView()
            }
        }
        .sheet(isPresented: $showSettings) {
            SettingsView()
        }
    }
}

/// Welcome screen shown when no conversation is selected.
struct WelcomeView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        VStack(spacing: 24) {
            Image(systemName: "magnifyingglass.circle.fill")
                .font(.system(size: 80))
                .foregroundStyle(.blue.gradient)

            Text("Deep Search")
                .font(.largeTitle.bold())

            Text("AI-powered deep research with multi-turn\nsearch, analysis, and reasoning")
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)

            if appState.activeServer == nil || appState.activeServer?.baseURL.isEmpty == true {
                VStack(spacing: 12) {
                    Text("Get Started")
                        .font(.headline)

                    Text("Configure your Deep Search Portal server\nin Settings to begin researching.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
                .padding(.top, 16)
            } else {
                Button {
                    let _ = appState.createConversation()
                } label: {
                    Label("New Research", systemImage: "plus.circle.fill")
                        .font(.headline)
                        .padding(.horizontal, 24)
                        .padding(.vertical, 12)
                }
                .buttonStyle(.borderedProminent)
                .padding(.top, 16)
            }

            Spacer()

            // Provider info
            VStack(alignment: .leading, spacing: 8) {
                Text("Available Research Modes")
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)

                ForEach(Provider.defaults) { provider in
                    HStack(spacing: 8) {
                        Image(systemName: iconForProvider(provider.id))
                            .foregroundStyle(.blue)
                            .frame(width: 20)
                        VStack(alignment: .leading) {
                            Text(provider.name)
                                .font(.caption.bold())
                            Text(provider.description)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .padding()
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
            .padding(.horizontal)
        }
        .padding()
    }

    private func iconForProvider(_ id: String) -> String {
        switch id {
        case "thinking-proxy": return "brain.head.profile"
        case "deep-research": return "magnifyingglass"
        case "persistent-research": return "arrow.triangle.2.circlepath"
        case "mistral-direct": return "bolt"
        default: return "circle"
        }
    }
}
