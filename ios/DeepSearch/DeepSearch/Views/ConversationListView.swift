import SwiftUI

/// Sidebar view showing all conversations with swipe-to-delete support.
struct ConversationListView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        Group {
            if appState.conversations.isEmpty {
                ContentUnavailableView(
                    "No Conversations",
                    systemImage: "bubble.left.and.bubble.right",
                    description: Text("Start a new research session to begin.")
                )
            } else {
                List(selection: $appState.activeConversationID) {
                    ForEach(appState.conversations) { conversation in
                        ConversationRow(conversation: conversation)
                            .tag(conversation.id)
                    }
                    .onDelete(perform: deleteConversations)
                }
                .listStyle(.sidebar)
            }
        }
    }

    private func deleteConversations(at offsets: IndexSet) {
        for index in offsets {
            let conversation = appState.conversations[index]
            appState.deleteConversation(conversation.id)
        }
    }
}

/// A single row in the conversation list.
struct ConversationRow: View {
    let conversation: Conversation

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(conversation.title)
                    .font(.headline)
                    .lineLimit(1)

                Spacer()

                Text(conversation.updatedAt, style: .relative)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            Text(conversation.preview)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(2)

            HStack(spacing: 4) {
                Image(systemName: modelIcon(conversation.model))
                    .font(.caption2)
                Text(modelDisplayName(conversation.model))
                    .font(.caption2)
            }
            .foregroundStyle(.tertiary)
        }
        .padding(.vertical, 4)
    }

    private func modelIcon(_ model: String) -> String {
        switch model {
        case "miroflow":
            return "magnifyingglass"
        case "persistent-miroflow":
            return "arrow.triangle.2.circlepath"
        case "mistral-large-thinking":
            return "brain.head.profile"
        default:
            return "bolt"
        }
    }

    private func modelDisplayName(_ model: String) -> String {
        switch model {
        case "miroflow":
            return "MiroFlow"
        case "persistent-miroflow":
            return "Persistent Research"
        case "mistral-large-thinking":
            return "Thinking"
        case "mistral-large-latest":
            return "Mistral Large"
        case "mistral-medium-latest":
            return "Mistral Medium"
        default:
            return model
        }
    }
}
