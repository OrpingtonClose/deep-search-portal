import SwiftUI

/// A chat message bubble with support for thinking/reasoning sections.
struct MessageBubble: View {
    let message: Message
    @State private var isThinkingExpanded = false

    var body: some View {
        HStack(alignment: .top) {
            if message.role == .user {
                Spacer(minLength: 60)
            }

            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 8) {
                // Role label
                if message.role == .assistant {
                    HStack(spacing: 4) {
                        Image(systemName: "sparkles")
                            .font(.caption2)
                        if let model = message.model {
                            Text(modelDisplayName(model))
                                .font(.caption2)
                        } else {
                            Text("Assistant")
                                .font(.caption2)
                        }
                    }
                    .foregroundStyle(.secondary)
                }

                // Thinking section (collapsible)
                if message.hasThinking {
                    ThinkingSection(
                        content: message.thinkingContent,
                        isExpanded: $isThinkingExpanded,
                        isStreaming: message.isStreaming && message.content.isEmpty
                    )
                }

                // Main content
                if !message.content.isEmpty || message.isStreaming {
                    contentBubble
                }

                // Streaming indicator
                if message.isStreaming && message.content.isEmpty && message.thinkingContent.isEmpty {
                    HStack(spacing: 6) {
                        ProgressView()
                            .scaleEffect(0.7)
                        Text("Connecting...")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(12)
                    .background(Color(.systemGray6), in: RoundedRectangle(cornerRadius: 16))
                }

                // Timestamp
                Text(message.timestamp, style: .time)
                    .font(.caption2)
                    .foregroundStyle(.quaternary)
            }

            if message.role == .assistant {
                Spacer(minLength: 20)
            }
        }
    }

    private var contentBubble: some View {
        Group {
            if message.role == .user {
                Text(message.content)
                    .padding(12)
                    .background(Color.blue, in: RoundedRectangle(cornerRadius: 16))
                    .foregroundStyle(.white)
            } else {
                MarkdownText(message.content)
                    .padding(12)
                    .background(Color(.systemGray6), in: RoundedRectangle(cornerRadius: 16))
                    .if(message.isStreaming) { view in
                        view.overlay(alignment: .bottomTrailing) {
                            StreamingDot()
                                .padding(8)
                        }
                    }
            }
        }
        .textSelection(.enabled)
    }

    private func modelDisplayName(_ model: String) -> String {
        switch model {
        case "miroflow": return "MiroFlow Deep Research"
        case "persistent-miroflow": return "Persistent Research"
        case "mistral-large-thinking": return "Mistral Thinking"
        case "mistral-large-latest": return "Mistral Large"
        case "mistral-medium-latest": return "Mistral Medium"
        default: return model
        }
    }
}

// MARK: - Thinking Section

/// A collapsible section showing the model's reasoning process.
struct ThinkingSection: View {
    let content: String
    @Binding var isExpanded: Bool
    let isStreaming: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            Button {
                withAnimation(.easeInOut(duration: 0.2)) {
                    isExpanded.toggle()
                }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                        .font(.caption2.bold())
                    Image(systemName: "brain")
                        .font(.caption)
                    Text(isStreaming ? "Researching..." : "Research Process")
                        .font(.caption.bold())

                    if isStreaming {
                        ProgressView()
                            .scaleEffect(0.5)
                    }

                    Spacer()

                    if !isExpanded {
                        Text("\(content.count) chars")
                            .font(.caption2)
                            .foregroundStyle(.quaternary)
                    }
                }
                .foregroundStyle(.orange)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
            }

            // Expandable content
            if isExpanded {
                Divider()
                    .padding(.horizontal, 12)

                ScrollView {
                    Text(content)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 400)
            }
        }
        .background(Color.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.orange.opacity(0.2), lineWidth: 1)
        )
    }
}

// MARK: - Streaming Dot

/// Animated dot indicating active streaming.
struct StreamingDot: View {
    @State private var isAnimating = false

    var body: some View {
        Circle()
            .fill(.blue)
            .frame(width: 8, height: 8)
            .opacity(isAnimating ? 0.3 : 1.0)
            .onAppear {
                withAnimation(.easeInOut(duration: 0.6).repeatForever()) {
                    isAnimating = true
                }
            }
    }
}

// MARK: - Conditional Modifier

extension View {
    @ViewBuilder
    func `if`<Content: View>(_ condition: Bool, transform: (Self) -> Content) -> some View {
        if condition {
            transform(self)
        } else {
            self
        }
    }
}
