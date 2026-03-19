import SwiftUI

/// Main chat view with message list, input bar, and research progress indicator.
struct ChatView: View {
    @EnvironmentObject var appState: AppState
    @StateObject private var viewModel = ChatViewModel()
    @State private var showModelPicker = false
    @FocusState private var isInputFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            // Messages
            messagesScrollView

            // Research progress bar
            if let progress = viewModel.researchProgress, viewModel.isStreaming {
                ResearchProgressBar(progress: progress)
                    .padding(.horizontal)
                    .padding(.top, 8)
            }

            // Error banner
            if let error = viewModel.error {
                errorBanner(error)
            }

            // Input bar
            inputBar
        }
        .navigationTitle(appState.activeConversation?.title ?? "Chat")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                modelPickerButton
            }
        }
        .sheet(isPresented: $showModelPicker) {
            ModelPickerView()
        }
    }

    // MARK: - Messages

    private var messagesScrollView: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 16) {
                    if let conversation = appState.activeConversation {
                        ForEach(conversation.messages) { message in
                            MessageBubble(message: message)
                                .id(message.id)
                        }
                    }
                }
                .padding()
            }
            .onChange(of: appState.activeConversation?.messages.count) { _, _ in
                scrollToBottom(proxy: proxy)
            }
            .onChange(of: viewModel.streamingAnswer) { _, _ in
                scrollToBottom(proxy: proxy)
            }
        }
    }

    private func scrollToBottom(proxy: ScrollViewProxy) {
        if let lastMessage = appState.activeConversation?.messages.last {
            withAnimation(.easeOut(duration: 0.2)) {
                proxy.scrollTo(lastMessage.id, anchor: .bottom)
            }
        }
    }

    // MARK: - Input Bar

    private var inputBar: some View {
        HStack(alignment: .bottom, spacing: 12) {
            TextField("Ask anything...", text: $viewModel.inputText, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...6)
                .focused($isInputFocused)
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .background(Color(.systemGray6), in: RoundedRectangle(cornerRadius: 20))
                .submitLabel(.send)
                .onSubmit {
                    viewModel.sendMessage(appState: appState)
                }

            if viewModel.isStreaming {
                Button {
                    viewModel.cancelStream()
                } label: {
                    Image(systemName: "stop.circle.fill")
                        .font(.title2)
                        .foregroundStyle(.red)
                }
            } else {
                Button {
                    viewModel.sendMessage(appState: appState)
                } label: {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.title2)
                        .foregroundStyle(
                            viewModel.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                            ? .gray : .blue
                        )
                }
                .disabled(viewModel.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial)
    }

    // MARK: - Model Picker Button

    private var modelPickerButton: some View {
        Button {
            showModelPicker = true
        } label: {
            HStack(spacing: 4) {
                Image(systemName: modelIcon(appState.selectedModel))
                    .font(.caption)
                Text(modelShortName(appState.selectedModel))
                    .font(.caption)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Color(.systemGray6), in: Capsule())
        }
    }

    // MARK: - Error Banner

    private func errorBanner(_ message: String) -> some View {
        HStack {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
            Text(message)
                .font(.caption)
                .lineLimit(2)
            Spacer()
            Button {
                viewModel.error = nil
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .background(Color.orange.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
        .padding(.horizontal)
        .padding(.top, 4)
    }

    // MARK: - Helpers

    private func modelIcon(_ model: String) -> String {
        switch model {
        case "miroflow": return "magnifyingglass"
        case "persistent-miroflow": return "arrow.triangle.2.circlepath"
        case "mistral-large-thinking": return "brain.head.profile"
        default: return "bolt"
        }
    }

    private func modelShortName(_ model: String) -> String {
        switch model {
        case "miroflow": return "MiroFlow"
        case "persistent-miroflow": return "Persistent"
        case "mistral-large-thinking": return "Thinking"
        case "mistral-large-latest": return "Large"
        case "mistral-medium-latest": return "Medium"
        default: return model
        }
    }
}
