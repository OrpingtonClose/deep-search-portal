import SwiftUI

/// A sheet view for selecting the active research model/provider.
struct ModelPickerView: View {
    @EnvironmentObject var appState: AppState
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                ForEach(Provider.defaults) { provider in
                    Section {
                        ForEach(provider.models) { model in
                            Button {
                                appState.selectedModel = model.id
                                dismiss()
                            } label: {
                                HStack {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(model.name)
                                            .font(.body)
                                            .foregroundStyle(.primary)
                                        Text(provider.description)
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }

                                    Spacer()

                                    if appState.selectedModel == model.id {
                                        Image(systemName: "checkmark.circle.fill")
                                            .foregroundStyle(.blue)
                                    }
                                }
                            }
                        }
                    } header: {
                        HStack(spacing: 6) {
                            Image(systemName: iconForProvider(provider.id))
                            Text(provider.name)
                        }
                    } footer: {
                        if provider.port > 0 {
                            Text("Port \(provider.port)")
                        }
                    }
                }
            }
            .navigationTitle("Select Model")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
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
