import SwiftUI

/// A progress bar showing the current state of a deep research session.
struct ResearchProgressBar: View {
    let progress: ResearchProgress

    var body: some View {
        VStack(spacing: 6) {
            // Progress bar
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color(.systemGray5))
                        .frame(height: 6)

                    RoundedRectangle(cornerRadius: 4)
                        .fill(
                            progress.isComplete
                            ? Color.green.gradient
                            : Color.blue.gradient
                        )
                        .frame(
                            width: geometry.size.width * progress.progressFraction,
                            height: 6
                        )
                        .animation(.easeInOut, value: progress.progressFraction)
                }
            }
            .frame(height: 6)

            // Status text
            HStack {
                if progress.isComplete {
                    Label("Research complete", systemImage: "checkmark.circle.fill")
                        .font(.caption2)
                        .foregroundStyle(.green)
                } else if progress.currentTurn > 0 {
                    Label(
                        "Turn \(progress.currentTurn)/\(progress.maxTurns)",
                        systemImage: "arrow.clockwise"
                    )
                    .font(.caption2)
                    .foregroundStyle(.blue)
                } else {
                    Label("Starting research...", systemImage: "hourglass")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                if progress.toolCallCount > 0 {
                    Text("\(progress.toolCallCount) tool calls")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }

                if !progress.lastTool.isEmpty {
                    Text(progress.lastTool)
                        .font(.caption2)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color(.systemGray5), in: Capsule())
                }
            }
        }
        .padding(.vertical, 4)
    }
}
