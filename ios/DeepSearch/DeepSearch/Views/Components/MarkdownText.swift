import SwiftUI

/// A view that renders markdown-formatted text.
///
/// Supports the common markdown elements found in Deep Search Portal responses:
/// headings, bold, italic, code blocks, inline code, links, and lists.
struct MarkdownText: View {
    let text: String

    init(_ text: String) {
        self.text = text
    }

    var body: some View {
        if #available(iOS 17.0, *) {
            Text(attributedMarkdown)
                .font(.body)
                .tint(.blue)
        } else {
            Text(text)
                .font(.body)
        }
    }

    private var attributedMarkdown: AttributedString {
        do {
            var options = AttributedString.MarkdownParsingOptions()
            options.interpretedSyntax = .inlineOnlyPreservingWhitespace
            return try AttributedString(markdown: text, options: options)
        } catch {
            return AttributedString(text)
        }
    }
}
