# Deep Search iOS App

A native iOS client for the [Deep Search Portal](../README.md) — an anti-censorship research platform powered by MiroFlow deep research agents.

## Features

- **Streaming Chat** — Real-time SSE streaming of responses from all Deep Search Portal proxies
- **Thinking/Reasoning Display** — Collapsible sections showing the model's research process (`<think>` tags rendered as expandable "Research Process" blocks)
- **Research Progress** — Live progress bar showing current turn, tool calls, and research status during MiroFlow sessions
- **Multi-Model Support** — Switch between Thinking Proxy, MiroFlow Deep Research, Persistent Research, and direct Mistral models
- **Conversation Management** — Create, browse, and delete research conversations with local persistence
- **Server Configuration** — Connect to any Deep Search Portal instance with health check support
- **Markdown Rendering** — Rich text display for research answers with links, code, and formatting

## Architecture

```
┌─────────────────────────────────────────────┐
│                 iOS App                      │
│                                              │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐ │
│  │ ChatView │  │ Settings  │  │ Model    │ │
│  │          │  │ View      │  │ Picker   │ │
│  └────┬─────┘  └─────┬─────┘  └────┬─────┘ │
│       │              │              │        │
│  ┌────┴──────────────┴──────────────┴─────┐ │
│  │           ChatViewModel                │ │
│  │    (streaming, thinking parser)        │ │
│  └────────────────┬───────────────────────┘ │
│                   │                          │
│  ┌────────────────┴───────────────────────┐ │
│  │            APIClient                   │ │
│  │   (SSE streaming, health checks)      │ │
│  └────────────────┬───────────────────────┘ │
│                   │                          │
│  ┌────────────────┴───────────────────────┐ │
│  │  SSEStreamParser  │  ThinkingParser    │ │
│  │  (SSE protocol)   │  (<think> tags)    │ │
│  └────────────────────────────────────────┘ │
└──────────────────┬──────────────────────────┘
                   │ HTTPS (SSE)
                   ▼
         Deep Search Portal
    ┌─────────────────────────┐
    │  Thinking Proxy (9100)  │
    │  MiroFlow       (9200)  │
    │  Persistent     (9300)  │
    │  Knowledge Eng. (9400)  │
    └─────────────────────────┘
```

## Project Structure

```
ios/DeepSearch/
├── DeepSearch.xcodeproj/       # Xcode project
├── Package.swift               # Swift Package Manager manifest
└── DeepSearch/
    ├── DeepSearchApp.swift     # App entry point
    ├── Models/
    │   ├── APIModels.swift     # OpenAI-compatible request/response types
    │   ├── Conversation.swift  # Conversation data model
    │   ├── Message.swift       # Message with thinking/tool call support
    │   ├── Provider.swift      # LLM provider definitions
    │   └── ServerConfig.swift  # Server connection configuration
    ├── Services/
    │   ├── APIClient.swift     # Async HTTP client with SSE streaming
    │   ├── SSEStreamParser.swift   # Server-Sent Events parser
    │   ├── ThinkingParser.swift    # <think> tag state machine
    │   └── PersistenceService.swift # Local JSON file storage
    ├── ViewModels/
    │   ├── AppState.swift      # Global app state (conversations, servers)
    │   └── ChatViewModel.swift # Chat interaction + research progress
    ├── Views/
    │   ├── RootView.swift      # Split view navigation
    │   ├── ChatView.swift      # Main chat interface + input bar
    │   ├── ConversationListView.swift  # Sidebar conversation list
    │   ├── ModelPickerView.swift       # Model/provider selector
    │   ├── SettingsView.swift          # Server management + about
    │   ├── ServerEditorView.swift      # Add/edit server configuration
    │   └── Components/
    │       ├── MessageBubble.swift     # Chat bubble with thinking section
    │       ├── MarkdownText.swift      # Markdown text renderer
    │       └── ResearchProgressBar.swift # Research turn/tool progress
    └── Resources/
        └── Assets.xcassets/    # App icon and accent color
```

## Requirements

- iOS 17.0+
- Xcode 15.4+
- Swift 5.9+
- A running Deep Search Portal instance

## Getting Started

1. Open `ios/DeepSearch/DeepSearch.xcodeproj` in Xcode
2. Select your target device or simulator
3. Build and run (Cmd+R)
4. In the app, go to **Settings** and add your Deep Search Portal server:
   - **Base URL**: Your portal URL (e.g., `https://deep-search.uk` or `http://192.168.1.100:9200`)
   - **API Key**: Your Mistral API key or proxy auth token
5. Tap **Test Connection** to verify connectivity
6. Start a new conversation and select your preferred research model

## Supported Models

| Model | Provider | Description |
|---|---|---|
| `miroflow` | Deep Research (9200) | 15-round agentic research with web search, page reading, Python |
| `persistent-miroflow` | Persistent Research (9300) | Multi-session research with knowledge accumulation |
| `mistral-large-thinking` | Thinking Proxy (9100) | Step-by-step reasoning with collapsible thinking |
| `mistral-large-latest` | Mistral Direct | Direct Mistral Large access |
| `mistral-medium-latest` | Mistral Direct | Direct Mistral Medium access |

## API Compatibility

The app communicates via the standard OpenAI-compatible chat completions API:

- `POST /v1/chat/completions` — Streaming chat (SSE)
- `GET /v1/models` — List available models
- `GET /health` — Server health check

All Deep Search Portal proxies implement this protocol, so the app works with any of them.
