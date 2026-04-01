# Persistent MiroFlow — Behavioral Specification

This document codifies the behavioral assumptions of the Persistent MiroFlow
proxy.  Every assertion below is testable; the corresponding test IDs are
listed so that `tests/` can reference them.

---

## 1. File Attachment Handling

### PMFB-ATT-01  Attachments are first-priority inputs
When file attachments are detected in a user message, they MUST be processed
**before** any other routing decision (large-document ingestion, normal
research).  The proxy decomposes the documents into atomic conditions,
cross-references internal facts, and fact-checks assertions against external
sources.

### PMFB-ATT-02  Attachment block parsing is nesting-aware
The LibreChat "Upload as Text" format wraps content in a triple-backtick `md`
block.  Uploaded documents may themselves contain code fences
(e.g. ` ```python ... ``` `).  The parser MUST use nesting-aware fence
tracking (not a single regex) so that:
- Code fences **inside** a document do not prematurely close the attachment block.
- Code fences **in the user prompt** after the attachment block are not consumed
  into the document content.

### PMFB-ATT-03  Document content becomes a system message
Each attached document's full text is injected as a system message with
directives to decompose, cross-reference, and fact-check.  The system
message MUST appear immediately before the user message in the message
array.

### PMFB-ATT-04  Prompt provides research direction
When the user provides both file attachments and a text prompt, the prompt
is extracted and used as the **research direction**.  It tells the pipeline
what angle or focus to apply to the attached material.

### PMFB-ATT-05  Default directive when no prompt given
When the user uploads file(s) without typing a prompt, the proxy MUST
substitute a default research directive that instructs the pipeline to
analyse the document(s) thoroughly: decompose claims, cross-reference
facts, identify contradictions, and fact-check key assertions.

### PMFB-ATT-06  Multiple documents supported
Multiple files within a single attachment block (separated by `---`) MUST
each be parsed as individual `AttachedDocument` objects with their own
filename and content.

---

## 2. Follow-Up Prompt Behavior

### PMFB-FU-01  Follow-up prompts restart research with new focus
Sending a new prompt does NOT stop or cancel the previous research process.
Instead, a new research session begins with the new focus.  Even if the
prior session has not yet finished, the new prompt launches its own
pipeline run.

### PMFB-FU-02  Prompt inheritance — new takes precedence on conflict
When a follow-up prompt contradicts a term from the prior prompt, the new
prompt's term MUST take precedence.  For example, if the prior prompt said
"focus on cost" and the new prompt says "focus on safety", the research
direction becomes safety-focused.

### PMFB-FU-03  Prompt inheritance — non-contradicting terms inherited
Terms from the prior prompt that are NOT contradicted by the new prompt
MUST be inherited into the new research direction.  For example, if the
prior prompt said "focus on insulin vendors in the UK, cost analysis"
and the new prompt says "now look at safety records", the inherited
direction becomes "insulin vendors in the UK, safety records" (UK scope
and insulin vendor subject are inherited; cost is replaced by safety).

### PMFB-FU-04  Prompt inheritance — unstated aspects inherited
Aspects of the research focus that are not mentioned at all in the new
prompt are assumed unchanged and MUST be carried forward.  The new prompt
only overrides what it explicitly states.

### PMFB-FU-05  Conversation continuity via conversation_id
Follow-up detection uses `derive_conversation_id()` to identify the
conversation thread.  Prior turns' conditions, entities, and summaries
are loaded from the conversation store and injected into the new
research session's initial state.

### PMFB-FU-06  File attachments in follow-ups get first priority
If a follow-up message includes file attachments, those files MUST be
decomposed with first priority, the same as in the initial message
(PMFB-ATT-01).  The prompt-inheritance rules (PMFB-FU-02 through
PMFB-FU-04) still apply to the text prompt portion.

---

## 3. Research Pipeline Behavior

### PMFB-RP-01  Utility requests bypass research
Messages matching utility patterns (title generation, tag generation,
autocomplete) are routed to passthrough mode and never enter the
research pipeline.

### PMFB-RP-02  Large documents without attachment markers route to ingestion
Plain text messages exceeding the large-document threshold (10,000 chars)
that do NOT contain LibreChat attachment markers are routed to the
document ingestion flow, not the research pipeline.

### PMFB-RP-03  Concurrency limiting
The proxy enforces a maximum number of concurrent research sessions.
When the limit is reached, new requests receive a 503 response.

### PMFB-RP-04  Research sessions are independent
Each call to `run_persistent_research` creates its own pipeline instance,
output queue, heartbeat, metrics collector, and checkpointer.  Multiple
concurrent sessions do not interfere with each other.

---

## 4. Prompt Merging Rules (Detailed)

The prompt-merge operation takes two inputs:
- `prior_focus`: the research direction from the most recent prior turn
  (may itself be a merged result from earlier turns)
- `new_prompt`: the user's new text prompt

And produces:
- `merged_focus`: the effective research direction for the new session

### PMFB-PM-01  Merge is performed by the LLM
The merge is delegated to the upstream LLM via a structured prompt that
explains the three inheritance rules (precedence, non-contradiction
inheritance, unstated inheritance).  The LLM returns the merged focus
as a single coherent research direction string.

### PMFB-PM-02  Merge prompt template
The merge prompt MUST instruct the LLM with these rules:
1. If the new prompt explicitly states something that conflicts with the
   prior focus, the new prompt wins.
2. If the prior focus contains terms/constraints not mentioned or
   contradicted by the new prompt, they carry forward unchanged.
3. The output is a single merged research direction — not a diff or
   explanation.

### PMFB-PM-03  First message has no merge
The very first message in a conversation has no prior focus.  The prompt
(or default directive) is used as-is with no merge step.

### PMFB-PM-04  Merge applies to the text prompt only
File attachment content is never merged — it is always injected in full.
Only the text prompt (research direction) goes through the merge logic.

---

## 5. Swarm Proxy — Document Ingestion

### PMFB-SW-01  Attachments submitted as corpus for gossiping
When the Swarm proxy detects file attachments, each document is submitted
to the swarm via `_submit_corpus()`.  The swarm agents read documents in
batches and discuss them continuously.

### PMFB-SW-02  Immediate synthesis when prompt accompanies attachments
If the user provides both files and a text prompt to the Swarm endpoint,
an immediate synthesis response is returned using document excerpts and
existing swarm knowledge, while the swarm continues background processing.

### PMFB-SW-03  No Graph-RAG for Swarm
The Swarm endpoint uses agent gossiping for document ingestion.  Graph-RAG
is NOT used for the Swarm's document understanding pipeline.

---

## Appendix: LibreChat Attachment Format

LibreChat's `extractFileContext()` produces the following format:

```
Attached document(s):
` ``md
# "filename.txt"
<file content here>

---

# "another-file.pdf"
<more content>
` ``
<user's actual prompt here>
```

(Backticks above are escaped for display; actual format uses triple backticks.)

Key properties:
- The opening marker is always `Attached document(s):\n` followed by
  triple-backtick `md`.
- Multiple files are separated by `\n\n---\n\n` within the block.
- Each file has a header line `# "filename.ext"`.
- The closing triple-backtick is on its own line.
- Everything after the closing fence is the user's text prompt.
