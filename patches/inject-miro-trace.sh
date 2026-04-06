#!/bin/bash
# Injects the Miro execution-trace & knowledge-corpus scripts into LibreChat.
# Idempotent — strips any previous injection before re-injecting.
#
# What it does:
#   1. Copies trace-page.html and knowledge-page.html into LibreChat's
#      client/dist/ so they're served as static files.
#   2. Injects miro-trace.js into index.html (before </body>).
#
# Usage:  bash patches/inject-miro-trace.sh /opt/LibreChat
#
# IMPORTANT: After running this script you MUST restart LibreChat — it caches
# index.html in memory at startup.

set -euo pipefail

LIBRECHAT_DIR="${1:-/opt/LibreChat}"
INDEX_HTML="$LIBRECHAT_DIR/client/dist/index.html"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

TRACE_JS="$SCRIPT_DIR/miro-trace.js"
TRACE_PAGE="$SCRIPT_DIR/trace-page.html"
KNOWLEDGE_PAGE="$SCRIPT_DIR/knowledge-page.html"

# ── Validate inputs ─────────────────────────────────────────────────────
for f in "$INDEX_HTML" "$TRACE_JS" "$TRACE_PAGE" "$KNOWLEDGE_PAGE"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: $f not found" >&2
    exit 1
  fi
done

# ── Copy HTML pages into dist/ ──────────────────────────────────────────
cp "$TRACE_PAGE" "$LIBRECHAT_DIR/client/dist/trace.html"
cp "$KNOWLEDGE_PAGE" "$LIBRECHAT_DIR/client/dist/knowledge.html"
echo "Copied trace.html and knowledge.html into $LIBRECHAT_DIR/client/dist/"

# ── Strip any previous miro-trace injection ─────────────────────────────
python3 -c "
import re
html = open('$INDEX_HTML').read()
html = re.sub(r'<script>\s*/\*\*\s*\*\s*Miro Execution Trace.*?</script>\s*', '', html, flags=re.DOTALL)
open('$INDEX_HTML', 'w').write(html)
"

# ── Inject miro-trace.js before </body> ─────────────────────────────────
python3 -c "
import sys
js = open('$TRACE_JS').read()
html = open('$INDEX_HTML').read()
tag = '<script>' + js + '</script>\n</body>'
html = html.replace('</body>', tag, 1)
open('$INDEX_HTML', 'w').write(html)
"

echo "Miro trace script injected into $INDEX_HTML"
echo "NOTE: Restart LibreChat for changes to take effect."
