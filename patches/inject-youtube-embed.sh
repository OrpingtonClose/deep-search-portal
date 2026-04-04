#!/bin/bash
# Injects the YouTube embed transformer script into LibreChat's index.html.
# Idempotent — skips if already injected.
#
# Usage:  bash patches/inject-youtube-embed.sh /opt/LibreChat

set -euo pipefail

LIBRECHAT_DIR="${1:-/opt/LibreChat}"
INDEX_HTML="$LIBRECHAT_DIR/client/dist/index.html"
SCRIPT_SRC="$(dirname "$0")/youtube-embed.js"

if [ ! -f "$INDEX_HTML" ]; then
  echo "ERROR: $INDEX_HTML not found" >&2
  exit 1
fi

if [ ! -f "$SCRIPT_SRC" ]; then
  echo "ERROR: $SCRIPT_SRC not found" >&2
  exit 1
fi

# Check if already injected
if grep -q 'yt-embed-wrapper' "$INDEX_HTML"; then
  echo "YouTube embed script already injected — skipping."
  exit 0
fi

# Inject before </body> using python to avoid sed escaping issues
python3 -c "
import sys
js = open('$SCRIPT_SRC').read()
html = open('$INDEX_HTML').read()
tag = '<script>' + js + '</script>\n</body>'
html = html.replace('</body>', tag, 1)
open('$INDEX_HTML', 'w').write(html)
"

echo "YouTube embed script injected into $INDEX_HTML"
