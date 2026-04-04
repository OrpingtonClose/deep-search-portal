#!/bin/bash
# Injects the YouTube embed transformer script into LibreChat's index.html.
# Idempotent — skips if already injected with the same version.
# Re-injects automatically when the script content changes.
#
# Usage:  bash patches/inject-youtube-embed.sh /opt/LibreChat
#
# IMPORTANT: After running this script you MUST restart LibreChat — it caches
# index.html in memory at startup.  Users may also need to unregister the
# service worker and hard-refresh (Ctrl+Shift+R) to see the update.

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

# Strip any previous injection so we can re-inject the latest version
python3 -c "
import re
html = open('$INDEX_HTML').read()
html = re.sub(r'<script>\s*/\*\*\s*\*\s*YouTube Embed Transformer.*?</script>\s*', '', html, flags=re.DOTALL)
open('$INDEX_HTML', 'w').write(html)
"

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
echo "NOTE: Restart LibreChat for changes to take effect."
