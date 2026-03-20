# Testing Deep Search Portal on Staging VM

## Overview
The deep-search-portal runs on a Vast.ai GPU instance. The staging VM hosts all services: Open WebUI, persistent research proxy, knowledge engine (Neo4j), SearXNG, Veritas, and Cloudflare tunnel.

## Devin Secrets Needed
- `VAST_AI_API_KEY` — Vast.ai API key to manage instances

## VM Access
- SSH: `ssh -o StrictHostKeyChecking=no -p <port> root@<ssh_host>`
- Get SSH host/port from: `curl -s -H "Authorization: Bearer $VAST_AI_API_KEY" https://console.vast.ai/api/v0/instances/`
- The staging VM instance ID may change; check the API for current instances
- If the VM is stopped, start it with: `curl -X PUT -H "Authorization: Bearer $VAST_AI_API_KEY" -H "Content-Type: application/json" -d '{"state": "running"}' https://console.vast.ai/api/v0/instances/<id>/`
- If `resources_unavailable` is returned, check the account balance — a negative balance prevents starting VMs

## Services & Ports
| Service | Port | Screen Name |
|---------|------|-------------|
| Open WebUI | 3000 | owui |
| Thinking Proxy | 9100 | thinking-proxy |
| Deep Research Proxy | 9200 | deep-research |
| Persistent Proxy | 9300 | persistent-proxy |
| Knowledge Engine | 9400 | knowledge-engine |
| Veritas | 9500 | veritas |
| SearXNG | 8888 | searxng |
| Neo4j Bolt | 7687 | (system service) |

## Deployment Steps
1. Pull latest code locally: `git checkout main && git pull origin main`
2. SCP updated files to the VM:
   ```
   scp -P <port> proxies/persistent_deep_research_proxy.py root@<host>:/opt/
   scp -P <port> services/knowledge-engine/knowledge_engine/ontology.py root@<host>:/opt/services/knowledge-engine/knowledge_engine/
   scp -P <port> proxies/veritas_inquisitor.py root@<host>:/opt/
   ```
3. Restart affected services via screen:
   - Kill old screen: `screen -S persistent-proxy -X quit`
   - Start new screen with env vars (see `/opt/start_production.sh` for the full command)
4. Verify services are running: `curl -s -o /dev/null -w '%{http_code}' http://localhost:9300/v1/models`

## Testing a Research Query
1. Clear logs: `truncate -s 0 /var/log/persistent-proxy.log`
2. Send query:
   ```
   curl -s -N http://localhost:9300/v1/chat/completions \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer test" \
     -d '{"model":"persistent-miroflow","messages":[{"role":"user","content":"YOUR QUERY"}],"stream":true}' \
     --max-time 600 > /tmp/research_output.txt 2>&1 &
   ```
3. Monitor logs: `strings /var/log/persistent-proxy.log | grep -E 'Starting subagent|Subagent complete'`
4. Use `strings` instead of `grep` on log files — they may contain binary null bytes from truncation

## Key Verification Points
- **Subagents produce conditions**: Look for `Subagent complete: X conditions` where X > 0
- **Parallel execution**: Multiple `Starting subagent` lines within 1 second of each other
- **No 422 errors**: `strings logfile | grep -iE 'error.*422|422.*error'` should return nothing (timestamps may contain "422" — check for actual HTTP 422 errors)
- **No moderation errors**: `strings logfile | grep 'Invalid model'` should return nothing
- **Tree branching**: Look for `depth=1` and `depth=2` in Starting subagent lines

## Known Issues
- **Mistral rate limiting (429)**: Under heavy parallel load (10+ concurrent subagents), the Mistral API returns 429 rate limit errors. The system retries automatically but this can cause the research to take much longer and may cause curl clients to disconnect
- **Cloudflare tunnel routing**: `staging.deep-search.uk` routes to port 3001 (configured in Cloudflare dashboard), while `deep-search.uk` routes to port 3000. If staging shows a 502, check whether Open WebUI is running on the expected port
- **Cloudflare tunnel token**: The tunnel token is stored in `/opt/start_production.sh`. If the tunnel dies, restart with: `screen -dmS cftunnel bash -c 'cloudflared tunnel run --token <TOKEN>'`
- **Log binary content**: After `truncate -s 0`, log files may have null bytes. Always use `strings` to filter before `grep`
- **Langfuse disabled**: Langfuse tracing is disabled unless `LANGFUSE_*` env vars are set. The Langfuse link won't appear in output without these
- **Report link**: The B2 report link only appears after the full pipeline completes (all subagents done + synthesis + B2 upload). If the client disconnects early, no report link is generated

## Startup Script
The full production startup script is at `/opt/start_production.sh`. It starts all services in screen sessions with the correct environment variables. Run it after a fresh VM boot if services aren't auto-started.
