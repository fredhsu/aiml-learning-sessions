# Hermes programmatic integration recommendation

**Question:** how should this session communicate with a locally running Hermes agent?

## Recommendation: Hermes API server over localhost HTTP

Use the **OpenAI-compatible API server** exposed by `hermes gateway`, not the ACP process or an interactive TUI.

Why:

- This session can invoke local commands and HTTP clients, but has no ACP client and cannot attach to another process's TUI terminal.
- The API server is explicitly intended for language-agnostic HTTP clients and automation. It offers `POST /v1/responses` for server-side, stateful conversations, plus health/capability endpoints. A named `conversation` can retain a Hermes conversation without the client managing response IDs.
- ACP is JSON-RPC over **stdio** for IDEs that speak the Agent Client Protocol. The TUI gateway is JSON-RPC over stdio/WebSocket for a custom host that needs Hermes-specific UI concerns such as slash commands and approval flows. Both require writing or operating a protocol client; neither is needed for the present shell/HTTP bridge.

## Minimal local setup

In `~/.hermes/.env`, set a nontrivial local bearer key:

```dotenv
API_SERVER_ENABLED=true
API_SERVER_KEY=<generate-a-long-random-local-secret>
```

Start the gateway (foreground is appropriate for the initial test):

```bash
hermes gateway run
```

The documented default listener is `http://127.0.0.1:8642`. Do not configure a public bind or CORS origin unless a browser client genuinely needs it.

## Verify without revealing the key

```bash
curl -fsS http://127.0.0.1:8642/health

set -a
source ~/.hermes/.env
set +a
curl -fsS \
  -H "Authorization: Bearer $API_SERVER_KEY" \
  http://127.0.0.1:8642/v1/capabilities
```

At research time, port 8642 was not listening; `hermes gateway` was available locally but not running.

## First stateful request

Use a dedicated named conversation so Hermes does not mix this coordination with its TUI or ACP sessions:

```bash
set -a
source ~/.hermes/.env
set +a
curl -fsS http://127.0.0.1:8642/v1/responses \
  -H "Authorization: Bearer $API_SERVER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hermes-agent",
    "conversation": "curriculum-bridge",
    "input": "You are the local Hermes collaborator. Confirm you can read the current working directory and report your available tools.",
    "store": true
  }'
```

Once this endpoint is running, this agent can make requests through a shell command without needing the secret pasted into chat, provided the command sources the local environment file.

## Operating boundary

Start with deliberate request/response handoffs, not autonomous back-and-forth loops. Hermes may have its own tools, permissions, working directory, and persistent session state. Each handoff should specify task, repository path, expected artifact, and verification command. Keep approval prompts enabled until the bridge behaviour is trusted.

## Sources

- Hermes, [Programmatic Integration](https://hermes-agent.nousresearch.com/docs/developer-guide/programmatic-integration) — protocol selection; ACP and TUI-gateway transports; API-server endpoints; named API use case.
- Hermes, [API Server](https://hermes-agent.nousresearch.com/docs/user-guide/features/api-server) — `.env` enablement, default localhost endpoint/port, bearer authentication, `POST /v1/responses`, and named conversations.
- Hermes, [Agent Loop Internals](https://hermes-agent.nousresearch.com/docs/developer-guide/agent-loop) — in-process `AIAgent` option and persistence/tool-execution context.
