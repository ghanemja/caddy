"""A tiny MCP server over stdio for testing caddy's client (no SolidCraft needed). One JSON-RPC message per line."""
import json
import sys

TOOLS = [{"name": "echo", "description": "say it back", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}},
         {"name": "picture", "description": "an image", "inputSchema": {"type": "object", "properties": {}}},
         {"name": "fail", "description": "an engine failure", "inputSchema": {"type": "object", "properties": {}}},
         {"name": "die", "description": "exit without answering", "inputSchema": {"type": "object", "properties": {}}}]

for line in sys.stdin:
    m = json.loads(line)
    if "id" not in m:
        continue
    method, p = m["method"], m.get("params") or {}
    if method == "initialize":
        r = {"result": {"protocolVersion": p["protocolVersion"], "serverInfo": {"name": "fake"}, "instructions": "hello"}}
    elif method == "tools/list":
        print("not a protocol line", flush=True)                       # servers sometimes log to stdout
        print(json.dumps({"jsonrpc": "2.0", "method": "notifications/progress"}), flush=True)
        r = {"result": {"tools": TOOLS}}
    elif method == "tools/call" and p["name"] == "echo":
        r = {"result": {"content": [{"type": "text", "text": json.dumps(p["arguments"])}]}}
    elif method == "tools/call" and p["name"] == "picture":
        r = {"result": {"content": [{"type": "image", "mimeType": "image/png", "data": "aGk="}, {"type": "text", "text": "{}"}]}}
    elif method == "tools/call" and p["name"] == "fail":
        r = {"result": {"isError": True, "content": [{"type": "text", "text": "boolean Cut: the operation failed"}]}}
    elif method == "tools/call" and p["name"] == "die":
        sys.stderr.write("fatal: out of cheese\n")
        sys.exit(3)
    else:
        r = {"error": {"code": -32602, "message": f"unknown tool {p.get('name')}"}}
    print(json.dumps({"jsonrpc": "2.0", "id": m["id"], **r}), flush=True)
