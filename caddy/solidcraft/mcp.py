"""A small stdio client for SolidCraft's MCP server (`solidcraft-cli mcp`).

JSON-RPC 2.0, one message per line. caddy starts the server headless with one folder it may write to, so the design
session lives in that process and every file it touches stays in the run's folder.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field

PROTOCOL = "2025-06-18"


class McpError(RuntimeError):
    """The server is missing, exited, timed out, or answered with a protocol error."""


def find_cli() -> str | None:
    """`SOLIDCRAFT_CLI`, else `solidcraft-cli` (or the snap's `solidcraft.cli`) on PATH."""
    env = os.environ.get("SOLIDCRAFT_CLI")
    if env:
        return env if (os.path.isfile(env) or shutil.which(env)) else None
    return shutil.which("solidcraft-cli") or shutil.which("solidcraft.cli")


@dataclass
class ToolResult:
    text: str
    images: list[tuple[str, str]] = field(default_factory=list)   # (mime type, base64)
    is_error: bool = False

    def json(self):
        """The reply as JSON (SolidCraft's tools answer in JSON text); None when it isn't."""
        try:
            return json.loads(self.text)
        except ValueError:
            return None


class Mcp:
    def __init__(self, argv: list[str], timeout: float = 330.0, env: dict | None = None):
        # 330 s: one SolidCraft call may block for 300 s before it reports "still running" (docs/mcp.md)
        self.argv, self.timeout = argv, timeout
        self._err = tempfile.TemporaryFile()
        try:
            self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err,
                                      text=True, bufsize=1, env=env)
        except OSError as e:
            raise McpError(f"could not start {argv[0]!r}: {e}") from None
        self._q: queue.Queue = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        self._id = 0
        self._lock = threading.Lock()
        init = self.request("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                           "clientInfo": {"name": "caddy", "version": "1"}})
        self.server = init.get("serverInfo", {})
        self.instructions = init.get("instructions", "")
        self.notify("notifications/initialized")

    def _read(self) -> None:
        for line in self.p.stdout:
            line = line.strip()
            if line:
                self._q.put(line)
        self._q.put(None)

    def stderr(self) -> str:
        self._err.seek(0)
        return self._err.read().decode(errors="replace")[-2000:]

    def notify(self, method: str, params: dict | None = None) -> None:
        msg = {"jsonrpc": "2.0", "method": method, **({"params": params} if params is not None else {})}
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()

    def request(self, method: str, params: dict | None = None) -> dict:
        with self._lock:
            self._id += 1
            msg = {"jsonrpc": "2.0", "id": self._id, "method": method, **({"params": params} if params is not None else {})}
            try:
                self.p.stdin.write(json.dumps(msg) + "\n")
                self.p.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                raise McpError(f"the MCP server exited: {self.stderr().strip() or 'no output'}") from None
            while True:
                try:
                    line = self._q.get(timeout=self.timeout)
                except queue.Empty:
                    raise McpError(f"no answer to {method} within {self.timeout:g} s") from None
                if line is None:
                    raise McpError(f"the MCP server exited: {self.stderr().strip() or 'no output'}")
                try:
                    r = json.loads(line)
                except ValueError:
                    continue                      # not a protocol line
                if r.get("id") != self._id:
                    continue                      # a notification, or an answer nobody waits for
                if "error" in r:
                    e = r["error"]
                    raise McpError(f"{method}: {e.get('message', e)}")
                return r.get("result", {})

    def tools(self) -> list[dict]:
        return self.request("tools/list").get("tools", [])

    def call(self, name: str, arguments: dict | None = None) -> ToolResult:
        r = self.request("tools/call", {"name": name, "arguments": arguments or {}})
        text, images = [], []
        for c in r.get("content", []):
            if c.get("type") == "text":
                text.append(c.get("text", ""))
            elif c.get("type") == "image":
                images.append((c.get("mimeType", "image/png"), c.get("data", "")))
        return ToolResult("\n".join(text), images, bool(r.get("isError")))

    def close(self) -> None:
        try:
            self.p.stdin.close()
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()
        self._err.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
