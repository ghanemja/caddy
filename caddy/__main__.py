"""python -m caddy            → the web app on http://127.0.0.1:8000
   python -m caddy --report   → build every catalogue size with the reference graph and print the verdicts
   python -m caddy --digest   → same, with the full digest per size"""
import sys

if __name__ == "__main__":
    if "--report" in sys.argv or "--digest" in sys.argv:
        from caddy.kernel import graph
        from caddy.kernel.family import build_family
        for r in build_family(graph.load()):
            if "--digest" in sys.argv:
                print(r.digest(), end="\n\n")
            else:
                n = sum(c["ok"] for c in r.checks)
                print(f"{r.size:10} {'✓' if r.ok else '✗'} checks {n}/{len(r.checks)}"
                      + "".join(f"  ✗ {c['id']}" for c in r.checks if not c["ok"]))
    else:
        from caddy.agent.loop import load_env
        from caddy.app.server import main
        load_env()
        port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8000
        host = sys.argv[sys.argv.index("--host") + 1] if "--host" in sys.argv else "127.0.0.1"
        main(host=host, port=port)
