"""Egress allowlist proxy for laptop runs: the local stand-in for Modal's sandbox cidr_allowlist.

Rollout containers sit on an internal Docker network with no route out; this proxy is attached to that
network and to the default bridge, and tunnels HTTPS CONNECT requests to the allowlisted hosts only
(OpenRouter). Everything else, plain HTTP included, gets 403. The containers reach it through
HTTPS_PROXY, which the OpenAI client honours, so the agent's shell can reach OpenRouter and nothing else,
as on Modal. Logs one line per connection: host, allowed or refused. Standard library only.
  python3 egress_proxy.py [port]     (default 8888; ALLOW_HOSTS="host:port,..." overrides the allowlist)
"""
import asyncio
import os
import sys
import time

ALLOW = set(os.environ.get("ALLOW_HOSTS", "openrouter.ai:443").split(","))
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8888


def log(msg):
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


async def pipe(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        writer.close()


async def handle(reader, writer):
    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 30)
        method, target = head.split(b"\r\n", 1)[0].decode().split(" ")[:2]
        if method != "CONNECT" or target not in ALLOW:
            log(f"refused {method} {target}")
            writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
            await writer.drain()
            writer.close()
            return
        host, port = target.rsplit(":", 1)
        up_r, up_w = await asyncio.wait_for(asyncio.open_connection(host, int(port)), 30)
        writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
        await writer.drain()
        log(f"allowed {target}")
        await asyncio.gather(pipe(reader, up_w), pipe(up_r, writer))
    except Exception as e:  # a bad client or a dead upstream ends only this connection
        log(f"error {type(e).__name__}: {e}")
        writer.close()


async def main():
    server = await asyncio.start_server(handle, "0.0.0.0", PORT)
    log(f"egress proxy on :{PORT}, allowing {sorted(ALLOW)}")
    async with server:
        await server.serve_forever()


asyncio.run(main())
