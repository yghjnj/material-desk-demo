"""Session-isolated public server for the Material Desk demo.

This process is intended to run behind a managed HTTPS reverse proxy. Every
browser session gets its own local knowledge-base directory; no endpoint can
list or query another session's documents.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from http.cookies import SimpleCookie
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
from threading import RLock
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "demo"
if str(PROJECT_ROOT / "src") not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT / "src"))

from local_kb import LocalKnowledgeBase, MAX_UPLOAD_BYTES, UploadError  # noqa: E402


SESSION_COOKIE = "material_desk_session"
SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{32,128}$")


class RateLimiter:
    def __init__(self, limit: int = 120, window_seconds: int = 60) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = RLock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            events = self._events[key]
            while events and events[0] <= now - self.window_seconds:
                events.popleft()
            if len(events) >= self.limit:
                return False
            events.append(now)
            return True


class SessionStore:
    def __init__(self, root: str | Path, retention_hours: int = 24) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.retention_seconds = max(1, retention_hours) * 3600
        self._sessions: dict[str, tuple[LocalKnowledgeBase, float]] = {}
        self._lock = RLock()

    @staticmethod
    def _valid(session_id: str | None) -> bool:
        return bool(session_id and SESSION_RE.fullmatch(session_id))

    def _session_path(self, session_id: str) -> Path:
        digest = hashlib.sha256(session_id.encode("ascii")).hexdigest()[:32]
        return self.root / digest

    def _new_id(self) -> str:
        return secrets.token_urlsafe(32)

    def acquire(self, requested_id: str | None) -> tuple[str, LocalKnowledgeBase, bool]:
        now = time.monotonic()
        with self._lock:
            self._sweep_locked(now)
            session_id = requested_id if self._valid(requested_id) else self._new_id()
            entry = self._sessions.get(session_id)
            if entry is not None:
                self._sessions[session_id] = (entry[0], now)
                return session_id, entry[0], False
            kb = LocalKnowledgeBase(self._session_path(session_id))
            self._sessions[session_id] = (kb, now)
            return session_id, kb, requested_id != session_id

    def close(self, session_id: str) -> None:
        with self._lock:
            entry = self._sessions.pop(session_id, None)
            if entry is not None:
                entry[0].close()
            path = self._session_path(session_id)
            if path.is_dir() and path.parent == self.root:
                shutil.rmtree(path, ignore_errors=True)

    def _sweep_locked(self, now: float) -> None:
        stale = [sid for sid, (_, touched) in self._sessions.items()
                 if touched <= now - self.retention_seconds]
        for sid in stale:
            entry = self._sessions.pop(sid)
            entry[0].close()
            path = self._session_path(sid)
            if path.is_dir() and path.parent == self.root:
                shutil.rmtree(path, ignore_errors=True)

    def close_all(self) -> None:
        with self._lock:
            for kb, _ in self._sessions.values():
                kb.close()
            self._sessions.clear()


class Handler(SimpleHTTPRequestHandler):
    server_version = "MaterialDeskPublic/1.0"

    def __init__(self, *args, store: SessionStore, limiter: RateLimiter, **kwargs):
        self.store = store
        self.limiter = limiter
        self.session_id: str | None = None
        self.kb: LocalKnowledgeBase | None = None
        self.new_session = False
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt, *args):
        # Do not write uploaded content, questions, or document names to logs.
        return

    def _cookie_session(self) -> str | None:
        cookie = SimpleCookie(self.headers.get("Cookie", ""))
        morsel = cookie.get(SESSION_COOKIE)
        return morsel.value if morsel else None

    def _ensure_session(self) -> LocalKnowledgeBase:
        if self.kb is None:
            self.session_id, self.kb, self.new_session = self.store.acquire(self._cookie_session())
        return self.kb

    def _json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        self._cors_headers()
        if self.new_session and self.session_id:
            secure = os.getenv("PUBLIC_SECURE_COOKIE", "1") == "1"
            same_site = "None" if os.getenv("PUBLIC_CORS_ORIGIN") else "Lax"
            cookie = f"{SESSION_COOKIE}={self.session_id}; Path=/; HttpOnly; SameSite={same_site}; Max-Age={self.store.retention_seconds}"
            if secure:
                cookie += "; Secure"
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(body)

    def _cors_headers(self) -> None:
        allowed = os.getenv("PUBLIC_CORS_ORIGIN", "").strip()
        origin = self.headers.get("Origin", "")
        if allowed and origin == allowed:
            self.send_header("Access-Control-Allow-Origin", allowed)
            self.send_header("Access-Control-Allow-Credentials", "true")
            self.send_header("Vary", "Origin")

    def _authorized(self) -> bool:
        expected = os.getenv("PUBLIC_ACCESS_TOKEN", "").strip()
        if not expected:
            return True
        supplied = self.headers.get("X-Material-Desk-Token", "")
        if not supplied:
            supplied = self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        return secrets.compare_digest(supplied, expected)

    def _guard(self) -> bool:
        if not self.limiter.allow(self.client_address[0]):
            self._json(429, {"error": "RATE_LIMITED"})
            return False
        if not self._authorized():
            self._json(401, {"error": "UNAUTHORIZED"})
            return False
        return True

    def _body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise UploadError("INVALID_CONTENT_LENGTH") from exc
        if length <= 0:
            raise UploadError("EMPTY_REQUEST")
        if length > MAX_UPLOAD_BYTES:
            raise UploadError("FILE_TOO_LARGE")
        return self.rfile.read(length)

    def do_OPTIONS(self):  # noqa: N802
        if not self._guard():
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Material-Desk-Token, Authorization")
        self.send_header("Access-Control-Max-Age", "600")
        self._cors_headers()
        self.end_headers()

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/api/") and not self._guard():
            return
        path = urlparse(self.path).path
        if path == "/api/health":
            kb = self._ensure_session()
            return self._json(200, {
                "status": "ok", "mode": "public", "documents": len(kb.documents()),
                "storage": "session-isolated", "retention_hours": self.store.retention_seconds // 3600,
            })
        if path == "/api/documents":
            return self._json(200, {"documents": self._ensure_session().documents()})
        return super().do_GET()

    def do_POST(self):  # noqa: N802
        if not self._guard():
            return
        parsed = urlparse(self.path)
        try:
            kb = self._ensure_session()
            if parsed.path == "/api/documents":
                query = parse_qs(parsed.query)
                filename = unquote(query.get("filename", [""])[0]) or self.headers.get("X-Filename", "")
                result = kb.ingest(filename, self._body())
                return self._json(201, {"document": result})
            if parsed.path == "/api/query":
                payload = json.loads(self._body().decode("utf-8"))
                return self._json(200, kb.query(str(payload.get("question", ""))))
            if parsed.path == "/api/requirements":
                payload = json.loads(self._body().decode("utf-8"))
                return self._json(200, LocalKnowledgeBase.extract_requirements(str(payload.get("text", ""))))
            return self._json(404, {"error": "NOT_FOUND"})
        except UploadError as exc:
            return self._json(400, {"error": str(exc)})
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._json(400, {"error": "INVALID_JSON"})
        except (OSError, ValueError, RuntimeError) as exc:
            return self._json(422, {"error": str(exc) or "REQUEST_FAILED"})

    def do_DELETE(self):  # noqa: N802
        if not self._guard():
            return
        if urlparse(self.path).path != "/api/session":
            return self._json(404, {"error": "NOT_FOUND"})
        session_id = self._ensure_session() and self.session_id
        if session_id:
            self.store.close(session_id)
        self.send_response(204)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Set-Cookie", f"{SESSION_COOKIE}=; Path=/; Max-Age=0; HttpOnly; SameSite=Lax")
        self._cors_headers()
        self.end_headers()


class PublicServer(HTTPServer):
    allow_reuse_address = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Material Desk session-isolated public server")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--storage", default=os.getenv("PUBLIC_STORAGE", str(PROJECT_ROOT / "work" / "public_kb")))
    parser.add_argument("--retention-hours", type=int, default=int(os.getenv("PUBLIC_RETENTION_HOURS", "24")))
    args = parser.parse_args()
    store = SessionStore(args.storage, args.retention_hours)
    limiter = RateLimiter()
    server = PublicServer((args.host, args.port), lambda *a, **kw: Handler(*a, store=store, limiter=limiter, **kw))
    print(f"http://{args.host}:{args.port}", flush=True)
    print(f"session storage: {Path(args.storage).resolve()}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        store.close_all()


if __name__ == "__main__":
    main()
