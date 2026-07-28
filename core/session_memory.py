"""
Session Memory — token-budgeted conversation history with deterministic summarization.

Keeps recent messages intact and summarizes older messages via truncation + ellipsis.
No LLM calls; fully deterministic and fast.
"""

import threading
import time
from typing import Any, Dict, List, Optional, Tuple


class _Session:
    __slots__ = ("messages", "last_accessed")

    def __init__(self) -> None:
        self.messages: List[Dict[str, str]] = []
        self.last_accessed: float = time.monotonic()


class SessionMemory:
    """Thread-safe session memory with token-budgeted summarization."""

    def __init__(
        self,
        max_tokens: int = 4000,
        summary_ratio: float = 0.3,
        ttl_seconds: float = 1800,
        max_messages_per_session: int = 50,
    ) -> None:
        self.max_tokens = max(500, max_tokens)
        self.summary_ratio = max(0.1, min(0.8, summary_ratio))
        self.ttl_seconds = max(60, ttl_seconds)
        self.max_messages_per_session = max(10, max_messages_per_session)
        self._sessions: Dict[str, _Session] = {}
        self._lock = threading.Lock()

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                sess = _Session()
                self._sessions[session_id] = sess
            sess.messages.append({"role": role, "content": content, "ts": time.time()})
            if len(sess.messages) > self.max_messages_per_session:
                sess.messages[: len(sess.messages) - self.max_messages_per_session] = []
            sess.last_accessed = time.monotonic()

    def get_context(self, session_id: str) -> Tuple[str, int]:
        """Return (context_text, token_count) for the session."""
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                return "", 0
            sess.last_accessed = time.monotonic()
            messages = list(sess.messages)

        if not messages:
            return "", 0

        recent, older = self._split_messages(messages)
        context_parts: List[str] = []
        token_count = 0

        if older:
            summary = self._summarize(older)
            context_parts.append(summary)
            token_count += self._count_tokens(summary)

        for msg in recent:
            line = f"[{msg['role']}]: {msg['content']}"
            context_parts.append(line)
            token_count += self._count_tokens(line)

        return "\n\n".join(context_parts), token_count

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def cleanup_expired(self) -> int:
        removed = 0
        with self._lock:
            expired = [
                sid
                for sid, sess in self._sessions.items()
                if (time.monotonic() - sess.last_accessed) > self.ttl_seconds
            ]
            for sid in expired:
                del self._sessions[sid]
                removed += 1
        return removed

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _split_messages(self, messages: List[Dict[str, str]]) -> Tuple[List[Dict], List[Dict]]:
        total = len(messages)
        if total <= 4:
            return messages, []
        keep_recent = max(2, int(total * (1 - self.summary_ratio)))
        older = messages[: total - keep_recent]
        recent = messages[total - keep_recent :]
        return recent, older

    def _summarize(self, messages: List[Dict[str, str]]) -> str:
        if not messages:
            return ""
        total_chars = sum(len(m["content"]) for m in messages)
        budget = max(200, int(total_chars * self.summary_ratio))
        parts: List[str] = []
        accumulated = 0
        for msg in messages:
            content = msg["content"]
            if accumulated + len(content) <= budget:
                parts.append(f"[{msg['role']}]: {content}")
                accumulated += len(content)
            else:
                remaining = budget - accumulated
                if remaining > 40:
                    parts.append(f"[{msg['role']}]: {content[:remaining]}...[summarized]")
                break
        header = f"[Earlier conversation summary ({len(messages)} messages):]\n"
        return header + "\n".join(parts)

    @staticmethod
    def _count_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                return None
            return {
                "message_count": len(sess.messages),
                "total_chars": sum(len(m["content"]) for m in sess.messages),
                "last_accessed_age_sec": time.monotonic() - sess.last_accessed,
            }
