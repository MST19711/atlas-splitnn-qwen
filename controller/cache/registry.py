from __future__ import annotations

import bisect
import threading
import time
from dataclasses import dataclass, field

from controller.cache.snapshot import CacheSnapshot


@dataclass
class CacheKey:
    full_hash: int
    token_seq: tuple[int, ...]


@dataclass
class CacheEntry:
    key: CacheKey
    backend_kind: str
    position: int
    snapshot: CacheSnapshot
    middle_session_id: str | None = None
    created_at: float = field(default_factory=time.time)
    last_access_at: float = field(default_factory=time.time)
    ref_count: int = 0
    sliding_window_offset: int = 0

    def touch(self) -> None:
        self.last_access_at = time.time()


class TrieNode:
    __slots__ = ("children", "entry")

    def __init__(self):
        self.children: dict[int, TrieNode] = {}
        self.entry: CacheEntry | None = None


class PrefixCacheRegistry:
    """LRU + TTL container backed by a prefix trie, with byte-level memory limit."""

    def __init__(
        self,
        *,
        max_entries: int = 8,
        max_cache_bytes: int = 0,
        ttl_sec: float = 300.0,
        min_prefix_len: int = 8,
        tag: str = "default",
        middle_client=None,
    ):
        self.max_entries = max_entries
        self.max_cache_bytes = max_cache_bytes
        self.ttl_sec = ttl_sec
        self.min_prefix_len = min_prefix_len
        self.tag = tag
        self.middle_client = middle_client

        self._trie = TrieNode()
        self._by_hash: dict[int, CacheEntry] = {}
        self._lru_entries: list[CacheEntry] = []
        self._total_bytes: int = 0
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._gc_thread = threading.Thread(target=self._gc_loop, daemon=True)
        self._gc_thread.start()

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def lookup(
        self, token_seq: tuple[int, ...]
    ) -> tuple[CacheEntry | None, int, int]:
        """Walk the trie and return the deepest matching entry.

        Returns (entry, entry_match_len, raw_trie_match_len).
        """
        with self._lock:
            node = self._trie
            last_entry: CacheEntry | None = None
            last_depth = 0
            walked = 0
            for i, tid in enumerate(token_seq):
                child = node.children.get(tid)
                if child is None:
                    walked = i
                    break
                node = child
                walked = i + 1
                if node.entry is not None:
                    last_entry = node.entry
                    last_depth = i + 1
            else:
                walked = len(token_seq)

            if last_entry is not None and last_depth >= self.min_prefix_len:
                return (last_entry, last_depth, walked)
            return (None, 0, walked)

    def acquire(self, entry: CacheEntry) -> CacheEntry:
        """Bump ref_count; copy-on-write when entry already in use."""
        with self._lock:
            if entry.ref_count > 0:
                old_bytes = entry.snapshot.byte_size()
                new_snap = entry.snapshot.copy()
                new_bytes = new_snap.byte_size()
                self._total_bytes -= old_bytes
                self._enforce_byte_limit_locked(new_bytes)
                new_entry = CacheEntry(
                    key=entry.key,
                    backend_kind=entry.backend_kind,
                    position=entry.position,
                    snapshot=new_snap,
                    middle_session_id=entry.middle_session_id,
                    ref_count=1,
                    sliding_window_offset=entry.sliding_window_offset,
                )
                self._by_hash[entry.key.full_hash] = new_entry
                self._trie_insert(entry.key.token_seq, new_entry)
                self._lru_insert(new_entry)
                self._total_bytes += new_bytes
                return new_entry
            entry.ref_count += 1
            entry.touch()
            self._lru_refresh(entry)
            return entry

    def release(self, entry: CacheEntry) -> None:
        """Decrement ref_count and update access time."""
        with self._lock:
            if entry.ref_count > 0:
                entry.ref_count -= 1
            entry.touch()
            self._lru_refresh(entry)

    def save(
        self,
        token_seq: tuple[int, ...],
        snapshot: CacheSnapshot,
        middle_session_id: str | None = None,
        position: int = 0,
    ) -> CacheEntry | None:
        if not token_seq:
            return None
        full_hash = self._hash_seq(token_seq)
        snap_bytes = snapshot.byte_size()
        with self._lock:
            existing = self._by_hash.get(full_hash)
            if existing is not None:
                delta = snap_bytes - existing.snapshot.byte_size()
                existing.snapshot = snapshot
                existing.middle_session_id = middle_session_id
                existing.position = position
                existing.touch()
                self._lru_refresh(existing)
                self._total_bytes += delta
                self._enforce_byte_limit_locked()
                return existing
            if len(self._by_hash) >= self.max_entries:
                self._evict_one_lru_locked()
            self._enforce_byte_limit_locked(snap_bytes)
            key = CacheKey(full_hash=full_hash, token_seq=token_seq)
            entry = CacheEntry(
                key=key,
                backend_kind=self.tag,
                position=position,
                snapshot=snapshot,
                middle_session_id=middle_session_id,
            )
            self._by_hash[full_hash] = entry
            self._trie_insert(token_seq, entry)
            self._lru_insert(entry)
            self._total_bytes += snap_bytes
            return entry

    def evict_one(self, entry: CacheEntry) -> None:
        with self._lock:
            self._evict_locked(entry)

    def stop(self) -> None:
        self._stop.set()
        self._gc_thread.join(timeout=1)

    def stats(self) -> dict:
        with self._lock:
            return {
                "tag": self.tag,
                "entries": len(self._by_hash),
                "max_entries": self.max_entries,
                "max_cache_bytes": self.max_cache_bytes,
                "current_cache_bytes": self._total_bytes,
                "ttl_sec": self.ttl_sec,
                "min_prefix_len": self.min_prefix_len,
                "total_snapshot_bytes": sum(
                    e.snapshot.byte_size() for e in self._by_hash.values()
                ),
            }

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_seq(token_seq: tuple[int, ...]) -> int:
        h = 5381
        for t in token_seq:
            h = ((h << 5) + h) + t  # djb2
        return h & 0xFFFFFFFFFFFFFFFF

    def _trie_insert(self, token_seq: tuple[int, ...], entry: CacheEntry) -> None:
        node = self._trie
        for tid in token_seq:
            node = node.children.setdefault(tid, TrieNode())
        node.entry = entry

    def _trie_remove(self, token_seq: tuple[int, ...]) -> None:
        node = self._trie
        path: list[tuple[int, TrieNode]] = []
        for tid in token_seq:
            child = node.children.get(tid)
            if child is None:
                return
            path.append((tid, child))
            node = child
        if node.entry is not None:
            node.entry = None
        for tid, child in reversed(path):
            if not child.children and child.entry is None:
                parent = self._trie
                for pid, _ in path[:-1]:
                    parent = parent.children[pid]
                parent.children.pop(tid, None)

    def _lru_insert(self, entry: CacheEntry) -> None:
        idx = bisect.bisect_left(
            self._lru_entries, entry.last_access_at, key=lambda e: e.last_access_at
        )
        self._lru_entries.insert(idx, entry)

    def _lru_refresh(self, entry: CacheEntry) -> None:
        try:
            self._lru_entries.remove(entry)
        except ValueError:
            pass
        self._lru_insert(entry)

    def _evict_one_lru_locked(self) -> None:
        for e in self._lru_entries:
            if e.ref_count == 0:
                self._evict_locked(e)
                return
        fallback = None
        for e in self._lru_entries:
            if e.ref_count == 0:
                fallback = e
                break
        if fallback is not None:
            self._evict_locked(fallback)

    def _evict_locked(self, entry: CacheEntry) -> None:
        try:
            self._lru_entries.remove(entry)
        except ValueError:
            pass
        self._by_hash.pop(entry.key.full_hash, None)
        self._trie_remove(entry.key.token_seq)
        self._total_bytes -= entry.snapshot.byte_size()
        if entry.middle_session_id and self.middle_client:
            try:
                self.middle_client.close(entry.middle_session_id, evict=True)
            except Exception:
                pass

    def _enforce_byte_limit_locked(self, incoming_bytes: int = 0) -> None:
        if self.max_cache_bytes <= 0:
            return
        while self._total_bytes + incoming_bytes > self.max_cache_bytes and self._lru_entries:
            evicted = False
            for e in list(self._lru_entries):
                if e.ref_count == 0:
                    self._evict_locked(e)
                    evicted = True
                    break
            if not evicted:
                break

    def _gc_loop(self) -> None:
        while not self._stop.wait(30.0):
            now = time.time()
            with self._lock:
                expired: list[CacheEntry] = []
                for e in list(self._lru_entries):
                    if e.last_access_at + self.ttl_sec < now and e.ref_count == 0:
                        expired.append(e)
                for e in expired:
                    self._evict_locked(e)
