"""SQLite persistence for captured LLM turns with WAL mode support."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import aiosqlite

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS captured_turns (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id           TEXT NOT NULL UNIQUE,
    created_at_ms     INTEGER NOT NULL,

    source_agent      TEXT,
    teacher_model     TEXT,

    request_json      TEXT NOT NULL,
    response_json     TEXT NOT NULL,

    latency_ms        INTEGER,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    status            TEXT NOT NULL DEFAULT 'success',
    error_message     TEXT,

    processed         INTEGER NOT NULL DEFAULT 0,
    processed_at_ms   INTEGER,

    metadata_json     TEXT
);

CREATE INDEX IF NOT EXISTS idx_captured_created
    ON captured_turns(created_at_ms);
CREATE INDEX IF NOT EXISTS idx_captured_processed
    ON captured_turns(processed, created_at_ms);
"""


class CaptureDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._connection: aiosqlite.Connection | None = None

    async def connect(self) -> "CaptureDatabase":
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(self.db_path)
        # Enable WAL mode and normal synchronous for zero writer contention and high throughput
        await self._connection.execute("PRAGMA journal_mode = WAL;")
        await self._connection.execute("PRAGMA synchronous = NORMAL;")
        await self._connection.executescript(SCHEMA_SQL)
        await self._connection.commit()
        return self

    async def close(self) -> None:
        if self._connection:
            await self._connection.close()
            self._connection = None

    async def insert_captures_batch(self, records: list[dict[str, Any]]) -> int:
        """Batch insert records inside a single transaction to eliminate SQLite overhead."""
        if not self._connection or not records:
            return 0

        rows = []
        now_ms = int(time.time() * 1000)
        for r in records:
            created_at = r.get("created_at_ms") or now_ms
            metadata = r.get("metadata")
            metadata_str = json.dumps(metadata, ensure_ascii=False) if metadata else None
            req_json = r["request_json"] if isinstance(r["request_json"], str) else json.dumps(r["request_json"], ensure_ascii=False)
            res_json = r["response_json"] if isinstance(r["response_json"], str) else json.dumps(r["response_json"], ensure_ascii=False)

            rows.append(
                (
                    r["turn_id"],
                    created_at,
                    r.get("source_agent"),
                    r.get("teacher_model"),
                    req_json,
                    res_json,
                    r.get("latency_ms"),
                    r.get("prompt_tokens"),
                    r.get("completion_tokens"),
                    r.get("status", "success"),
                    r.get("error_message"),
                    metadata_str,
                )
            )

        try:
            await self._connection.executemany(
                """
                INSERT OR IGNORE INTO captured_turns (
                    turn_id, created_at_ms, source_agent, teacher_model,
                    request_json, response_json, latency_ms, prompt_tokens,
                    completion_tokens, status, error_message, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            await self._connection.commit()
            return len(rows)
        except Exception as e:
            logger.exception("Failed batch insert into captured_turns: %s", e)
            return 0
