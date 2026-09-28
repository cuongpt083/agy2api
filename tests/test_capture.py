import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import aiosqlite
from app.core.capture_db import CaptureDatabase
from app.core.capture import (
    CaptureManager,
    extract_reasoning_from_text,
    extract_thinking_from_brain_async,
    parse_usage_and_reasoning_from_chunks,
)

# Test compatibility with capture-proxy's message_builder
CAPTURE_PROXY_PATH = r"C:\Users\Admin\Workspaces\proxy-for-bot\capture-proxy"
if os.path.isdir(CAPTURE_PROXY_PATH) and CAPTURE_PROXY_PATH not in sys.path:
    sys.path.insert(0, CAPTURE_PROXY_PATH)

try:
    from app.message_builder import response_to_assistant_turn
except ImportError:
    response_to_assistant_turn = None


class TestCaptureHelpers(unittest.TestCase):
    def test_extract_reasoning_from_think_tags(self):
        content = "<think>Let me calculate 2+2.\nIt is 4.</think>The answer is 4."
        reasoning, cleaned = extract_reasoning_from_text(content)
        self.assertEqual(reasoning, "Let me calculate 2+2.\nIt is 4.")
        self.assertEqual(cleaned, "The answer is 4.")

    def test_extract_reasoning_none_when_no_tags(self):
        content = "Regular answer without thinking."
        reasoning, cleaned = extract_reasoning_from_text(content)
        self.assertIsNone(reasoning)
        self.assertEqual(cleaned, content)

    def test_parse_usage_and_reasoning_from_sse_chunks(self):
        chunks = [
            {
                "choices": [
                    {
                        "delta": {"role": "assistant", "reasoning_content": "Step 1: check. "}
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {"reasoning_content": "Step 2: done."}
                    }
                ]
            },
            {
                "choices": [{"delta": {"content": "Final result"}}],
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 8,
                    "prompt_tokens_details": {"cached_tokens": 5},
                },
            },
        ]
        p, c, cached, reasoning = parse_usage_and_reasoning_from_chunks(chunks)
        self.assertEqual(p, 15)
        self.assertEqual(c, 8)
        self.assertEqual(cached, 5)
        self.assertEqual(reasoning, "Step 1: check. Step 2: done.")


class TestCaptureDatabase(unittest.IsolatedAsyncioTestCase):
    async def test_batch_insert(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_capture.db")
            db = CaptureDatabase(db_path)
            await db.connect()

            records = [
                {
                    "turn_id": "turn-1",
                    "created_at_ms": 1000,
                    "source_agent": "nanobot",
                    "teacher_model": "gemini-3.7-flash",
                    "request_json": {"messages": [{"role": "user", "content": "hi"}]},
                    "response_json": {"choices": [{"message": {"content": "hello"}}]},
                    "latency_ms": 150,
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "status": "success",
                    "metadata": {"stream": False},
                },
                {
                    "turn_id": "turn-2",
                    "created_at_ms": 1005,
                    "source_agent": "clawx",
                    "teacher_model": "gemini-3.7-flash",
                    "request_json": {"messages": [{"role": "user", "content": "test"}]},
                    "response_json": {"choices": [{"message": {"content": "ok"}}]},
                    "latency_ms": 200,
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "status": "success",
                    "metadata": {"has_reasoning": True},
                },
            ]

            inserted = await db.insert_captures_batch(records)
            self.assertEqual(inserted, 2)

            async with aiosqlite.connect(db_path) as conn:
                cursor = await conn.execute("SELECT turn_id, source_agent, teacher_model, status FROM captured_turns")
                rows = await cursor.fetchall()
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0][0], "turn-1")
                self.assertEqual(rows[0][1], "nanobot")
                self.assertEqual(rows[1][0], "turn-2")
                self.assertEqual(rows[1][1], "clawx")

            await db.close()


class TestCaptureManagerQueue(unittest.IsolatedAsyncioTestCase):
    async def test_manager_enqueue_and_flush(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "queue_test.db")
            manager = CaptureManager(
                db_path=db_path,
                enabled=True,
                queue_maxsize=100,
                batch_size=2,
                batch_timeout=0.1,
            )
            await manager.start()

            # Enqueue non-stream turn with reasoning
            manager.enqueue_turn(
                request_payload={"model": "gemini-3.7-flash-high", "messages": [{"role": "user", "content": "compute 2+2"}]},
                response_data={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "<think>2+2=4</think>4",
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
                source_agent="agent-math",
                latency_ms=85,
                stream=False,
            )

            # Enqueue stream turn
            sse_frames = [
                b'data: {"choices": [{"delta": {"reasoning_content": "I am thinking."}}]}\n\n',
                b'data: {"choices": [{"delta": {"content": "Answer"}}]}\n\n',
                b'data: {"usage": {"prompt_tokens": 20, "completion_tokens": 10}}\n\n',
                b'data: [DONE]\n\n',
            ]
            manager.enqueue_turn(
                request_payload={"model": "gemini-3.7-flash-high", "messages": [{"role": "user", "content": "hello"}]},
                response_data=sse_frames,
                source_agent="agent-chat",
                latency_ms=120,
                stream=True,
            )

            # Wait briefly for batch writer to persist
            await asyncio.sleep(0.3)
            await manager.stop()

            # Verify in SQLite database
            async with aiosqlite.connect(db_path) as conn:
                cursor = await conn.execute(
                    "SELECT source_agent, prompt_tokens, completion_tokens, metadata_json, response_json FROM captured_turns ORDER BY id ASC"
                )
                rows = await cursor.fetchall()
                self.assertEqual(len(rows), 2)

                # Row 1: extracted from <think> tag
                self.assertEqual(rows[0][0], "agent-math")
                self.assertEqual(rows[0][1], 10)
                self.assertEqual(rows[0][2], 5)
                meta1 = json.loads(rows[0][3])
                self.assertTrue(meta1.get("has_reasoning"))

                # Row 2: extracted from SSE reasoning_content
                self.assertEqual(rows[1][0], "agent-chat")
                self.assertEqual(rows[1][1], 20)
                self.assertEqual(rows[1][2], 10)
                meta2 = json.loads(rows[1][3])
                self.assertTrue(meta2.get("has_reasoning"))

                # Check compatibility with capture-proxy's message_builder
                if response_to_assistant_turn is not None:
                    row2_resp = rows[1][4]
                    turn = response_to_assistant_turn(row2_resp)
                    self.assertIsNotNone(turn)
                    self.assertIn("<think>", turn.get("content", ""))
                    self.assertIn("I am thinking.", turn.get("content", ""))

    async def test_brain_transcript_background_enrichment(self):
        with tempfile.TemporaryDirectory() as brain_tmpdir, tempfile.TemporaryDirectory() as db_tmpdir:
            conv_id = "test-conv-brain-123"
            log_dir = Path(brain_tmpdir) / conv_id / ".system_generated" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            transcript_file = log_dir / "transcript_full.jsonl"

            # Write mock brain transcript
            with open(transcript_file, "w", encoding="utf-8") as f:
                f.write(json.dumps({"step_index": 0, "type": "USER_INPUT", "content": "What is the capital of France?"}) + "\n")
                f.write(json.dumps({
                    "step_index": 1,
                    "type": "PLANNER_RESPONSE",
                    "content": "Paris",
                    "thinking": "France capital query. Answer is Paris directly.",
                }) + "\n")

            db_path = os.path.join(db_tmpdir, "brain_capture.db")
            manager = CaptureManager(
                db_path=db_path,
                enabled=True,
                queue_maxsize=10,
                batch_size=1,
                batch_timeout=0.05,
                brain_dir=brain_tmpdir,
            )
            await manager.start()

            # Enqueue turn with NO reasoning in stream chunks, but WITH conversation_id
            sse_frames = [
                b'data: {"choices": [{"delta": {"content": "Paris"}}]}\n\n',
                b'data: {"usage": {"prompt_tokens": 12, "completion_tokens": 1}}\n\n',
                b'data: [DONE]\n\n',
            ]
            manager.enqueue_turn(
                request_payload={"model": "gemini-3.7-flash", "messages": [{"role": "user", "content": "What is the capital of France?"}]},
                response_data=sse_frames,
                source_agent="geo-bot",
                latency_ms=90,
                stream=True,
                conversation_id=conv_id,
            )

            await asyncio.sleep(0.3)
            await manager.stop()

            # Verify that background worker enriched the turn with thinking from brain
            async with aiosqlite.connect(db_path) as conn:
                cursor = await conn.execute(
                    "SELECT source_agent, metadata_json, response_json FROM captured_turns WHERE turn_id IS NOT NULL"
                )
                row = await cursor.fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row[0], "geo-bot")
                meta = json.loads(row[1])
                self.assertTrue(meta.get("has_reasoning"))
                self.assertEqual(meta.get("cot_source"), "brain_transcript")
                self.assertEqual(meta.get("conversation_id"), conv_id)

                # Check that response_json has reasoning_content and formats with message_builder
                response_json = row[2]
                self.assertIn("reasoning_content", response_json)
                self.assertIn("France capital query.", response_json)

                if response_to_assistant_turn is not None:
                    turn = response_to_assistant_turn(response_json)
                    self.assertIsNotNone(turn)
                    self.assertIn("<think>\nFrance capital query. Answer is Paris directly.\n</think>", turn.get("content", ""))


if __name__ == "__main__":
    unittest.main()
