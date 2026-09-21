"""
Long-term Dataset Writer for CoT (Chain of Thought / Reasoning) turns.
Persists high-quality reasoning turns to rotating monthly JSONL dataset files.
"""

import os
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default location for long-term dataset storage
DEFAULT_DATA_DIR = Path("/app/data/cot_dataset") if Path("/app/data").is_dir() else Path(__file__).resolve().parents[2] / "data" / "cot_dataset"
DATASET_DIR = Path(os.environ.get("COT_DATASET_DIR", str(DEFAULT_DATA_DIR)))


def get_current_dataset_file() -> Path:
    """Returns the current month's dataset filepath, ensuring parent directory exists."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    month_str = datetime.now(timezone.utc).strftime("%Y-%m")
    return DATASET_DIR / f"cot_dataset_{month_str}.jsonl"


def save_cot_turn(
    messages: List[Dict[str, Any]] | str,
    response_content: str,
    reasoning_content: Optional[str],
    model: str,
    usage: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    conversation_id: Optional[str] = None,
) -> bool:
    """
    Appends a completed assistant turn with reasoning to the long-term dataset.
    Only saves if reasoning_content is present and non-empty.
    Fail-safe: logs errors without raising exceptions.
    """
    if not reasoning_content or not reasoning_content.strip():
        return False

    try:
        record_id = conversation_id or f"turn-{uuid.uuid4().hex[:12]}"
        now_iso = datetime.now(timezone.utc).isoformat()

        # Format messages structure
        formatted_messages = []
        if isinstance(messages, str):
            formatted_messages.append({"role": "user", "content": messages})
        elif isinstance(messages, list):
            for m in messages:
                if isinstance(m, dict):
                    formatted_messages.append({
                        "role": m.get("role", "user"),
                        "content": m.get("content", ""),
                    })
                elif hasattr(m, "role") and hasattr(m, "content"):
                    formatted_messages.append({
                        "role": m.role,
                        "content": m.content,
                    })

        # Append assistant turn
        formatted_messages.append({
            "role": "assistant",
            "content": response_content,
            "reasoning_content": reasoning_content.strip(),
        })

        record = {
            "id": record_id,
            "timestamp": now_iso,
            "model": model,
            "messages": formatted_messages,
            "reasoning_content": reasoning_content.strip(),
            "content": response_content,
            "usage": usage or {},
            "metadata": metadata or {},
        }

        target_file = get_current_dataset_file()
        with open(target_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("Saved CoT turn %s (%d chars reasoning) to %s", record_id, len(reasoning_content), target_file.name)
        return True
    except Exception as e:
        logger.error("Failed to save CoT turn to dataset: %s", e, exc_info=True)
        return False
