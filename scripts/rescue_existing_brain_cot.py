#!/usr/bin/env python3
"""
Rescue script: Extracts reasoning/chain of thought (thinking) and original prompts
from AGY CLI brain conversation transcripts (~/.gemini/antigravity-cli/brain/)
and exports them to a persistent JSONL dataset file.
"""

import os
import sys
import glob
import json
import re
from datetime import datetime
from pathlib import Path


def clean_view_file_prompt(content: str) -> str:
    """Cleans up prompt text returned by AGY's view_file tool step."""
    lines = content.splitlines()
    cleaned = []
    in_file_content = False
    for line in lines:
        # AGY view_file header markers
        if line.startswith("Showing lines ") or "The following code has been modified to include a line number" in line:
            in_file_content = True
            continue
        if line.startswith("The above content ") or line.startswith("File Path:"):
            continue
        if in_file_content:
            # Strip leading line number prefix: "123: "
            m = re.match(r"^\s*\d+:\s?(.*)$", line)
            if m:
                cleaned.append(m.group(1))
            else:
                cleaned.append(line)
        else:
            cleaned.append(line)
    result = "\n".join(cleaned).strip()
    return result if result else content.strip()


def extract_cot_from_transcript(transcript_path: str) -> dict | None:
    """Extracts a high-quality CoT turn from a transcript_full.jsonl file."""
    steps = []
    try:
        with open(transcript_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    steps.append(json.loads(line))
    except Exception as e:
        print(f"Error reading {transcript_path}: {e}", file=sys.stderr)
        return None

    if not steps:
        return None

    conv_id = Path(transcript_path).parents[2].name
    model = "Gemini 3.7 Flash"
    created_at = None
    user_prompt = ""
    thinking = ""
    response = ""
    task_type = "general"

    # Step 1: Discover model from first step metadata if present
    first_content = str(steps[0].get("content", ""))
    if "Model Selection` from" in first_content:
        m = re.search(r"Model Selection` from (.*?) to (.*?)\.", first_content)
        if m:
            model = m.group(2).strip()

    created_at = steps[0].get("created_at")

    # Step 2: Extract real user prompt
    # Check if there is a view_file step that read the prompt file
    file_prompt_content = ""
    for s in steps:
        if s.get("type") == "GENERIC" and "prompt.txt" in str(s.get("content", "")):
            file_prompt_content = clean_view_file_prompt(s.get("content", ""))
            break

    if file_prompt_content:
        user_prompt = file_prompt_content
    else:
        # Fallback to step 0 USER_INPUT
        raw_user = steps[0].get("content", "")
        # Strip <USER_REQUEST> tags if present
        m = re.search(r"<USER_REQUEST>(.*?)</USER_REQUEST>", raw_user, re.DOTALL)
        if m:
            user_prompt = m.group(1).strip()
        else:
            user_prompt = raw_user.strip()

    # Step 3: Extract thinking and final response
    for s in steps:
        if s.get("thinking"):
            thinking = s.get("thinking").strip()
        if s.get("type") == "PLANNER_RESPONSE" and s.get("content"):
            response = s.get("content").strip()

    if not thinking:
        return None

    # Classify task
    if "Knowledge Graph Specialist" in user_prompt or "Entity Extraction" in user_prompt:
        task_type = "knowledge_graph_extraction"
    elif "def " in user_prompt or "import " in user_prompt:
        task_type = "coding"
    elif "toán" in user_prompt.lower() or "bài tập" in user_prompt.lower() or "sgk" in user_prompt.lower():
        task_type = "math_curriculum"

    return {
        "id": conv_id,
        "timestamp": created_at or datetime.utcnow().isoformat() + "Z",
        "model": model,
        "task_type": task_type,
        "messages": [
            {"role": "user", "content": user_prompt},
            {
                "role": "assistant",
                "content": response,
                "reasoning_content": thinking,
            }
        ],
        "reasoning_content": thinking,
        "content": response,
        "metadata": {
            "source": "agy_brain_transcript",
            "thinking_chars": len(thinking),
            "response_chars": len(response),
            "prompt_chars": len(user_prompt),
        }
    }


def main():
    brain_dir = os.path.expanduser("~/.gemini/antigravity-cli/brain")
    output_dir = Path("/home/cuongpt/agy2api/data/cot_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "rescued_cot_turns.jsonl"

    print(f"Scanning {brain_dir} for conversations...")
    pattern = os.path.join(brain_dir, "*", ".system_generated", "logs", "transcript_full.jsonl")
    transcript_files = glob.glob(pattern)
    print(f"Found {len(transcript_files)} conversation transcripts.")

    rescued = []
    skipped_no_thinking = 0

    for tf in transcript_files:
        record = extract_cot_from_transcript(tf)
        if record:
            rescued.append(record)
        else:
            skipped_no_thinking += 1

    print(f"Total rescued conversations with thinking: {len(rescued)}")
    print(f"Skipped without thinking: {skipped_no_thinking}")

    # Write to output file
    with open(output_file, "w", encoding="utf-8") as f:
        for r in rescued:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Successfully exported {len(rescued)} CoT records to {output_file} ({os.path.getsize(output_file)} bytes)")


if __name__ == "__main__":
    main()
