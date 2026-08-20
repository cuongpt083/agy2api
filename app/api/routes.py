import time
import uuid
import io
from typing import List, Optional
from fastapi import APIRouter, Depends, Request, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import StreamingResponse, Response, JSONResponse
from pydantic import BaseModel, Field
from app.api.models import ChatCompletionRequest, ChatCompletionResponse, Choice, ChoiceMessage, Usage, ModelList, Model, SpeechRequest
from app.core.security import get_api_key
from app.core.agy_runner import run_agy_prompt, stream_agy_prompt
from app.core.openai_sse import (
    extract_agent_text_delta,
    format_sse,
    next_text_delta,
    openai_chunk,
    usage_from_agy,
)
import logging
import re
from app.core.file_handler import TempFileManager
from app.core.capcut_api import AsyncCapCutWrapper
from app.core.model_manager import get_available_models

logger = logging.getLogger(__name__)

router = APIRouter()
capcut_wrapper = AsyncCapCutWrapper()

class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., description="A text description of the desired image(s). (Tips: You can include desired aspect ratios here like 9:16 or 16:9)")
    n: Optional[int] = Field(1, description="The number of images to generate")
    response_format: Optional[str] = Field("url", description="The format in which the generated images are returned. Must be one of url or b64_json")
    reference_images: Optional[List[str]] = Field(None, description="Optional list of base64 data URIs to use as reference images.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "A cute orange cat playing with a ball of yarn, cartoon style, tỉ lệ 9:16",
                    "n": 1,
                    "response_format": "url"
                }
            ]
        }
    }

class ImageObject(BaseModel):
    url: Optional[str] = None
    b64_json: Optional[str] = None

class ImageGenerationResponse(BaseModel):
    created: int
    data: List[ImageObject]

@router.get("/models", response_model=ModelList, summary="List Models", description="Returns a list of available AI models.")
async def list_models(api_key: str = Depends(get_api_key)):
    models = await get_available_models()
    return ModelList(data=models)

def _ext_from_data_uri(url: str) -> str:
    ext = ".png"
    match = re.match(r"^data:([^/]+)/([^;,]+)", url)
    if not match:
        return ext
    mime_sub = match.group(2).lower()
    if mime_sub in ["jpeg", "jpg"]:
        return ".jpg"
    if mime_sub == "pdf":
        return ".pdf"
    if mime_sub == "msword":
        return ".doc"
    if "wordprocessingml" in mime_sub:
        return ".docx"
    if mime_sub == "plain":
        return ".txt"
    if mime_sub == "csv":
        return ".csv"
    if mime_sub in ["png", "gif", "webp"]:
        return f".{mime_sub}"
    return f".{mime_sub}"


def build_chat_prompt(req: ChatCompletionRequest, file_mgr: TempFileManager) -> tuple[str, list[str]]:
    prompt_lines = []
    files_to_attach = []

    for msg in req.messages:
        if isinstance(msg.content, str):
            prompt_lines.append(f"{msg.role.capitalize()}: {msg.content}")
        elif isinstance(msg.content, list):
            text_parts = []
            for p in msg.content:
                if p.get("type") == "text":
                    text_parts.append(p.get("text", ""))
                elif p.get("type") == "image_url":
                    url = p.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        ext = _ext_from_data_uri(url)
                        try:
                            fpath = file_mgr.add_base64_file(url, ext=ext)
                            files_to_attach.append(fpath)
                            text_parts.append(f"[Attached Image: {fpath}]")
                        except Exception as e:
                            text_parts.append(f"[Failed to attach image: {e}]")
                    else:
                        text_parts.append(f"[Image URL: {url}]")
            prompt_lines.append(f"{msg.role.capitalize()}: {' '.join(text_parts)}")

    prompt_lines.append("Assistant: ")
    return "\n".join(prompt_lines), files_to_attach


def _assistant_text(agy_response) -> str:
    if isinstance(agy_response, dict):
        return agy_response.get("text") or agy_response.get("content") or agy_response.get("response") or str(agy_response)
    return str(agy_response)


async def _sse_chat_stream(prompt: str, model: str, chat_id: str, created: int, file_mgr: TempFileManager):
    sent = ""
    role_sent = False
    try:
        async for event in stream_agy_prompt(prompt=prompt, model=model):
            piece, sent = extract_agent_text_delta(event, sent)
            if piece:
                delta = {"content": piece}
                if not role_sent:
                    delta["role"] = "assistant"
                    role_sent = True
                yield format_sse(openai_chunk(chat_id, created, model, delta))
            if event.get("event") != "result":
                continue
            result = event.get("result") or {}
            final_text = result.get("response") or result.get("text") or ""
            piece, sent = next_text_delta(final_text, sent)
            if piece:
                delta = {"content": piece}
                if not role_sent:
                    delta["role"] = "assistant"
                    role_sent = True
                yield format_sse(openai_chunk(chat_id, created, model, delta))
            yield format_sse(
                openai_chunk(
                    chat_id,
                    created,
                    model,
                    {},
                    finish_reason="stop",
                    usage=usage_from_agy(result.get("usage")),
                )
            )
        yield format_sse("[DONE]")
    except Exception as e:
        logger.error("AGY stream failed: %s", e, exc_info=True)
        err = {
            "error": {
                "message": str(e),
                "type": "server_error",
            }
        }
        yield format_sse(err)
        yield format_sse("[DONE]")
    finally:
        file_mgr.cleanup()


@router.post("/chat/completions", summary="Chat Completions", description="Creates a model response for the given chat conversation. Supports multimodal inputs via base64 data URIs. Set stream=true for OpenAI-compatible SSE.")
async def chat_completions(req: ChatCompletionRequest, background_tasks: BackgroundTasks, api_key: str = Depends(get_api_key)):
    logger.info(f"Processing chat completions for model: {req.model} stream={req.stream}")
    file_mgr = TempFileManager()
    final_prompt, files_to_attach = build_chat_prompt(req, file_mgr)

    if req.stream:
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        return StreamingResponse(
            _sse_chat_stream(final_prompt, req.model, chat_id, created, file_mgr),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    background_tasks.add_task(file_mgr.cleanup)
    agy_response = await run_agy_prompt(prompt=final_prompt, model=req.model, files=files_to_attach)
    assistant_text = _assistant_text(agy_response)
    usage_data = Usage()
    if isinstance(agy_response, dict) and agy_response.get("usage"):
        mapped = usage_from_agy(agy_response.get("usage"))
        usage_data = Usage(
            prompt_tokens=mapped["prompt_tokens"],
            completion_tokens=mapped["completion_tokens"],
            total_tokens=mapped["total_tokens"],
        )
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=req.model,
        choices=[Choice(message=ChoiceMessage(content=assistant_text))],
        usage=usage_data,
    )

@router.post("/images/generations", response_model=ImageGenerationResponse, summary="Image Generations", description="Creates an image given a prompt using the AGY artist skills.")
async def generate_image(req: ImageGenerationRequest, background_tasks: BackgroundTasks, api_key: str = Depends(get_api_key)):
    logger.info(f"Generating image. Prompt: {req.prompt[:50]}...")
    file_mgr = TempFileManager()
    background_tasks.add_task(file_mgr.cleanup)
    
    ref_paths = []
    if req.reference_images:
        for url in req.reference_images:
            if url.startswith("data:"):
                ext = ".png"
                if "jpeg" in url or "jpg" in url: ext = ".jpg"
                try:
                    fpath = file_mgr.add_base64_file(url, ext=ext)
                    ref_paths.append(fpath)
                except Exception:
                    pass

    # We instruct AGY to generate an image and return the path/base64 in JSON format
    prompt = f"Generate an image for the following prompt: '{req.prompt}'. Return ONLY the absolute local file path of the generated image in your response, do not include any other conversational text."
    
    if ref_paths:
        paths_str = ", ".join([f"'{p}'" for p in ref_paths])
        prompt = f"Use the reference images at {paths_str} to generate an image for the following prompt: '{req.prompt}'. Return ONLY the absolute local file path of the generated image in your response, do not include any other conversational text."
    
    agy_response = await run_agy_prompt(prompt=prompt, output_format="json")
    
    # Extract path
    image_path = ""
    if isinstance(agy_response, dict):
        image_path = agy_response.get("text") or agy_response.get("content") or agy_response.get("response") or str(agy_response)
    else:
        image_path = str(agy_response)
        
    image_path = image_path.strip()
    img_data = ImageObject(url=image_path)
    
    import os
    if os.path.exists(image_path) and os.path.isfile(image_path):
        import base64
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
            if req.response_format == "b64_json":
                img_data = ImageObject(b64_json=b64)
            else:
                img_data = ImageObject(url=f"data:image/png;base64,{b64}")
                
        # Schedule cleanup of the generated image file after returning the response
        def remove_file(path):
            try:
                os.remove(path)
            except Exception:
                pass
        
        background_tasks.add_task(remove_file, image_path)

    return ImageGenerationResponse(
        created=int(time.time()),
        data=[img_data]
    )

import subprocess

@router.get("/logs", summary="Get System Logs", description="Read the latest system logs of the AGY Wrapper service.")
async def get_logs(lines: int = 100, api_key: str = Depends(get_api_key)):
    try:
        result = subprocess.run(
            ["journalctl", "--user", "-u", "agy-wrapper.service", "-n", str(lines), "--no-pager"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return {"logs": result.stdout}
        else:
            return {"logs": f"Failed to read logs (code {result.returncode}): {result.stderr}"}
    except Exception as e:
        return {"logs": f"Error reading logs: {str(e)}"}

@router.post(
    "/audio/speech", 
    summary="Text to Speech (Audio Generations)", 
    description="Tạo tệp âm thanh từ văn bản dựa trên chuẩn OpenAI Audio API (engine CapCut).",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Binary stream của file MP3 (audio/mpeg)",
            "content": {"audio/mpeg": {}}
        }
    }
)
async def audio_speech(req: SpeechRequest, api_key: str = Depends(get_api_key)):
    logger.info(f"Generating speech (voice={req.voice}, speed={req.speed}). Text: {req.input[:50]}...")
    try:
        audio_bytes = await capcut_wrapper.generate_speech(
            text=req.input,
            voice=req.voice,
            speed=req.speed
        )
        return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/mpeg")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.get(
    "/audio/voices", 
    summary="List Voices", 
    description="Lấy danh sách tất cả các giọng đọc (voices) khả dụng từ engine CapCut."
)
async def audio_voices(api_key: str = Depends(get_api_key)):
    try:
        voices = capcut_wrapper.get_voices()
        return JSONResponse(content={"voices": voices})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post(
    "/audio/transcriptions", 
    summary="Speech to Text (Audio Transcriptions)", 
    description="Chuyển đổi file âm thanh thành văn bản hoặc phụ đề thời gian chuẩn."
)
async def audio_transcriptions(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Tệp âm thanh cần upload (mp3, mp4, wav, v.v...)"),
    model: str = Form("whisper-1", description="ID của mô hình (vd: whisper-1)"),
    language: str = Form(None, description="Mã ngôn ngữ (vd: en-US, vi-VN). Bỏ trống để tự nhận diện."),
    response_format: str = Form("json", description="Định dạng trả về (json, text, srt, vtt)"),
    api_key: str = Depends(get_api_key)
):
    logger.info(f"Transcribing audio file: {file.filename}, language: {language}, format: {response_format}")
    try:
        file_mgr = TempFileManager()
        background_tasks.add_task(file_mgr.cleanup)
        
        import os
        ext = os.path.splitext(file.filename)[1] if file.filename else ".mp3"
        temp_path = os.path.join(file_mgr.temp_dir.name, f"upload{ext}")
        with open(temp_path, "wb") as f:
            f.write(await file.read())
            
        transcription = await capcut_wrapper.transcribe_audio(
            file_path=temp_path,
            response_format=response_format,
            language=language
        )
        
        if response_format in ["json", "verbose_json"]:
            import json
            return JSONResponse(content=json.loads(transcription))
        else:
            return Response(content=transcription, media_type="text/plain")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
