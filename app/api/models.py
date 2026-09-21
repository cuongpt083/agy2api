from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Dict, Any, Union

class FunctionSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Optional[str] = "function"
    function: Optional[FunctionSpec] = None

class ToolCallFunction(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    arguments: Optional[str] = "{}"

class ToolCall(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: Optional[str] = None
    type: Optional[str] = "function"
    function: Optional[ToolCallFunction] = None
    extra_content: Optional[Dict[str, Any]] = None
    index: Optional[int] = None

class Message(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str = Field(..., description="The role of the messages author, e.g. user, assistant, system, or tool")
    content: Optional[Union[str, List[Dict[str, Any]]]] = Field(
        None,
        description="The contents of the message. Can be a string, an array of content parts, or null when tool_calls is set.",
    )
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "description": "Để đính kèm file (hình ảnh, tài liệu pdf, docx, txt...), hãy sử dụng mảng content và truyền chuỗi base64 dạng data URI (vd: data:image/jpeg;base64,... hoặc data:application/pdf;base64,...) vào trường image_url. Mặc dù chuẩn gốc là image_url, hệ thống hỗ trợ phân giải tự động các loại file khác dựa vào mime type trong data URI.",
            "examples": [
                {
                    "model": "Gemini 3.6 Flash (High)",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Viết cho tôi một hàm Python tính Fibonacci"
                        }
                    ]
                },
                {
                    "model": "Gemini 3.6 Flash (High)",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Tóm tắt nội dung tài liệu này và mô tả bức ảnh."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD..."
                                    }
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "data:application/pdf;base64,JVBERi0xLjQKJcOkw7zDtsOfCjI..."
                                    }
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    )
    model: str = Field(..., description="ID of the model to use, e.g. 'Gemini 3.6 Flash (High)'")
    messages: List[Message]
    temperature: Optional[float] = Field(1.0, description="Sampling temperature")
    stream: Optional[bool] = Field(False, description="Whether to stream back partial progress")
    tools: Optional[List[ToolSpec]] = Field(None, description="OpenAI function tools. When set, requests go to the Gemini API (not agy).")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None

class ChoiceMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str = "assistant"
    content: Optional[str] = None
    reasoning_content: Optional[str] = Field(None, description="Reasoning or chain-of-thought content")
    tool_calls: Optional[List[Dict[str, Any]]] = None

class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    completion_tokens_details: Optional[Dict[str, Any]] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

class Model(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "google"

class ModelList(BaseModel):
    object: str = "list"
    data: List[Model]

class SpeechRequest(BaseModel):
    model: str = Field(..., description="ID của model (vd: 'tts-1', 'tts-1-hd')")
    input: str = Field(..., description="Đoạn văn bản cần chuyển thành giọng nói.")
    voice: str = Field("alloy", description="Giọng đọc cần sử dụng. (vd: 'alloy', 'BV074_streaming')")
    response_format: Optional[str] = Field("mp3", description="Định dạng trả về. Mặc định là mp3.")
    speed: Optional[float] = Field(1.0, description="Tốc độ đọc (0.25 đến 4.0). Mặc định là 1.0.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "model": "tts-1",
                "input": "Xin chào thế giới!",
                "voice": "alloy",
                "response_format": "mp3",
                "speed": 1.0
            }
        }
    }
