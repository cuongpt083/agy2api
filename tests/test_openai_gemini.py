import json
import unittest

from app.core.openai_gemini import (
    ThoughtSignatureCache,
    build_gemini_request,
    gemini_candidate_to_openai_choice,
    gemini_stream_to_sse,
    resolve_gemini_model,
)


class TestResolveGeminiModel(unittest.TestCase):
    def test_strips_effort_suffix_from_agy_slug(self):
        resolved = resolve_gemini_model("gemini-3.7-flash-high")
        self.assertEqual(resolved.api_model, "gemini-3.7-flash")
        self.assertEqual(resolved.thinking_level, "high")

    def test_maps_display_name_to_api_model(self):
        resolved = resolve_gemini_model("Gemini 3.7 Flash (High)")
        self.assertEqual(resolved.api_model, "gemini-3.7-flash")
        self.assertEqual(resolved.thinking_level, "high")

    def test_passes_through_gemini_api_id(self):
        resolved = resolve_gemini_model("gemini-2.5-flash")
        self.assertEqual(resolved.api_model, "gemini-2.5-flash")
        self.assertIsNone(resolved.thinking_level)

    def test_rejects_non_gemini_models(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_gemini_model("claude-sonnet-4-6")
        self.assertIn("Gemini API", str(ctx.exception))


class TestBuildGeminiRequest(unittest.TestCase):
    def test_maps_openai_function_tools_to_function_declarations(self):
        call = build_gemini_request(
            {
                "model": "gemini-3.7-flash-high",
                "messages": [{"role": "user", "content": "read foo"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read",
                            "description": "Read a file",
                            "parameters": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                            },
                        },
                    }
                ],
                "tool_choice": "auto",
            }
        )
        self.assertEqual(call.model, "gemini-3.7-flash")
        self.assertEqual(
            call.body["tools"],
            [
                {
                    "functionDeclarations": [
                        {
                            "name": "read",
                            "description": "Read a file",
                            "parameters": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                            },
                        }
                    ]
                }
            ],
        )
        self.assertEqual(
            call.body["toolConfig"]["functionCallingConfig"]["mode"],
            "AUTO",
        )
        self.assertEqual(call.body["generationConfig"]["thinkingConfig"]["thinkingLevel"], "HIGH")

    def test_required_tool_choice_sets_any_mode(self):
        call = build_gemini_request(
            {
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "bash", "parameters": {"type": "object"}},
                    }
                ],
                "tool_choice": "required",
            }
        )
        self.assertEqual(call.body["toolConfig"]["functionCallingConfig"]["mode"], "ANY")

    def test_named_tool_choice_restricts_allowed_functions(self):
        call = build_gemini_request(
            {
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "read", "parameters": {"type": "object"}},
                    }
                ],
                "tool_choice": {"type": "function", "function": {"name": "read"}},
            }
        )
        cfg = call.body["toolConfig"]["functionCallingConfig"]
        self.assertEqual(cfg["mode"], "ANY")
        self.assertEqual(cfg["allowedFunctionNames"], ["read"])

    def test_merges_system_and_developer_into_system_instruction(self):
        call = build_gemini_request(
            {
                "model": "gemini-2.5-flash",
                "messages": [
                    {"role": "system", "content": "You are OMP."},
                    {"role": "developer", "content": "Use tools."},
                    {"role": "user", "content": "fix the bug"},
                ],
            }
        )
        self.assertEqual(
            call.body["systemInstruction"]["parts"],
            [{"text": "You are OMP.\nUse tools."}],
        )
        self.assertEqual(
            call.body["contents"],
            [{"role": "user", "parts": [{"text": "fix the bug"}]}],
        )

    def test_converts_assistant_tool_calls_and_tool_results(self):
        cache = ThoughtSignatureCache()
        cache.put("call_read_1", "sig-abc", name="read")
        call = build_gemini_request(
            {
                "model": "gemini-2.5-flash",
                "messages": [
                    {"role": "user", "content": "read src/foo.ts"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_read_1",
                                "type": "function",
                                "function": {
                                    "name": "read",
                                    "arguments": '{"path":"src/foo.ts"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_read_1",
                        "name": "read",
                        "content": "export const x = 1",
                    },
                ],
            },
            signatures=cache,
        )
        contents = call.body["contents"]
        self.assertEqual(contents[0]["role"], "user")
        model_part = contents[1]["parts"][0]
        self.assertEqual(contents[1]["role"], "model")
        self.assertEqual(
            model_part["functionCall"],
            {"name": "read", "args": {"path": "src/foo.ts"}, "id": "call_read_1"},
        )
        self.assertEqual(model_part["thoughtSignature"], "sig-abc")
        self.assertEqual(contents[2]["role"], "user")
        self.assertEqual(
            contents[2]["parts"][0]["functionResponse"],
            {
                "name": "read",
                "id": "call_read_1",
                "response": {"output": "export const x = 1"},
            },
        )

    def test_uses_extra_content_signature_when_present(self):
        call = build_gemini_request(
            {
                "model": "gemini-2.5-flash",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "bash", "arguments": "{}"},
                                "extra_content": {
                                    "google": {"thought_signature": "from-wire"}
                                },
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
                ],
            }
        )
        self.assertEqual(call.body["contents"][1]["parts"][0]["thoughtSignature"], "from-wire")

    def test_merges_consecutive_tool_results_into_one_user_turn(self):
        call = build_gemini_request(
            {
                "model": "gemini-2.5-flash",
                "messages": [
                    {"role": "user", "content": "do both"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "read", "arguments": "{}"},
                                "extra_content": {"google": {"thought_signature": "s1"}},
                            },
                            {
                                "id": "c2",
                                "type": "function",
                                "function": {"name": "bash", "arguments": "{}"},
                                "extra_content": {"google": {"thought_signature": "s2"}},
                            },
                        ],
                    },
                    {"role": "tool", "tool_call_id": "c1", "name": "read", "content": "a"},
                    {"role": "tool", "tool_call_id": "c2", "name": "bash", "content": "b"},
                ],
            }
        )
        self.assertEqual(len(call.body["contents"]), 3)
        responses = [p["functionResponse"]["id"] for p in call.body["contents"][2]["parts"]]
        self.assertEqual(responses, ["c1", "c2"])

    def test_maps_max_tokens_to_max_output_tokens(self):
        call = build_gemini_request(
            {
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 2048,
            }
        )
        self.assertEqual(call.body["generationConfig"]["maxOutputTokens"], 2048)

    def test_reasoning_effort_overrides_model_suffix(self):
        call = build_gemini_request(
            {
                "model": "gemini-3.7-flash-high",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "low",
            }
        )
        self.assertEqual(call.body["generationConfig"]["thinkingConfig"]["thinkingLevel"], "LOW")


class TestGeminiCandidateToOpenai(unittest.TestCase):
    def test_text_only_choice_has_stop_finish_reason(self):
        choice = gemini_candidate_to_openai_choice(
            {
                "content": {"parts": [{"text": "hello"}]},
                "finishReason": "STOP",
            }
        )
        self.assertEqual(choice["message"]["role"], "assistant")
        self.assertEqual(choice["message"]["content"], "hello")
        self.assertIsNone(choice["message"].get("tool_calls"))
        self.assertEqual(choice["finish_reason"], "stop")

    def test_function_call_becomes_openai_tool_calls_and_caches_signature(self):
        cache = ThoughtSignatureCache()
        choice = gemini_candidate_to_openai_choice(
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "id": "call_read_1",
                                "name": "read",
                                "args": {"path": "src/foo.ts"},
                            },
                            "thoughtSignature": "sig-xyz",
                        }
                    ]
                },
                "finishReason": "STOP",
            },
            signatures=cache,
        )
        tool_call = choice["message"]["tool_calls"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIsNone(choice["message"]["content"])
        self.assertEqual(tool_call["id"], "call_read_1")
        self.assertEqual(tool_call["type"], "function")
        self.assertEqual(tool_call["function"]["name"], "read")
        self.assertEqual(json.loads(tool_call["function"]["arguments"]), {"path": "src/foo.ts"})
        self.assertEqual(
            tool_call["extra_content"]["google"]["thought_signature"],
            "sig-xyz",
        )
        self.assertEqual(cache.get("call_read_1", name="read"), "sig-xyz")


class TestGeminiStreamToSse(unittest.TestCase):
    def test_streams_text_then_done(self):
        frames = list(
            gemini_stream_to_sse(
                [
                    {"candidates": [{"content": {"parts": [{"text": "Hi"}]}}]},
                    {
                        "candidates": [{"finishReason": "STOP"}],
                        "usageMetadata": {
                            "promptTokenCount": 3,
                            "candidatesTokenCount": 1,
                            "totalTokenCount": 4,
                        },
                    },
                ],
                chat_id="chatcmpl-test",
                created=1,
                model="gemini-2.5-flash",
            )
        )
        decoded = [f.decode("utf-8") for f in frames]
        self.assertIn('"content": "Hi"', decoded[0])
        self.assertIn('"role": "assistant"', decoded[0])
        self.assertIn('"finish_reason": "stop"', decoded[-2])
        self.assertIn('"prompt_tokens": 3', decoded[-2])
        self.assertEqual(decoded[-1], "data: [DONE]\n\n")

    def test_streams_tool_calls_with_finish_reason(self):
        cache = ThoughtSignatureCache()
        frames = list(
            gemini_stream_to_sse(
                [
                    {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "functionCall": {
                                                "id": "call_1",
                                                "name": "read",
                                                "args": {"path": "a.py"},
                                            },
                                            "thoughtSignature": "sig",
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    {"candidates": [{"finishReason": "STOP"}]},
                ],
                chat_id="chatcmpl-test",
                created=1,
                model="gemini-2.5-flash",
                signatures=cache,
            )
        )
        decoded = [f.decode("utf-8") for f in frames]
        joined = "".join(decoded)
        self.assertIn('"tool_calls"', joined)
        self.assertIn('"name": "read"', joined)
        self.assertIn('"finish_reason": "tool_calls"', joined)
        self.assertIn("data: [DONE]", decoded[-1])
        self.assertEqual(cache.get("call_1", name="read"), "sig")


if __name__ == "__main__":
    unittest.main()
