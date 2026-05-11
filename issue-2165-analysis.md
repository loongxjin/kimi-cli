# Issue #2165 分析与修复方案

> 原文：https://github.com/MoonshotAI/kimi-cli/issues/2165
> 问题：Invalid tool call corrupt the whole session

---

## 问题描述

当模型生成 malformed JSON 的 `tool_call.function.arguments`（例如缺失开头引号：`description": "..."`）时：

1. 该 assistant message 被写入 session history（`~/.kimi/sessions/.../context.jsonl`）
2. 下一回合，完整 message history 被发送到 LLM backend
3. vLLM / OpenAI-compatible server 在服务端尝试 `json.loads()` 这些参数，抛出 400 Bad Request
4. 由于 poisoned message 永久存在于 history 中，**后续每次请求都会失败，session 彻底报废**

---

## 根因定位

问题出在 chat provider 发送请求前，没有对 history 中的 `tool_calls[*].function.arguments` 做 JSON 合法性校验。

| Provider | 处理位置 | 原有行为 |
|---|---|---|
| Anthropic | `anthropic.py` | `json.loads()` 失败时抛 `ChatProviderError` |
| Google GenAI | `google_genai.py` | `json.loads()` 失败时抛 `ChatProviderError` |
| OpenAI Responses | `openai_responses.py` | 原样透传 `arguments` string，不做校验 |
| OpenAI Legacy | `openai_legacy.py` | `message.model_dump()` 原样序列化，不做校验 |

对于 OpenAI-compatible backend（包括 vLLM），服务端会校验 `tool_calls[*].function.arguments` 是否为合法 JSON string，非法时直接返回 400。

---

## 修复方案

**核心思路**：在发送请求到 backend **之前**，对所有 provider 都做统一的防御性处理——`json.loads()` 校验，非法时静默 fallback 到空对象，避免 session 被永久阻塞。

### 1. Anthropic Provider

**文件**：`packages/kosong/src/kosong/contrib/chat_provider/anthropic.py`

```python
if tool_call.function.arguments:
    try:
        parsed_arguments = json.loads(tool_call.function.arguments, strict=False)
    except json.JSONDecodeError:  # defensive guard for malformed history
        # Historical tool calls may contain malformed JSON arguments
        # (e.g. from a previous LLM output that failed JSON validation).
        # Fall back to an empty dict so the conversation can continue
        # instead of permanently blocking the session.
        parsed_arguments = {}
    if not isinstance(parsed_arguments, dict):
        parsed_arguments = {}
    tool_input = cast(dict[str, object], parsed_arguments)
else:
    tool_input = {}
```

**变化**：
- `except` 分支从抛 `ChatProviderError` 改为 `parsed_arguments = {}`
- 非 dict 也从抛异常改为 `parsed_arguments = {}`

### 2. Google GenAI Provider

**文件**：`packages/kosong/src/kosong/contrib/chat_provider/google_genai.py`

```python
if tool_call.function.arguments:
    try:
        parsed_arguments = json.loads(tool_call.function.arguments, strict=False)
    except json.JSONDecodeError:  # defensive guard for malformed history
        # Historical tool calls may contain malformed JSON arguments
        # (e.g. from a previous LLM output that failed JSON validation).
        # Fall back to an empty dict so the conversation can continue
        # instead of permanently blocking the session.
        parsed_arguments = {}
    if not isinstance(parsed_arguments, dict):
        parsed_arguments = {}
    args = cast(dict[str, object], parsed_arguments)
else:
    args = {}
```

**变化**：与 Anthropic 完全一致。

### 3. OpenAI Responses Provider

**文件**：`packages/kosong/src/kosong/contrib/chat_provider/openai_responses.py`

```python
import json

for tool_call in message.tool_calls or []:
    arguments = tool_call.function.arguments or "{}"
    try:
        json.loads(arguments, strict=False)
    except json.JSONDecodeError:  # defensive guard for malformed history
        # Historical tool calls may contain malformed JSON arguments
        # (e.g. from a previous LLM output that failed JSON validation).
        # Fall back to an empty dict so the conversation can continue
        # instead of permanently blocking the session.
        arguments = "{}"
    result.append(
        {
            "arguments": arguments,
            "call_id": tool_call.id,
            "name": tool_call.function.name,
            "type": "function_call",
        }
    )
```

**变化**：
- 新增 `import json`
- 发送前校验 `arguments` JSON 合法性，非法时 fallback 到 `"{}"`

### 4. OpenAI Legacy Provider

**文件**：`packages/kosong/src/kosong/contrib/chat_provider/openai_legacy.py`

```python
import json

message = message.model_copy(deep=True)
# defensive guard for malformed history
if message.tool_calls:
    for tc in message.tool_calls:
        if tc.function.arguments:
            try:
                json.loads(tc.function.arguments, strict=False)
            except json.JSONDecodeError:
                # Historical tool calls may contain malformed JSON arguments
                # (e.g. from a previous LLM output that failed JSON validation).
                # Fall back to an empty dict so the conversation can continue
                # instead of permanently blocking the session.
                tc.function.arguments = "{}"
```

**变化**：
- 新增 `import json`
- 在 `model_dump()` 序列化前，深拷贝并清理 `tool_calls` 中的非法 JSON

---

## 影响分析

### 不影响正常逻辑

- 正常合法 JSON 会顺利通过 `json.loads()` 校验，不会进入 fallback 分支
- 对性能影响微乎其微（仅对 history 中的 tool_calls 做一次解析）

### 不修改本地持久化数据

- `openai_legacy.py`：修改的是 `model_copy(deep=True)` 的副本，`context.jsonl` 中的原始数据保持不变
- `openai_responses.py`：修改的是本地变量，不影响原始 message
- `anthropic.py` / `google_genai.py`：同理，仅影响发送给 backend 的请求

### 副作用

- 被 fallback 的 tool call arguments 会变成空对象 `{}`
- 这意味着 LLM backend 收到的是空参数，该 tool call 在 history 中的记录丢失了原始参数
- 但这远优于 session 彻底报废，且 malformed JSON 本来就没有有效信息

---

## 完整 Diff

```diff
diff --git a/packages/kosong/src/kosong/contrib/chat_provider/anthropic.py b/packages/kosong/src/kosong/contrib/chat_provider/anthropic.py
--- a/packages/kosong/src/kosong/contrib/chat_provider/anthropic.py
+++ b/packages/kosong/src/kosong/contrib/chat_provider/anthropic.py
@@ -488,10 +488,14 @@ class Anthropic:
             if tool_call.function.arguments:
                 try:
                     parsed_arguments = json.loads(tool_call.function.arguments, strict=False)
-                except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
-                    raise ChatProviderError("Tool call arguments must be valid JSON.") from exc
+                except json.JSONDecodeError:  # defensive guard for malformed history
+                    # Historical tool calls may contain malformed JSON arguments
+                    # (e.g. from a previous LLM output that failed JSON validation).
+                    # Fall back to an empty dict so the conversation can continue
+                    # instead of permanently blocking the session.
+                    parsed_arguments = {}
                 if not isinstance(parsed_arguments, dict):
-                    raise ChatProviderError("Tool call arguments must be a JSON object.")
+                    parsed_arguments = {}
                 tool_input = cast(dict[str, object], parsed_arguments)
             else:
                 tool_input = {}

diff --git a/packages/kosong/src/kosong/contrib/chat_provider/google_genai.py b/packages/kosong/src/kosong/contrib/chat_provider/google_genai.py
--- a/packages/kosong/src/kosong/contrib/chat_provider/google_genai.py
+++ b/packages/kosong/src/kosong/contrib/chat_provider/google_genai.py
@@ -655,10 +655,14 @@ def message_to_google_genai(message: Message) -> Content:
         if tool_call.function.arguments:
             try:
                 parsed_arguments = json.loads(tool_call.function.arguments, strict=False)
-            except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
-                raise ChatProviderError("Tool call arguments must be valid JSON.") from exc
+            except json.JSONDecodeError:  # defensive guard for malformed history
+                # Historical tool calls may contain malformed JSON arguments
+                # (e.g. from a previous LLM output that failed JSON validation).
+                # Fall back to an empty dict so the conversation can continue
+                # instead of permanently blocking the session.
+                parsed_arguments = {}
             if not isinstance(parsed_arguments, dict):
-                raise ChatProviderError("Tool call arguments must be a JSON object.")
+                parsed_arguments = {}
             args = cast(dict[str, object], parsed_arguments)
         else:
             args = {}

diff --git a/packages/kosong/src/kosong/contrib/chat_provider/openai_responses.py b/packages/kosong/src/kosong/contrib/chat_provider/openai_responses.py
--- a/packages/kosong/src/kosong/contrib/chat_provider/openai_responses.py
+++ b/packages/kosong/src/kosong/contrib/chat_provider/openai_responses.py
@@ -1,4 +1,5 @@
 import copy
+import json
 import uuid
 from collections.abc import AsyncIterator, Sequence
 from typing import TYPE_CHECKING, Any, Self, TypedDict, Unpack, cast, get_args
@@ -328,9 +329,18 @@ class OpenAIResponses:
             flush_pending_parts()
 
         for tool_call in message.tool_calls or []:
+            arguments = tool_call.function.arguments or "{}"
+            try:
+                json.loads(arguments, strict=False)
+            except json.JSONDecodeError:  # defensive guard for malformed history
+                # Historical tool calls may contain malformed JSON arguments
+                # (e.g. from a previous LLM output that failed JSON validation).
+                # Fall back to an empty dict so the conversation can continue
+                # instead of permanently blocking the session.
+                arguments = "{}"
             result.append(
                 {
-                    "arguments": tool_call.function.arguments or "{}",
+                    "arguments": arguments,
                     "call_id": tool_call.id,
                     "name": tool_call.function.name,
                     "type": "function_call",
                 }
             )

diff --git a/packages/kosong/src/kosong/contrib/chat_provider/openai_legacy.py b/packages/kosong/src/kosong/contrib/chat_provider/openai_legacy.py
--- a/packages/kosong/src/kosong/contrib/chat_provider/openai_legacy.py
+++ b/packages/kosong/src/kosong/contrib/chat_provider/openai_legacy.py
@@ -1,4 +1,5 @@
 import copy
+import json
 import uuid
 from collections.abc import AsyncIterator, Sequence
 from typing import TYPE_CHECKING, Any, Self, Unpack, cast
@@ -199,6 +200,18 @@ class OpenAILegacy:
         # So we use `system` role here. OpenAIResponses will use `developer` role.
         # See https://cdn.openai.com/spec/model-spec-2024-05-08.html#definitions
         message = message.model_copy(deep=True)
+        # defensive guard for malformed history
+        if message.tool_calls:
+            for tc in message.tool_calls:
+                if tc.function.arguments:
+                    try:
+                        json.loads(tc.function.arguments, strict=False)
+                    except json.JSONDecodeError:
+                        # Historical tool calls may contain malformed JSON arguments
+                        # (e.g. from a previous LLM output that failed JSON validation).
+                        # Fall back to an empty dict so the conversation can continue
+                        # instead of permanently blocking the session.
+                        tc.function.arguments = "{}"
         reasoning_content: str = ""
         content: list[ContentPart] = []
         for part in message.content:
```

---

## 验证结果

- `pyright` 类型检查：0 errors, 0 warnings
- OpenAI 相关单元测试（47 个）：全部通过
