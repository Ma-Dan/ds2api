"""
OpenAI-compatible API routes.
Implements /v1/chat/completions and related endpoints.
"""
import json
import time
import uuid
from typing import Optional, List, Dict, Any, AsyncGenerator
from fastapi import APIRouter, HTTPException, Header, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from .config import config_store
from .account_pool import account_pool
from .deepseek_client import deepseek_client, RequestAuth


router = APIRouter(prefix="/v1", tags=["OpenAI API"])


# Request/Response Models
class ChatMessage(BaseModel):
    """Chat message model."""
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None


class ToolFunction(BaseModel):
    """Tool function definition."""
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict] = None


class Tool(BaseModel):
    """Tool definition."""
    type: str = "function"
    function: ToolFunction


class ChatCompletionRequest(BaseModel):
    """Chat completion request model."""
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Any] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    user: Optional[str] = None


class ModelInfo(BaseModel):
    """Model information."""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "deepseek"


class ModelsResponse(BaseModel):
    """Models list response."""
    object: str = "list"
    data: List[ModelInfo]


# Supported models
SUPPORTED_MODELS = [
    "deepseek-chat",
    "deepseek-reasoner",
    "deepseek-chat-search",
    "deepseek-reasoner-search",
]


def resolve_model(model: str) -> tuple:
    """
    Resolve model name and extract features.
    Returns (actual_model, thinking, search).
    """
    # Apply alias mapping
    actual_model = config_store.resolve_model(model)

    thinking = False
    search = False

    # Extract features from model name
    if "reasoner" in actual_model:
        thinking = True

    if "search" in actual_model:
        search = True

    # Remove feature suffixes to get base model
    actual_model = actual_model.replace("-search", "").replace("-reasoner", "chat")
    if actual_model == "deepseek-reasoner":
        actual_model = "deepseek-reasoner"

    return actual_model, thinking, search


async def verify_api_key(authorization: Optional[str] = None) -> str:
    """Verify API key and return whether it's a managed key."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    # Extract Bearer token
    if authorization.startswith("Bearer "):
        api_key = authorization[7:]
    else:
        api_key = authorization

    # Check if it's a managed key
    if config_store.is_valid_key(api_key):
        return "managed"

    # Treat as direct DeepSeek token
    return "direct"


async def get_auth_for_request(
    api_key_type: str,
    authorization: Optional[str] = None,
    target_account: Optional[str] = None
) -> RequestAuth:
    """Get authentication for the request."""
    if api_key_type == "direct":
        # Direct token mode
        token = authorization[7:] if authorization.startswith("Bearer ") else authorization
        return RequestAuth(
            deepseek_token=token,
            account_id="direct",
            use_config_token=False
        )

    # Managed mode - acquire from pool
    auth = await account_pool.acquire(target_account)
    if not auth:
        raise HTTPException(
            status_code=429,
            detail="No available account slots. Please try again later."
        )

    return auth


def build_deepseek_messages(messages: List[ChatMessage]) -> List[Dict]:
    """Convert OpenAI messages to DeepSeek format."""
    ds_messages = []

    for msg in messages:
        ds_msg = {"role": msg.role}

        if msg.content:
            ds_msg["content"] = msg.content

        if msg.tool_calls:
            ds_msg["tool_calls"] = [tc.dict() if hasattr(tc, 'dict') else tc for tc in msg.tool_calls]

        if msg.tool_call_id:
            ds_msg["tool_call_id"] = msg.tool_call_id

        ds_messages.append(ds_msg)

    return ds_messages


def generate_completion_id() -> str:
    """Generate a unique completion ID."""
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


async def stream_response(
    session_id: str,
    auth: RequestAuth,
    request: ChatCompletionRequest,
    model: str,
    thinking: bool,
    search: bool
) -> AsyncGenerator[str, None]:
    """Generate streaming SSE response."""
    completion_id = generate_completion_id()
    created = int(time.time())

    # Get PoW
    pow_header = await deepseek_client.get_pow(auth)
    if not pow_header:
        yield f"data: {json.dumps({'error': {'message': 'Failed to get PoW', 'type': 'server_error'}})}\n\n"
        return

    # Convert messages
    ds_messages = build_deepseek_messages(request.messages)

    # Convert tools if present
    tools = None
    if request.tools:
        tools = [{"type": t.type, "function": t.function.dict()} for t in request.tools]

    # Stream from DeepSeek
    current_type = "thinking" if thinking else "text"
    accumulated_text = ""
    accumulated_thinking = ""

    async for event in deepseek_client.chat_completion(
        auth=auth,
        session_id=session_id,
        messages=ds_messages,
        model=model,
        stream=True,
        thinking=thinking,
        search=search,
        pow_header=pow_header,
        tools=tools
    ):
        if event.get("error"):
            yield f"data: {json.dumps({'error': {'message': event.get('message', 'Unknown error')}})}\n\n"
            return

        if event.get("done"):
            # Send finish chunk
            finish_reason = "stop"
            delta = {}

            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': request.model, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish_reason}]})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Parse DeepSeek event
        # DeepSeek uses a specific SSE format:
        # - First event: v.response.fragments[0].content contains initial text
        # - Subsequent events: v is a string to append, o is operation (APPEND)
        v = event.get("v")
        operation = event.get("o", "")  # APPEND or other operations
        p = event.get("p", "")  # Path like "response/fragments/-1/content"

        # Handle different event formats
        if isinstance(v, dict):
            # Initial event with response structure
            response = v.get("response", {})
            fragments = response.get("fragments", [])

            for fragment in fragments:
                if not isinstance(fragment, dict):
                    continue
                content = fragment.get("content", "")
                if content and operation != "APPEND":
                    # This is initial content
                    accumulated_text = content
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

        elif isinstance(v, str):
            # Incremental content (APPEND operation)
            if v:
                delta_text = v
                accumulated_text += delta_text

                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": delta_text},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"


@router.get("/models")
async def list_models():
    """List available models."""
    models = []
    for model_id in SUPPORTED_MODELS:
        models.append(ModelInfo(id=model_id))

    # Add aliases
    for alias, target in config_store.config.model_aliases.items():
        if alias not in SUPPORTED_MODELS:
            models.append(ModelInfo(id=alias))

    return ModelsResponse(data=models)


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get model information."""
    resolved = config_store.resolve_model(model_id)

    if resolved not in SUPPORTED_MODELS and model_id not in config_store.config.model_aliases:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    return ModelInfo(id=model_id)


@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None),
    x_ds2_target_account: Optional[str] = Header(None)
):
    """Handle chat completion requests."""
    # Verify API key
    api_key_type = await verify_api_key(authorization)

    # Get authentication
    auth = await get_auth_for_request(api_key_type, authorization, x_ds2_target_account)

    try:
        # Resolve model
        model, thinking, search = resolve_model(request.model)

        # Create session
        session_id = await deepseek_client.create_session(auth)
        if not session_id:
            raise HTTPException(
                status_code=401,
                detail="Failed to create session. Token may be invalid."
            )

        if request.stream:
            # Streaming response
            return StreamingResponse(
                stream_response(session_id, auth, request, model, thinking, search),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            # Non-streaming response
            pow_header = await deepseek_client.get_pow(auth)
            if not pow_header:
                raise HTTPException(status_code=500, detail="Failed to get PoW")

            # Convert messages
            ds_messages = build_deepseek_messages(request.messages)

            # Convert tools
            tools = None
            if request.tools:
                tools = [{"type": t.type, "function": t.function.dict()} for t in request.tools]

            # Collect all events
            accumulated_text = ""
            accumulated_thinking = ""
            finish_reason = "stop"

            async for event in deepseek_client.chat_completion(
                auth=auth,
                session_id=session_id,
                messages=ds_messages,
                model=model,
                stream=True,
                thinking=thinking,
                search=search,
                pow_header=pow_header,
                tools=tools
            ):
                if event.get("error"):
                    raise HTTPException(
                        status_code=event.get("status_code", 500),
                        detail=event.get("message", "Unknown error")
                    )

                if event.get("done"):
                    break

                # Parse DeepSeek event (same format as streaming)
                v = event.get("v")
                operation = event.get("o", "")

                if isinstance(v, dict):
                    # Initial event with response structure
                    response = v.get("response", {})
                    fragments = response.get("fragments", [])
                    for fragment in fragments:
                        if not isinstance(fragment, dict):
                            continue
                        content = fragment.get("content", "")
                        if content and operation != "APPEND":
                            accumulated_text = content

                elif isinstance(v, str):
                    # Incremental content (APPEND operation)
                    accumulated_text += v

            # Build response
            completion_id = generate_completion_id()
            created = int(time.time())

            response = {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": accumulated_text
                    },
                    "finish_reason": finish_reason
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }

            # Add reasoning content if thinking was enabled
            if thinking and accumulated_thinking:
                response["choices"][0]["message"]["reasoning_content"] = accumulated_thinking

            return JSONResponse(content=response)

    finally:
        # Release account slot
        if auth.use_config_token:
            account_pool.release(auth)

        # Auto-delete session if configured
        if config_store.config.auto_delete.mode != "none":
            try:
                if config_store.config.auto_delete.mode == "single":
                    await deepseek_client.delete_session(auth, session_id)
                elif config_store.config.auto_delete.mode == "all":
                    await deepseek_client.delete_all_sessions(auth)
            except Exception as e:
                print(f"Auto-delete error: {e}")


@router.post("/embeddings")
async def embeddings(
    request: Request,
    authorization: Optional[str] = Header(None)
):
    """Handle embedding requests (deterministic/mock implementation)."""
    # Verify API key
    await verify_api_key(authorization)

    body = await request.json()
    input_text = body.get("input", "")

    if isinstance(input_text, list):
        texts = input_text
    else:
        texts = [input_text]

    # Deterministic embedding based on text hash
    # This is a placeholder - real embeddings would use a proper model
    embeddings_data = []

    for i, text in enumerate(texts):
        # Create deterministic vector from text
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        # Create a 1536-dim vector (OpenAI embedding size)
        embedding = []
        for j in range(1536):
            byte_idx = j % len(h)
            embedding.append(float(h[byte_idx]) / 255.0 - 0.5)

        embeddings_data.append({
            "object": "embedding",
            "index": i,
            "embedding": embedding
        })

    return {
        "object": "list",
        "data": embeddings_data,
        "model": body.get("model", "text-embedding-ada-002"),
        "usage": {
            "prompt_tokens": sum(len(t.split()) for t in texts),
            "total_tokens": sum(len(t.split()) for t in texts)
        }
    }
