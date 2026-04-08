"""
DeepSeek API client for authentication and chat completion.
"""
import json
import asyncio
from typing import Optional, Dict, Any, List, AsyncGenerator
from dataclasses import dataclass
import httpx

from .config import config_store, AccountConfig
from .pow import Challenge


# DeepSeek API URLs
DEEPSEEK_HOST = "chat.deepseek.com"
DEEPSEEK_LOGIN_URL = "https://chat.deepseek.com/api/v0/users/login"
DEEPSEEK_CREATE_SESSION_URL = "https://chat.deepseek.com/api/v0/chat_session/create"
DEEPSEEK_CREATE_POW_URL = "https://chat.deepseek.com/api/v0/chat/create_pow_challenge"
DEEPSEEK_COMPLETION_URL = "https://chat.deepseek.com/api/v0/chat/completion"
DEEPSEEK_CONTINUE_URL = "https://chat.deepseek.com/api/v0/chat/continue"
DEEPSEEK_DELETE_SESSION_URL = "https://chat.deepseek.com/api/v0/chat_session/delete"
DEEPSEEK_DELETE_ALL_SESSIONS_URL = "https://chat.deepseek.com/api/v0/chat_session/delete_all"

# Default headers for DeepSeek API
BASE_HEADERS = {
    "Host": "chat.deepseek.com",
    "User-Agent": "DeepSeek/1.8.0 Android/35",
    "Accept": "application/json",
    "Content-Type": "application/json",
    "x-client-platform": "android",
    "x-client-version": "1.8.0",
    "x-client-locale": "zh_CN",
    "accept-charset": "UTF-8",
}


@dataclass
class RequestAuth:
    """Authentication info for a request."""
    deepseek_token: str
    account_id: str
    use_config_token: bool = True


class DeepSeekClient:
    """Async client for DeepSeek Web API."""

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        self._max_retries = 3

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _normalize_mobile(self, raw: str) -> tuple:
        """Normalize mobile number for login."""
        s = raw.strip()
        if not s:
            return "", None

        has_plus = s.startswith("+")
        digits = "".join(c for c in s if c.isdigit())

        if not digits:
            return "", None

        # Chinese mobile with +86 or 86 prefix
        if (has_plus or digits.startswith("86")) and digits.startswith("86") and len(digits) == 13:
            return digits[2:], None

        return digits, None

    async def login(self, account: AccountConfig) -> Optional[str]:
        """Login to DeepSeek and return token."""
        client = await self._get_client()

        payload = {
            "password": account.password.strip(),
            "device_id": "deepseek_to_api",
            "os": "android"
        }

        if account.email and account.email.strip():
            payload["email"] = account.email.strip()
        elif account.mobile and account.mobile.strip():
            mobile, area_code = self._normalize_mobile(account.mobile)
            payload["mobile"] = mobile
            if area_code:
                payload["area_code"] = area_code
        else:
            raise ValueError("Missing email/mobile")

        account_id = account.email or account.mobile
        print(f"[login] Attempting login for: {account_id}")

        try:
            resp = await client.post(
                DEEPSEEK_LOGIN_URL,
                headers=BASE_HEADERS,
                json=payload
            )
            data = resp.json()

            print(f"[login] Response status: {resp.status_code}")
            print(f"[login] Response code: {data.get('code')}, msg: {data.get('msg')}")

            if data.get("code", -1) != 0:
                raise ValueError(f"Login failed: {data.get('msg', 'Unknown error')}")

            biz_data = data.get("data", {}).get("biz_data", {})
            print(f"[login] biz_code: {biz_data.get('biz_code')}, biz_msg: {biz_data.get('biz_msg')}")

            if biz_data.get("biz_code", -1) != 0:
                raise ValueError(f"Login failed: {biz_data.get('biz_msg', 'Unknown error')}")

            user = biz_data.get("biz_data", {}).get("user", {})
            token = user.get("token", "")

            if not token.strip():
                raise ValueError("Missing login token")

            print(f"[login] Login successful for: {account_id}")
            return token

        except Exception as e:
            print(f"[login] Login error for {account.email or account.mobile}: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def create_session(self, auth: RequestAuth, max_attempts: int = 3) -> Optional[str]:
        """Create a chat session and return session ID."""
        client = await self._get_client()

        for attempt in range(max_attempts):
            headers = {**BASE_HEADERS, "authorization": f"Bearer {auth.deepseek_token}"}

            try:
                resp = await client.post(
                    DEEPSEEK_CREATE_SESSION_URL,
                    headers=headers,
                    json={"agent": "chat"}
                )
                data = resp.json()

                code = data.get("code", -1)
                biz_code = data.get("data", {}).get("biz_code", -1)

                if resp.status_code == 200 and code == 0 and biz_code == 0:
                    biz_data = data.get("data", {}).get("biz_data", {})
                    session_id = biz_data.get("id") or biz_data.get("chat_session", {}).get("id")
                    if session_id:
                        return session_id

            except Exception as e:
                print(f"Create session error (attempt {attempt + 1}): {e}")

            await asyncio.sleep(1)

        return None

    async def get_pow(self, auth: RequestAuth, max_attempts: int = 3) -> Optional[str]:
        """Get PoW challenge and solve it."""
        client = await self._get_client()

        for attempt in range(max_attempts):
            headers = {**BASE_HEADERS, "authorization": f"Bearer {auth.deepseek_token}"}

            try:
                resp = await client.post(
                    DEEPSEEK_CREATE_POW_URL,
                    headers=headers,
                    json={"target_path": "/api/v0/chat/completion"}
                )
                data = resp.json()

                code = data.get("code", -1)
                msg = data.get("msg", "")

                print(f"[get_pow] Response status: {resp.status_code}, code: {code}, msg: {msg}")

                if resp.status_code != 200:
                    print(f"[get_pow] HTTP error: {resp.status_code}")
                    await asyncio.sleep(1)
                    continue

                # Check response structure
                resp_data = data.get("data", {})
                biz_code = resp_data.get("biz_code", -1)
                biz_msg = resp_data.get("biz_msg", "")

                print(f"[get_pow] biz_code: {biz_code}, biz_msg: {biz_msg}")

                if code != 0 or biz_code != 0:
                    print(f"[get_pow] API error: code={code}, biz_code={biz_code}")
                    await asyncio.sleep(1)
                    continue

                biz_data = resp_data.get("biz_data", {})
                challenge_data = biz_data.get("challenge", {})

                print(f"[get_pow] Challenge data keys: {list(challenge_data.keys()) if challenge_data else 'None'}")

                if not challenge_data:
                    print(f"[get_pow] No challenge data in response")
                    await asyncio.sleep(1)
                    continue

                # Log challenge details
                print(f"[get_pow] Challenge algorithm: {challenge_data.get('algorithm')}")
                print(f"[get_pow] Challenge difficulty: {challenge_data.get('difficulty')}")

                challenge = Challenge(challenge_data)

                print(f"[get_pow] Solving PoW challenge...")
                result = challenge.solve_and_build_header()

                if result:
                    print(f"[get_pow] PoW solved successfully")
                    return result
                else:
                    print(f"[get_pow] PoW solve failed - no solution found")

            except Exception as e:
                print(f"[get_pow] Error (attempt {attempt + 1}): {e}")
                import traceback
                traceback.print_exc()

            await asyncio.sleep(1)

        print(f"[get_pow] All attempts failed")
        return None

    async def chat_completion(
        self,
        auth: RequestAuth,
        session_id: str,
        messages: List[Dict[str, Any]],
        model: str = "deepseek-chat",
        stream: bool = True,
        thinking: bool = False,
        search: bool = False,
        pow_header: Optional[str] = None,
        tools: Optional[List[Dict]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Send chat completion request and yield SSE events.
        """
        client = await self._get_client()
        headers = {**BASE_HEADERS, "authorization": f"Bearer {auth.deepseek_token}"}

        if pow_header:
            headers["x-ds-pow-response"] = pow_header

        # Convert messages to prompt format
        prompt = self._messages_to_prompt(messages)

        # Build request payload - DeepSeek uses prompt, not messages
        payload = {
            "chat_session_id": session_id,
            "model_type": "default",  # or "reasoner" for thinking models
            "parent_message_id": None,
            "prompt": prompt,
            "ref_file_ids": [],
            "thinking_enabled": thinking,
            "search_enabled": search
        }

        if thinking:
            payload["model_type"] = "reasoner"

        async with client.stream(
            "POST",
            DEEPSEEK_COMPLETION_URL,
            headers=headers,
            json=payload
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                yield {
                    "error": True,
                    "status_code": response.status_code,
                    "message": body.decode("utf-8", errors="ignore")
                }
                return

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        yield {"done": True}
                        break
                    try:
                        data = json.loads(data_str)
                        yield data
                    except json.JSONDecodeError:
                        continue

    async def delete_session(self, auth: RequestAuth, session_id: str) -> bool:
        """Delete a chat session."""
        client = await self._get_client()
        headers = {**BASE_HEADERS, "authorization": f"Bearer {auth.deepseek_token}"}

        try:
            resp = await client.post(
                DEEPSEEK_DELETE_SESSION_URL,
                headers=headers,
                json={"chat_session_id": session_id}
            )
            return resp.status_code == 200
        except Exception as e:
            print(f"Delete session error: {e}")
            return False

    async def delete_all_sessions(self, auth: RequestAuth) -> bool:
        """Delete all chat sessions."""
        client = await self._get_client()
        headers = {**BASE_HEADERS, "authorization": f"Bearer {auth.deepseek_token}"}

        try:
            resp = await client.post(
                DEEPSEEK_DELETE_ALL_SESSIONS_URL,
                headers=headers,
                json={}
            )
            return resp.status_code == 200
        except Exception as e:
            print(f"Delete all sessions error: {e}")
            return False

    def _messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """
        Convert OpenAI-style messages to DeepSeek prompt format.
        Uses DeepSeek's special markers for different roles.
        """
        # DeepSeek role markers
        SYSTEM_MARKER = "<｜System｜>"
        USER_MARKER = "<｜User｜>"
        ASSISTANT_MARKER = "<｜Assistant｜>"
        END_SENTENCE = "<｜end▁of▁sentence｜>"
        END_INSTRUCTIONS = "<｜end▁of▁instructions｜>"

        parts = []

        for msg in messages:
            role = msg.get("role", "")
            content = self._normalize_content(msg.get("content"))

            if not content.strip():
                continue

            if role == "system":
                parts.append(f"{SYSTEM_MARKER}\n{content}{END_INSTRUCTIONS}")
            elif role == "user":
                parts.append(f"{USER_MARKER}\n{content}{END_SENTENCE}")
            elif role == "assistant":
                parts.append(f"{ASSISTANT_MARKER}\n{content}{END_SENTENCE}")
            else:
                parts.append(content)

        return "\n\n".join(parts)

    def _normalize_content(self, content: Any) -> str:
        """Normalize message content to string."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Handle multi-part content
            texts = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type", "").lower()
                    if item_type in ("text", "output_text", "input_text"):
                        text = item.get("text") or item.get("content", "")
                        if text:
                            texts.append(text)
            return "\n".join(texts)
        return str(content)


# Global client instance
deepseek_client = DeepSeekClient()
