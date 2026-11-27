"""Centralized OpenAI-compatible chat completions client with strict origin auth."""
import json
import os
import urllib.parse
import urllib.request

OPENROUTER_HOSTS = frozenset({"openrouter.ai", "api.openrouter.ai"})
OPENAI_HOSTS = frozenset({"api.openai.com"})


def normalize_chat_endpoint(base: str) -> str:
    """Normalize a base URL or full endpoint to a valid chat completions URL.

    Examples:
      http://127.0.0.1:8000            -> http://127.0.0.1:8000/v1/chat/completions
      http://127.0.0.1:8000/v1         -> http://127.0.0.1:8000/v1/chat/completions
      https://openrouter.ai/api/v1     -> https://openrouter.ai/api/v1/chat/completions
      https://openrouter.ai/api/v1/chat/completions -> unchanged
    """
    if not isinstance(base, str) or not base.strip():
        raise ValueError("Endpoint base URL must be a non-empty string")
    raw = base.strip()
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"Invalid endpoint URL: {base!r}")

    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        norm_path = path
    elif path.endswith("/v1"):
        norm_path = f"{path}/chat/completions"
    else:
        norm_path = f"{path}/v1/chat/completions"

    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, norm_path, "", "", ""))


def is_official_openrouter_origin(url: str) -> bool:
    """Return True only for HTTPS URLs targeting an official OpenRouter host."""
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and host in OPENROUTER_HOSTS


def is_official_openai_origin(url: str) -> bool:
    """Return True only for HTTPS URLs targeting the official OpenAI host."""
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and host in OPENAI_HOSTS


def resolve_auth_token(url: str, api_key: str | None = None, api_key_env: str | None = None) -> str | None:
    """Resolve Bearer token with strict provider origin validation.

    Provider-specific environment keys (OPENROUTER_API_KEY, OPENAI_API_KEY) are only
    attached automatically when the parsed URL strictly matches that provider's
    official HTTPS origin — never by substring matching across the full URL.
    """
    if api_key is not None:
        token = api_key.strip()
        return token or None
    if api_key_env is not None:
        token = (os.environ.get(api_key_env) or "").strip()
        return token or None
    if is_official_openrouter_origin(url):
        token = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
        return token or None
    if is_official_openai_origin(url):
        token = (os.environ.get("OPENAI_API_KEY") or "").strip()
        return token or None
    return None


def build_chat_request(
    base: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int = 900,
    temperature: float = 0.3,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> urllib.request.Request:
    """Construct a normalized, properly authenticated urllib Request."""
    endpoint = normalize_chat_endpoint(base)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if is_official_openrouter_origin(endpoint):
        payload["reasoning"] = {"enabled": False}
        payload["provider"] = {"quantizations": ["fp8", "int8"]}

    headers = {"Content-Type": "application/json"}
    token = resolve_auth_token(endpoint, api_key=api_key, api_key_env=api_key_env)
    if token:
        headers["Authorization"] = f"Bearer {token}"

    return urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )


def chat_completion(
    base: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int = 900,
    temperature: float = 0.3,
    timeout: int = 180,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> str:
    """Send a chat completion request and return the non-null assistant string content."""
    req = build_chat_request(
        base,
        model,
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        api_key=api_key,
        api_key_env=api_key_env,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    body = json.loads(raw)
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Malformed chat completion response: {exc}") from exc
    if not isinstance(content, str):
        raise ValueError(f"Expected string message content, got {type(content).__name__}")
    return content
