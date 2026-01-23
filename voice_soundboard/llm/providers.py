"""
LLM Provider integrations.

Supports multiple LLM backends:
- Ollama (local)
- OpenAI (API)
- vLLM (self-hosted)
- Mock (testing)
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Dict, List, Optional, Any, Union
import time


class ProviderType(Enum):
    """Supported LLM provider types."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    VLLM = "vllm"
    MOCK = "mock"


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""

    # Model settings
    model: str = "llama3.2"
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.9
    top_k: int = 40

    # Connection settings
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: float = 60.0

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0

    # Streaming settings
    stream: bool = True


@dataclass
class LLMResponse:
    """Response from an LLM."""

    content: str
    model: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    latency_ms: float = 0.0

    # Metadata
    provider: str = ""
    raw_response: Optional[Dict[str, Any]] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._initialized = False

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Get the provider type."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a complete response."""
        pass

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream tokens as they're generated."""
        pass

    async def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> LLMResponse:
        """
        Chat with message history.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": "..."}
        """
        # Default implementation: concatenate messages into prompt
        prompt_parts = []
        system = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system = content
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt = "\n".join(prompt_parts)
        return await self.generate(prompt, system=system, **kwargs)

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream chat response."""
        prompt_parts = []
        system = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system = content
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt = "\n".join(prompt_parts)
        async for token in self.stream(prompt, system=system, **kwargs):
            yield token


class OllamaProvider(LLMProvider):
    """
    Ollama LLM provider for local models.

    Requires Ollama to be running locally.
    """

    DEFAULT_URL = "http://localhost:11434"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OLLAMA

    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        self.base_url = self.config.base_url or self.DEFAULT_URL

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate using Ollama API."""
        try:
            import aiohttp
        except ImportError:
            raise ImportError("aiohttp required for Ollama provider: pip install aiohttp")

        start_time = time.time()

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": kwargs.get("model", self.config.model),
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "top_k": kwargs.get("top_k", self.config.top_k),
            },
        }

        if system:
            payload["system"] = system

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            ) as response:
                response.raise_for_status()
                data = await response.json()

        latency_ms = (time.time() - start_time) * 1000

        return LLMResponse(
            content=data.get("response", ""),
            model=data.get("model", self.config.model),
            finish_reason="stop",
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
            latency_ms=latency_ms,
            provider="ollama",
            raw_response=data,
        )

    async def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream tokens from Ollama."""
        try:
            import aiohttp
        except ImportError:
            raise ImportError("aiohttp required for Ollama provider: pip install aiohttp")

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": kwargs.get("model", self.config.model),
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "top_k": kwargs.get("top_k", self.config.top_k),
            },
        }

        if system:
            payload["system"] = system

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            ) as response:
                response.raise_for_status()

                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode("utf-8"))
                            token = data.get("response", "")
                            if token:
                                yield token
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue


class OpenAIProvider(LLMProvider):
    """
    OpenAI API provider.

    Works with OpenAI API and compatible endpoints (Azure, local servers).
    """

    DEFAULT_URL = "https://api.openai.com/v1"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OPENAI

    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        self.base_url = self.config.base_url or self.DEFAULT_URL
        if not self.config.api_key:
            import os
            self.config.api_key = os.environ.get("OPENAI_API_KEY")

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate using OpenAI API."""
        try:
            import aiohttp
        except ImportError:
            raise ImportError("aiohttp required for OpenAI provider: pip install aiohttp")

        start_time = time.time()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": kwargs.get("model", self.config.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "stream": False,
        }

        headers = {
            "Content-Type": "application/json",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            ) as response:
                response.raise_for_status()
                data = await response.json()

        latency_ms = (time.time() - start_time) * 1000

        choice = data.get("choices", [{}])[0]
        usage = data.get("usage", {})

        return LLMResponse(
            content=choice.get("message", {}).get("content", ""),
            model=data.get("model", self.config.model),
            finish_reason=choice.get("finish_reason"),
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
            latency_ms=latency_ms,
            provider="openai",
            raw_response=data,
        )

    async def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream tokens from OpenAI API."""
        try:
            import aiohttp
        except ImportError:
            raise ImportError("aiohttp required for OpenAI provider: pip install aiohttp")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": kwargs.get("model", self.config.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": True,
        }

        headers = {
            "Content-Type": "application/json",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            ) as response:
                response.raise_for_status()

                async for line in response.content:
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue


class VLLMProvider(LLMProvider):
    """
    vLLM provider for self-hosted high-performance inference.

    Uses OpenAI-compatible API.
    """

    DEFAULT_URL = "http://localhost:8000/v1"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.VLLM

    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        self.base_url = self.config.base_url or self.DEFAULT_URL
        # vLLM uses OpenAI-compatible API, so we can reuse OpenAI provider
        self._openai = OpenAIProvider(LLMConfig(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            base_url=self.base_url,
            api_key=self.config.api_key or "EMPTY",  # vLLM often doesn't need a key
        ))

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate using vLLM."""
        response = await self._openai.generate(prompt, system, **kwargs)
        response.provider = "vllm"
        return response

    async def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream tokens from vLLM."""
        async for token in self._openai.stream(prompt, system, **kwargs):
            yield token


class MockLLMProvider(LLMProvider):
    """
    Mock LLM provider for testing.

    Returns predefined responses or generates simple patterns.
    """

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.MOCK

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        responses: Optional[Dict[str, str]] = None,
        default_response: str = "This is a mock response.",
        token_delay_ms: float = 50.0,
    ):
        super().__init__(config)
        self.responses = responses or {}
        self.default_response = default_response
        self.token_delay_ms = token_delay_ms

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate mock response."""
        start_time = time.time()

        # Check for matching response
        content = self.default_response
        for pattern, response in self.responses.items():
            if pattern.lower() in prompt.lower():
                content = response
                break

        # Simulate some latency
        await asyncio.sleep(0.1)

        latency_ms = (time.time() - start_time) * 1000

        return LLMResponse(
            content=content,
            model="mock",
            finish_reason="stop",
            usage={
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(content.split()),
            },
            latency_ms=latency_ms,
            provider="mock",
        )

    async def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream mock tokens."""
        # Get response
        response = await self.generate(prompt, system, **kwargs)
        content = response.content

        # Stream word by word
        words = content.split()
        for i, word in enumerate(words):
            await asyncio.sleep(self.token_delay_ms / 1000)
            if i < len(words) - 1:
                yield word + " "
            else:
                yield word


def create_provider(
    provider_type: Union[str, ProviderType],
    config: Optional[LLMConfig] = None,
    **kwargs,
) -> LLMProvider:
    """
    Factory function to create LLM providers.

    Args:
        provider_type: Provider type (string or enum)
        config: LLM configuration
        **kwargs: Additional provider-specific arguments

    Returns:
        LLMProvider instance
    """
    if isinstance(provider_type, str):
        provider_type = ProviderType(provider_type.lower())

    if provider_type == ProviderType.OLLAMA:
        return OllamaProvider(config)
    elif provider_type == ProviderType.OPENAI:
        return OpenAIProvider(config)
    elif provider_type == ProviderType.VLLM:
        return VLLMProvider(config)
    elif provider_type == ProviderType.MOCK:
        return MockLLMProvider(config, **kwargs)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
