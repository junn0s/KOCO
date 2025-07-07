import os
import logging
from openai import OpenAI
from dotenv import load_dotenv
from src.solchat.config import settings
from langsmith import traceable
from src.solchat.core.llm_key_manager import APIKeyManager
from src.solchat.core.utils.stream_utils import wrap_stream_response

load_dotenv()
logger = logging.getLogger(__name__)

solar_key_manager = APIKeyManager(os.getenv("SOLAR_API_KEYS").split(","))

client = OpenAI(
    api_key=solar_key_manager.next_key(),
    base_url="https://api.upstage.ai/v1"
)

@traceable(name="interview-llm", tags=["interview", "llm"])
async def call_agent(prompt: str, stream: bool = True, max_tokens: int = None, session_id: str = None):
    max_tokens = max_tokens or settings.max_tokens_chat  # fallback
    try:
        client = OpenAI(
            api_key=solar_key_manager.next_key(),
            base_url="https://api.upstage.ai/v1"
        )

        response = client.chat.completions.create(
            model=settings.model_chat,
            temperature=settings.temperature_chat,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            stream=stream
        )

        if stream:
            return wrap_stream_response(response, session_id=session_id, prompt=prompt)
        else:
            return response.choices[0].message.content

    except Exception as e:
        logger.error("Interview Agent 호출 실패", exc_info=True)
        raise RuntimeError("인터뷰 에이전트 응답 생성 실패") from e