# src/adapters/llm_client.py

import os
import logging
from google import genai
from google.genai import types
from langsmith import traceable
from src.solchat.config import settings
from dotenv import load_dotenv
from src.solchat.schemas.v2.solution_schema_v2 import SolutionResponse

# 환경변수 로드
load_dotenv()

logger = logging.getLogger(__name__)

# 1) genai 클라이언트 생성
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

@traceable(run_type="llm")
def generate_solution(prompt_text: str) -> SolutionResponse:
    try:
        response = client.models.generate_content(
            model=settings.model_solution,
            contents=prompt_text,
            config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SolutionResponse,
            temperature=settings.temperature_solution,
            max_output_tokens=settings.max_tokens_solution, 
            ),
        )
        return response.parsed
    except Exception as e:
        logger.error("Gemini API 호출 실패", exc_info=True)
        raise RuntimeError("해설 생성 중 오류가 발생했습니다.") from e