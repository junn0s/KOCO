# 전체 오케스트레이션 함수

import logging
import os

from src.crawler.v2.post_client_v2 import post_to_backend
from src.crawler.v2.request_mapper_v2 import to_solution_request
from src.crawler.v2.solution_generater_v2 import generate_explanation

logger = logging.getLogger(__name__)


def _load_generation_attempts() -> int:
    try:
        return max(1, int(os.getenv("SOLUTION_GENERATION_ATTEMPTS", "2")))
    except ValueError:
        return 2


DEFAULT_GENERATION_ATTEMPTS = _load_generation_attempts()


# pydantic으로 변환 -> 해설 생성 -> 백엔드에 포스팅
async def crawl_generate_post(
    problem_data: dict,
    language: str = "python",
    max_generation_attempts: int = DEFAULT_GENERATION_ATTEMPTS,
) -> bool:
    req = to_solution_request(problem_data, language)
    problem_id = problem_data["problem_number"]

    for attempt in range(1, max_generation_attempts + 1):
        try:
            response = await generate_explanation(req)
        except Exception as exc:
            logger.warning(
                "[%s] 해설 생성 %d/%d 실패: %s",
                problem_id,
                attempt,
                max_generation_attempts,
                exc,
                exc_info=True,
            )
            continue

        if response is None:
            logger.warning(
                "[%s] 해설 생성 %d/%d 실패: LLM 응답 없음",
                problem_id,
                attempt,
                max_generation_attempts,
            )
            continue

        if not hasattr(response, "model_dump"):
            logger.warning(
                "[%s] 해설 생성 %d/%d 실패: Pydantic 응답 아님 (%s)",
                problem_id,
                attempt,
                max_generation_attempts,
                type(response),
            )
            continue

        return post_to_backend(problem_id, response)

    logger.error("[%s] 해설 생성 재시도 모두 실패", problem_id)
    return False
