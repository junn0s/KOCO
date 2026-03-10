# 실행 파일 (매일 아침 10시에 실행)

# 백준 문제 크롤링 및 해설지 생성을 위한 메인 함수
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from src.core.logger import setup_logging
from src.crawler.v2.boj_crawler_v2 import (
    crawl_boj_problem_with_selenium,
    create_driver,
    login_with_cookies,
)
from src.crawler.v2.daily_crawler_v2 import (
    get_problem_ids_from_workbook,
    get_today_workbook_id,
)
from src.crawler.v2.pipeline_v2 import crawl_generate_post

setup_logging()
logger = logging.getLogger(__name__)

FAILURE_STORE_PATH = Path("logs/solution_pipeline_failures.json")


def _load_reprocess_rounds() -> int:
    try:
        return max(0, int(os.getenv("CRAWLER_REPROCESS_ROUNDS", "1")))
    except ValueError:
        return 1


MAX_REPROCESS_ROUNDS = _load_reprocess_rounds()


def load_pending_failures() -> list[int]:
    if not FAILURE_STORE_PATH.is_file():
        return []

    try:
        payload = json.loads(FAILURE_STORE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("실패 기록 파일을 읽지 못했습니다: %s", exc)
        return []

    pending_ids: list[int] = []
    for item in payload.get("failures", []):
        pid = item.get("problem_id")
        if isinstance(pid, int) and pid not in pending_ids:
            pending_ids.append(pid)
    return pending_ids


def merge_problem_ids(*problem_id_groups: list[int]) -> list[int]:
    merged: list[int] = []
    seen: set[int] = set()

    for group in problem_id_groups:
        for pid in group:
            if pid in seen:
                continue
            seen.add(pid)
            merged.append(pid)

    return merged


def build_failure(problem_id: int, stage: str, reason: str) -> dict:
    return {
        "problem_id": problem_id,
        "stage": stage,
        "reason": reason,
    }


def save_failures(failures: list[dict]) -> None:
    FAILURE_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "failures": failures,
    }
    FAILURE_STORE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def clear_failures() -> None:
    if FAILURE_STORE_PATH.exists():
        FAILURE_STORE_PATH.unlink()


async def process_one_problem(pid: int, driver):
    data = crawl_boj_problem_with_selenium(driver, pid)
    if not data["title"]:
        logger.warning(f"[{pid}] 문제 데이터 크롤링 실패: 제목이 비어있음")
        return build_failure(pid, "crawl", "문제 데이터 크롤링 실패")

    logger.info(f"[{pid}] 문제 데이터 크롤링 성공: {data['title']}")
    logger.info(f"[{pid}] 해설 생성 및 포스팅 시작…")

    if not await crawl_generate_post(data):
        logger.warning(f"[{pid}] 해설 생성 또는 전송 실패")
        return build_failure(pid, "generate_or_post", "해설 생성 또는 전송 실패")

    logger.info(f"[{pid}] 해설 생성 및 포스팅 완료")
    return None


async def process_problem_batch(pids: list[int], driver, phase: str) -> list[dict]:
    logger.info("%s 처리 시작: %d건", phase, len(pids))
    failures: list[dict] = []

    for pid in pids:
        failure = await process_one_problem(pid, driver)
        if failure is not None:
            failures.append(failure)

    logger.info("%s 처리 종료: 실패 %d건", phase, len(failures))
    return failures


async def run_pipeline_with_reprocess(pids: list[int], driver) -> list[dict]:
    failures = await process_problem_batch(pids, driver, phase="main")

    for round_idx in range(1, MAX_REPROCESS_ROUNDS + 1):
        if not failures:
            break

        failed_ids = [failure["problem_id"] for failure in failures]
        logger.info(
            "실패 건 재처리 %d/%d 시작: %s",
            round_idx,
            MAX_REPROCESS_ROUNDS,
            failed_ids,
        )
        failures = await process_problem_batch(
            failed_ids,
            driver,
            phase=f"reprocess-{round_idx}",
        )

    return failures


if __name__ == "__main__":
    GROUP_ID = 23567
    driver = create_driver()

    try:
        login_with_cookies(driver)
        today_wb_id = get_today_workbook_id(driver)
        daily_pids = get_problem_ids_from_workbook(
            driver,
            group_id=GROUP_ID,
            workbook_id=today_wb_id,
        )

        pending_failure_ids = load_pending_failures()
        if pending_failure_ids:
            logger.info(
                "이전 실패 건 %d개를 우선 재처리 목록에 포함합니다.",
                len(pending_failure_ids),
            )

        pids = merge_problem_ids(pending_failure_ids, daily_pids)
        remaining_failures = asyncio.run(run_pipeline_with_reprocess(pids, driver))

        if remaining_failures:
            save_failures(remaining_failures)
            logger.warning(
                "최종 실패 %d건을 %s 에 저장했습니다.",
                len(remaining_failures),
                FAILURE_STORE_PATH,
            )
        else:
            clear_failures()
            logger.info("남은 실패 건이 없어 실패 기록 파일을 정리했습니다.")

    except Exception as e:
        logger.error(f"파이프라인 실행 중 오류 발생: {str(e)}", exc_info=True)
        raise
    finally:
        driver.quit()
