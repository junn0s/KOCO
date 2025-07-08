import logging
from langsmith import traceable
from src.solchat.core.v2.prompt_templates_v2 import SOLUTION_PROMPT
from src.solchat.adapters.v2.llm_client_v2 import generate_solution
from src.solchat.schemas.v2.solution_schema_v2 import SolutionRequest, SolutionResponse
from src.solchat.core.vector_store import load_vectorstore

# 로깅
logger = logging.getLogger(__name__)

# FAISS에서 검색 가능하도록 retriever 생성
retriever = load_vectorstore().as_retriever()

@traceable(run_type="retriever")
async def retrieve_docs(query: str):
    return await retriever.ainvoke(query)

# 문제 요청을 기반으로 해설 생성하는 서비스 함수
@traceable
async def explain_solution(req: SolutionRequest) -> SolutionResponse:
    # 문제 설명 기반으로 관련 문서 검색
    query = "query: " + " ".join(req.algorithm)
    # query = "query: " + req.description
    docs = await retrieve_docs(query)
    context = "\n\n".join(d.page_content[:500] for d in docs)  # 길이 제한

    # 프롬프트 템플릿에 문제 정보 삽입
    prompt = SOLUTION_PROMPT.invoke(
        {
            "problem_number": req.problem_number,
            "title":          req.title,
            "description":    req.description,
            "input":          req.input,
            "output":         req.output,
            "input_example":  req.input_example,
            "output_example": req.output_example,
            "context":        context
        }
    )
    result = generate_solution(prompt.text) # 요부분!
    return result