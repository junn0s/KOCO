import gradio as gr
import asyncio
import httpx
import uuid, random

with gr.Blocks() as demo:
    gr.Markdown("## 💬 챗봇 테스트 Gradio")

    session_id = gr.State("")
    chat_history = gr.State([])
    summary_buffer = gr.State([])         # 최근 10개 메시지를 저장
    summary_text = gr.State("")           # 직전 요약 (string)
    message_count = gr.State(0)           # 누적 메시지 개수

    mode_selector = gr.Radio(
        choices=["feedback", "interview"],
        value="feedback",
        label="💡 모드 선택"
    )
    mode = gr.State("feedback")

    # 사용자가 선택한 값을 상태값 mode에 저장
    mode_selector.change(fn=lambda m: m, inputs=mode_selector, outputs=mode)

    # 문제 정보 입력
    with gr.Row():
        title = gr.Textbox(label="문제 제목", value="A + B")
        language = gr.Dropdown(choices=["python", "cpp", "java"], value="python", label="코드 언어")

    description = gr.Textbox(label="문제 설명", value="두 정수 A와 B를 입력받아 출력하는 문제")
    input_desc = gr.Textbox(label="입력 설명", value="두 정수 A, B (0 < A, B < 10)")
    output_desc = gr.Textbox(label="출력 설명", value="A + B 출력")
    input_ex = gr.Textbox(label="입력 예시", value="1 2")
    output_ex = gr.Textbox(label="출력 예시", value="3")
    code = gr.Code(label="사용자 코드", value="a, b = map(int, input().split())\nprint(a + b)")
    mode = gr.State("feedback")

    start_btn = gr.Button("Start")
    output_box = gr.Markdown(label="📨 첫 피드백 응답")

    followup_input = gr.Textbox(label="후속 질문")
    followup_btn = gr.Button("질문하기")
    followup_output = gr.Markdown(label="📨 후속 응답")

    # 💡 전체 히스토리 시각화 (추가 권장)
    chat_display = gr.Markdown(label="🧾 대화 히스토리")

    with gr.Accordion("📘 요약 보기", open=False):
        summary_display = gr.Markdown(label="🔎 현재 요약 내용")

    # ✅ summary_text 값이 바뀔 때마다 요약 표시 영역 업데이트
    summary_text.change(fn=lambda s: s, inputs=summary_text, outputs=summary_display)

    async def start_chat(
        title, description, input_desc, output_desc, input_ex, output_ex, language, code, mode
    ):
        output = ""
        session_id = random.randint(100000, 999999)

        payload = {
            "sessionId": session_id,
            "problemNumber": 1000,
            "title": title,
            "description": description,
            "inputDescription": input_desc,
            "outputDescription": output_desc,
            "inputExample": input_ex,
            "outputExample": output_ex,
            "codeLanguage": language,
            "code": code,
        }

        url = (
            "http://localhost:8000/api/ai/v2/feedback/start"
            if mode == "feedback"
            else "http://localhost:8000/api/ai/v2/interview/start"
        )

        history = []
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=payload) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        content = line.replace("data: ", "")
                        output += content
                        # yield 모든 outputs을 한번에: markdown, history, session_id
                        yield output, history, session_id

        # 마지막 출력
        history = [
            {"role": "user", "content": code},
            {"role": "assistant", "content": output}
        ]
        yield output, history, session_id


    start_btn.click(
        start_chat,
        inputs=[title, description, input_desc, output_desc, input_ex, output_ex, language, code, mode],
        outputs=[output_box, chat_history, session_id]
    )

    async def answer_chat(user_msg, history, session_id, mode, count, buffer, prev_summary):
        if not session_id:
            yield "먼저 Start를 눌러 세션을 시작하세요.", history, session_id, count, buffer, prev_summary
            return

        # 사용자 메시지 추가
        history.append({"role": "user", "content": user_msg})
        buffer.append({"role": "user", "content": user_msg})
        count += 1

        # 요약 생성 조건 확인
        summary_to_use = prev_summary
        if count >= 10:
            summary_payload = [{
                "sessionId": session_id,
                "messages": buffer
            }]
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post("http://localhost:8000/api/ai/v2/summary", json=summary_payload)
                summary_to_use = resp.json().get("summary", "")

            count = 0
            buffer = []  # 메시지 버퍼 초기화

        # 최근 10개 메시지만 추출
        messages_to_send = history[-10:]

        payload = {
            "sessionId": session_id,
            "messages": messages_to_send,
            "summary": summary_to_use
        }

        url = (
            "http://localhost:8000/api/ai/v2/feedback/answer"
            if mode == "feedback"
            else "http://localhost:8000/api/ai/v2/interview/answer"
        )

        async with httpx.AsyncClient(timeout=None) as client:
            output = ""
            async with client.stream("POST", url, json=payload) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        content = line.replace("data: ", "")
                        output += content

        # assistant 응답 추가
        history.append({"role": "assistant", "content": output})
        buffer.append({"role": "assistant", "content": output})
        count += 1

        yield output, history, session_id, count, buffer, summary_to_use

    followup_btn.click(
        answer_chat,
        inputs=[followup_input, chat_history, session_id, mode, message_count, summary_buffer, summary_text],
        outputs=[followup_output, chat_history, session_id, message_count, summary_buffer, summary_text]
)


    # 💬 대화 히스토리 실시간 표시
    def render_chat(history):  # history: List[Dict]
        return "\n\n".join([f"**{m['role']}**: {m['content']}" for m in history])

    chat_history.change(render_chat, inputs=chat_history, outputs=chat_display)

if __name__ == "__main__":
    demo.launch()