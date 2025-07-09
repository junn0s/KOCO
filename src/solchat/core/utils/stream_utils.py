import asyncio
import re
from typing import AsyncGenerator
from langsmith.client import Client

client = Client()

async def wrap_stream_response(response, session_id: str = None, prompt: str = "") -> AsyncGenerator[str, None]:
    line_buf = ""
    full_output = ""

    try:
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                content = delta.content
                line_buf += content
                full_output += content

                # 줄 단위로 먼저 처리
                while "\n" in line_buf:
                    line, line_buf = line_buf.split("\n", 1)
                    print(f"[완성된 줄] {line}")
                    # 공백과 단어를 분리하여 모두 토큰화
                    tokens = re.findall(r"\S+|\s", line)
                    for token in tokens:
                        if token == " ":
                            yield "data:  \n\n"  # 띄어쓰기 명시
                        else:
                            yield f"data: {token}\n\n"
                    # 줄바꿈 명시
                    yield "data: \\n\n\n"

                await asyncio.sleep(0)

        # 남은 줄 처리
        if line_buf.strip():
            tokens = re.findall(r"\S+|\s", line_buf)
            for token in tokens:
                if token == " ":
                    yield "data:  \n\n"
                else:
                    yield f"data: {token}\n\n"

        client.create_run(
            name="Streamed Response",
            run_type="llm",
            inputs={"prompt": prompt},
            outputs={"output": full_output},
            tags=["feedback", "streamed"],
            metadata={"session_id": session_id or "unknown"}
        )

    except Exception as e:
        yield f"data: [ERROR] {str(e)}\n\n"
