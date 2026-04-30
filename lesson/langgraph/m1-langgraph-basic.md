
### 시나리오 ###
영어 원문을 받아서 → translate 노드가 한국어로 번역 → summarize 노드가 한 줄 요약 → 최종 상태 반환.

### 샘플코드 ###
```
"""
실습 1-2: 번역 → 요약 2-노드 파이프라인
"""
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class State(TypedDict):
    original: str        # 영어 원문
    translated: str      # 한국어 번역
    summary: str         # 한 줄 요약


def translate(state: State) -> dict:
    # TODO: state["original"]을 한국어로 번역해 translated에 담아 반환
    ...




def summarize(state: State) -> dict:
    # TODO: state["translated"]를 한 문장으로 요약해 summary에 담아 반환
    ...


def build_graph():
    builder = StateGraph(State)
    # TODO: translate, summarize 노드 추가
    # TODO: START → translate → summarize → END 로 연결
    return builder.compile()


if __name__ == "__main__":
    graph = build_graph()
    text = (
        "LangGraph is a low-level framework for building stateful, "
        "multi-actor applications with LLMs. It extends LangChain "
        "with the ability to coordinate multiple chains across "
        "multiple steps of computation in a cyclic manner."
    )
    result = graph.invoke({"original": text})
    for k, v in result.items():
        print(f"[{k}]\n{v}\n")

```
