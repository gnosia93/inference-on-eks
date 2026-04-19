import os
from functools import lru_cache
from mcp.server.fastmcp import FastMCP
from RAGSearch import RAGSearch

mcp = FastMCP("rag-search")

@lru_cache(maxsize=1)
def get_rag() -> RAGSearch:
    return RAGSearch(
        host=os.getenv("MILVUS_HOST", "milvus.milvus.svc.cluster.local"),
        port=os.getenv("MILVUS_PORT", "19530"),
        collection_name=os.getenv("MILVUS_COLLECTION", "papers"),
        bedrock_model_id=os.getenv(
            "BEDROCK_MODEL_ID",
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
        ),
        aws_region=os.getenv("AWS_REGION", "us-west-2"),
    )

@mcp.tool()
def search_papers(query: str, top_k: int = 20, top_n: int = 5) -> dict:
    """논문 벡터 DB에서 질의와 관련된 내용을 검색하고 LLM으로 답변 생성"""
    rag = get_rag()
    result = rag.query(query, top_k=top_k, top_n=top_n)
    contexts = [
        {
            "doc_name": c["doc_name"],
            "page": c["page"],
            "rerank_score": round(c["rerank_score"], 3),
            "text": c["text"][:300] + ("..." if len(c["text"]) > 300 else ""),
        }
        for c in result["contexts"]
    ]
    return {"answer": result["answer"], "contexts": contexts}

@mcp.tool()
def retrieve_only(query: str, top_k: int = 10) -> list[dict]:
    """LLM 호출 없이 벡터 검색 + 재순위 결과만 반환"""
    rag = get_rag()
    hits = rag.retrieve(query, top_k=top_k)
    reranked = rag.rerank(query, hits, top_n=top_k)
    return [
        {
            "doc_name": h["doc_name"],
            "page": h["page"],
            "score": round(h["score"], 3),
            "rerank_score": round(h["rerank_score"], 3),
            "text": h["text"],
        }
        for h in reranked
    ]

@mcp.tool()
def health() -> dict:
    """헬스체크용"""
    return {"status": "ok"}

if __name__ == "__main__":
    # SSE(HTTP) 모드로 실행. 0.0.0.0:8000에서 수신
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
