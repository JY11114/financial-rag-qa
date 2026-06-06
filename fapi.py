"""
FastAPI 服务入口

启动方式：
    uvicorn fapi:app --reload --host 0.0.0.0 --port 8000

接口文档：http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from service.rag_service import RAGService
from core.config import Config

app = FastAPI(
    title="金融研报问答 API",
    description="基于 RAG 的金融研报智能问答服务（混合检索 + Reranker + 多轮对话）",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_rag: Optional[RAGService] = None
sessions: dict[str, list[dict]] = {}


@app.on_event("startup")
def startup():
    global _rag
    print("正在初始化 RAG 服务...")
    _rag = RAGService()
    print("服务就绪")


# ── 数据模型 ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"
    company: Optional[str] = None  # 元数据过滤，如 "宁德时代"

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    session_id: str
    history_length: int


# ── 接口 ──────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "rag_ready": _rag is not None}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    金融研报问答（支持多轮对话与元数据过滤）

    - query: 用户问题
    - session_id: 会话 ID，相同 ID 共享对话历史
    - company: 可选，限定检索范围（如 "宁德时代"、"贵州茅台"）
    """
    if _rag is None:
        raise HTTPException(status_code=503, detail="RAG 服务未就绪")

    history = sessions.get(req.session_id, [])

    try:
        answer, sources = _rag.chat_with_sources(
            req.query,
            history=history,
            company_filter=req.company,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答失败：{str(e)}")

    history.append({"role": "user", "content": req.query})
    history.append({"role": "assistant", "content": answer})
    sessions[req.session_id] = history[-Config.MAX_HISTORY_LENGTH:]

    return ChatResponse(
        answer=answer,
        sources=sources[:5],
        session_id=req.session_id,
        history_length=len(sessions[req.session_id]) // 2,
    )


@app.get("/sessions/{session_id}/history")
def get_history(session_id: str):
    return {"session_id": session_id, "history": sessions.get(session_id, [])}


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"message": f"会话 {session_id} 已清空"}


@app.get("/knowledge-base/stats")
def kb_stats():
    if _rag is None:
        raise HTTPException(status_code=503, detail="RAG 服务未就绪")
    engine = _rag.engine
    companies = list({m.get("company") for m in engine.chunk_metadata})
    institutions = list({m.get("institution") for m in engine.chunk_metadata})
    return {
        "total_chunks": len(engine.chunks),
        "companies": companies,
        "institutions": institutions,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fapi:app", host=Config.API_HOST, port=Config.API_PORT, reload=True)
