from core.llm_client import LLMClient
from core.rag_engine import RAGEngine
from core.config import Config


class RAGService:
    def __init__(self):
        self.llm = LLMClient()
        self.engine = RAGEngine()
        self.engine.build_vector_db()

    def chat(self, user_input: str, company_filter: str = None) -> str:
        answer, _ = self.chat_with_sources(user_input, company_filter=company_filter)
        return answer

    def chat_with_sources(
        self,
        user_input: str,
        history: list[dict] = None,
        company_filter: str = None,
    ) -> tuple[str, list[str]]:
        """
        执行完整 RAG 流程：改写 → 混合检索 → Reranker → 生成。

        返回：(answer, sources)
          - answer: LLM 生成的回答
          - sources: 引用的研报来源列表
        """
        history = history or []

        # 1. Query 改写
        rewritten = self.engine.rewrite_query(user_input, self.llm)

        # 2. 元数据过滤（可选）
        chroma_filter = {"company": company_filter} if company_filter else None

        # 3. 混合检索
        candidates = self.engine.hybrid_search(rewritten, filter=chroma_filter)

        # 4. Reranker 精排
        docs = self.engine.rerank(rewritten, candidates)

        # 5. 构建上下文
        context = "\n\n---\n\n".join(docs)

        # 6. 获取来源信息
        sources = self.engine.get_source_info(docs)

        # 7. 构建消息（含对话历史）
        messages = list(history)
        messages.append({
            "role": "user",
            "content": f"参考研报资料：\n{context}\n\n用户问题：{user_input}",
        })

        # 8. 生成回答
        reply = self.llm.send_message(
            messages=messages,
            system=Config.RAG_SYSTEM_PROMPT,
        )

        return reply, sources
