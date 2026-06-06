import os

class Config:
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    MODEL_NAME = "claude-haiku-4-5"
    MAX_TOKENS = 2048
    MAX_HISTORY_LENGTH = 20

    SYSTEM_PROMPT = "你是一位专业的金融分析师助手，请用简洁准确的语言回答问题。"

    # RAG 配置
    KNOWLEDGE_BASE_DIR = "knowledge_base"          # 知识库目录（支持多文档）
    COLLECTION_NAME = "financial_reports"          # ChromaDB collection 名
    CHUNK_SIZE = 400                               # 金融研报段落较长，适当增大
    CHUNK_OVERLAP = 80
    TOP_K = 5                                      # 候选召回数
    RERANK_TOP_K = 3                               # Reranker 精排后保留数

    RAG_SYSTEM_PROMPT = """你是一位专业的金融研报分析师助手。
请严格基于以下研报材料回答问题，引用具体数据和来源机构。
如果材料中没有相关信息，请明确说明"研报中未提及该内容"，不要编造数据。
回答时注意：
1. 优先引用具体数字（营收、利润、市占率等）
2. 注明数据来源（发布机构、报告日期）
3. 如有风险提示，需一并告知"""

    # FastAPI
    API_HOST = "0.0.0.0"
    API_PORT = 8000

    # RAGAS 评估（需要 OpenAI key，仅评估时使用）
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
