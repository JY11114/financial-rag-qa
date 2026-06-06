import os
import re
import jieba
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from core.config import Config


class RAGEngine:
    def __init__(self):
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name=Config.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "，", " ", ""],
        )
        self.chunks: list[str] = []
        self.chunk_metadata: list[dict] = []
        self.bm25 = None

        print("正在加载 Reranker 模型...")
        self.reranker = CrossEncoder("BAAI/bge-reranker-base")
        print("RAG 引擎初始化完成")

    # ── 元数据提取 ────────────────────────────────────────────
    def _extract_metadata(self, content: str, filename: str) -> dict:
        """从研报文件头部提取结构化元数据。"""
        meta = {
            "source": filename,
            "company": "综合",
            "code": "",
            "sector": "新能源",
            "institution": "未知机构",
            "report_date": "2024",
            "rating": "未知",
        }
        header = content[:600]

        # 公司名 & 股票代码
        m = re.search(r'【(.+?)（(\d{6})）', header)
        if m:
            meta["company"] = m.group(1)
            meta["code"] = m.group(2)

        # 行业判断
        if "行业" in filename:
            meta["company"] = "行业报告"
        if "茅台" in header or "白酒" in header:
            meta["sector"] = "白酒"
        if "比亚迪" in header:
            meta["sector"] = "新能源汽车"

        # 发布机构
        for inst in ["华泰证券", "中信证券", "国泰君安", "申万宏源", "招商证券", "广发证券", "海通证券"]:
            if inst in header:
                meta["institution"] = inst
                break

        # 发布日期
        m = re.search(r'发布日期[：:]\s*(\d{4}年\d{1,2}月\d{1,2}日)', header)
        if m:
            meta["report_date"] = m.group(1)

        # 评级
        m = re.search(r'评级[：:]\s*(\S+)', header)
        if m:
            meta["rating"] = m.group(1)

        return meta

    # ── 知识库加载（支持目录下多文档）────────────────────────
    def load_knowledge(self) -> tuple[list[str], list[dict]]:
        kb_dir = Config.KNOWLEDGE_BASE_DIR
        all_chunks: list[str] = []
        all_metadata: list[dict] = []
        doc_count = 0

        for filename in sorted(os.listdir(kb_dir)):
            if not filename.endswith(".txt"):
                continue
            filepath = os.path.join(kb_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            base_meta = self._extract_metadata(content, filename)
            chunks = self.splitter.split_text(content)

            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({**base_meta, "chunk_index": i})

            doc_count += 1
            print(f"  [{doc_count}] {filename} → {len(chunks)} chunks（{base_meta['company']} / {base_meta['institution']}）")

        self.chunks = all_chunks
        self.chunk_metadata = all_metadata
        print(f"知识库加载完成：{doc_count} 份文档，共 {len(all_chunks)} 个 chunks")
        return all_chunks, all_metadata

    # ── 向量数据库构建 ────────────────────────────────────────
    def build_vector_db(self):
        chunks, metadata_list = self.load_knowledge()
        ids = [f"chunk_{i}" for i in range(len(chunks))]

        self.collection.upsert(
            documents=chunks,
            ids=ids,
            metadatas=metadata_list,
        )

        tokenized = [list(jieba.cut(c)) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

        print(f"向量数据库 + BM25 索引构建完成，共 {len(chunks)} 条")

    # ── 向量检索 ──────────────────────────────────────────────
    def search(self, query: str, top_k: int = None, filter: dict = None) -> list[str]:
        if top_k is None:
            top_k = Config.TOP_K
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filter,
        )
        docs = results["documents"][0]
        print(f"向量检索：找到 {len(docs)} 条")
        return docs

    # ── BM25 检索（支持元数据过滤）────────────────────────────
    def bm25_search(self, query: str, top_k: int = None, filter: dict = None) -> list[str]:
        if top_k is None:
            top_k = Config.TOP_K
        if not self.bm25:
            return []

        tokenized_query = list(jieba.cut(query))

        if filter:
            valid_idx = [
                i for i, m in enumerate(self.chunk_metadata)
                if all(m.get(k) == v for k, v in filter.items())
            ]
            filtered_chunks = [self.chunks[i] for i in valid_idx]
            if not filtered_chunks:
                return []
            tokenized_filtered = [list(jieba.cut(c)) for c in filtered_chunks]
            bm25_local = BM25Okapi(tokenized_filtered)
            scores = bm25_local.get_scores(tokenized_query)
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            return [filtered_chunks[i] for i in top_idx]

        scores = self.bm25.get_scores(tokenized_query)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        docs = [self.chunks[i] for i in top_idx]
        print(f"BM25 检索：找到 {len(docs)} 条")
        return docs

    # ── RRF 混合检索 ──────────────────────────────────────────
    def hybrid_search(self, query: str, top_k: int = None, filter: dict = None) -> list[str]:
        if top_k is None:
            top_k = Config.TOP_K * 4

        vector_results = self.search(query, top_k=top_k, filter=filter)
        bm25_results = self.bm25_search(query, top_k=top_k, filter=filter)

        rrf_scores: dict[str, float] = {}
        k = 60
        for rank, doc in enumerate(vector_results):
            rrf_scores[doc] = rrf_scores.get(doc, 0) + 1 / (k + rank + 1)
        for rank, doc in enumerate(bm25_results):
            rrf_scores[doc] = rrf_scores.get(doc, 0) + 1 / (k + rank + 1)

        sorted_docs = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)
        result = sorted_docs[:top_k]
        print(f"混合检索（RRF）：{len(result)} 条候选")
        return result

    # ── Query 改写 ────────────────────────────────────────────
    def rewrite_query(self, query: str, llm) -> str:
        prompt = f"""将以下用户问题改写为更适合在金融研报知识库中检索的形式。
要求：
1. 补全省略的公司名称（如"他们"→具体公司名）
2. 去除口语化表达
3. 保持原意不变
4. 只返回改写后的问题，不要解释

用户问题：{query}"""
        messages = [{"role": "user", "content": prompt}]
        rewritten = llm.send_message(messages=messages)
        print(f"Query 改写：{query} → {rewritten}")
        return rewritten

    # ── Reranker 精排 ─────────────────────────────────────────
    def rerank(self, query: str, docs: list[str], top_k: int = None) -> list[str]:
        if top_k is None:
            top_k = Config.RERANK_TOP_K
        if not docs:
            return []
        pairs = [(query, doc) for doc in docs]
        scores = self.reranker.predict(pairs)
        top_idx = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [docs[i] for i in top_idx]

    # ── 检索结果来源元数据 ────────────────────────────────────
    def get_source_info(self, docs: list[str]) -> list[str]:
        """根据文档内容反查元数据，返回来源描述。"""
        sources = []
        for doc in docs:
            for i, chunk in enumerate(self.chunks):
                if chunk == doc and i < len(self.chunk_metadata):
                    m = self.chunk_metadata[i]
                    src = f"{m.get('company','未知')}（{m.get('institution','未知机构')} · {m.get('report_date','')} · 评级：{m.get('rating','')}）"
                    if src not in sources:
                        sources.append(src)
                    break
        return sources
