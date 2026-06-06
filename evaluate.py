"""
RAG 评估脚本

评估指标：
  检索质量：Hit Rate（命中率）、MRR（平均倒数排名）
  生成质量：faithfulness、answer_relevancy、context_precision、context_recall（RAGAS）

运行方式：
    python evaluate.py

注意：RAGAS 指标需要 ANTHROPIC_API_KEY；完整评估还需 OPENAI_API_KEY
"""

import os
import re
import sys
sys.path.insert(0, os.path.dirname(__file__))

import anthropic
from core.config import Config
from core.rag_engine import RAGEngine

# ── 测试问答集（覆盖多份研报）────────────────────────────────
TEST_DATA = [
    {
        "query": "宁德时代2024年Q3的毛利率是多少？",
        "reference": "宁德时代2024年Q3毛利率为26.3%，创近八季度新高。",
    },
    {
        "query": "宁德时代全球动力电池市占率是多少？",
        "reference": "宁德时代2024年Q3全球动力电池市占率约为37.2%，排名全球第一。",
    },
    {
        "query": "宁德时代的神行超充电池有什么特点？",
        "reference": "神行超充电池是首款量产4C超充磷酸铁锂电池，充电15分钟续航600km，年产能规划50GWh。",
    },
    {
        "query": "宁德时代储能业务的增长情况？",
        "reference": "宁德时代2024年Q3储能电池出货量62GWh，同比增长42%。",
    },
    {
        "query": "贵州茅台的直销渠道占比是多少？",
        "reference": "贵州茅台2024年前三季度直销占比约44.9%，同比提升约6个百分点。",
    },
    {
        "query": "贵州茅台的毛利率和净利率是多少？",
        "reference": "贵州茅台毛利率约92.1%，净利率约49.8%，均为消费品行业最高水平。",
    },
    {
        "query": "飞天茅台的出厂价是多少？",
        "reference": "飞天茅台（53%vol，500ml）出厂价为1169元/瓶。",
    },
    {
        "query": "比亚迪2024年前三季度新能源车销量是多少？",
        "reference": "比亚迪2024年前三季度累计销量274.8万辆，同比增长32.1%。",
    },
    {
        "query": "比亚迪的第五代DM技术有什么亮点？",
        "reference": "比亚迪第五代DM技术亏电油耗低至2.9L/100km，综合续航超过2000km。",
    },
    {
        "query": "2025年新能源汽车渗透率预测是多少？",
        "reference": "预计2025年中国新能源汽车渗透率将突破52%，销量约1,420万辆。",
    },
    {
        "query": "2025年中国大储装机量预测是多少？",
        "reference": "预计2025年中国大型储能新增装机约120GWh，同比增长60%。",
    },
]


# ── 检索指标：Hit Rate & MRR ──────────────────────────────────

def _extract_keywords(reference: str) -> list[str]:
    """从参考答案中提取关键词（数字+专有名词）。"""
    keywords = []
    # 提取数字串（含百分比、金额等）
    keywords += re.findall(r'\d+\.?\d*%?', reference)
    # 提取2字以上汉字词组
    keywords += [w for w in re.split(r'[，。、%元亿倍次]', reference) if len(w) >= 2]
    return list(set(keywords[:6]))  # 取前6个关键词


def calculate_hit_rate(retrieved_docs: list[str], reference: str) -> float:
    """Hit Rate：检索到的文档中是否至少有一条包含参考答案的关键信息。"""
    keywords = _extract_keywords(reference)
    for doc in retrieved_docs:
        if any(kw in doc for kw in keywords[:3]):
            return 1.0
    return 0.0


def calculate_mrr(retrieved_docs: list[str], reference: str) -> float:
    """MRR（Mean Reciprocal Rank）：第一条相关文档排名的倒数。"""
    keywords = _extract_keywords(reference)
    for i, doc in enumerate(retrieved_docs):
        if any(kw in doc for kw in keywords[:3]):
            return 1.0 / (i + 1)
    return 0.0


def generate_response(query: str, docs: list[str]) -> str:
    context = "\n\n".join(docs)
    client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=Config.MODEL_NAME,
        max_tokens=Config.MAX_TOKENS,
        messages=[{"role": "user", "content": f"根据以下研报资料回答问题，只使用材料中的信息。\n\n资料：\n{context}\n\n问题：{query}"}],
    )
    return resp.content[0].text


# ── 主评估流程 ────────────────────────────────────────────────

def run_evaluation():
    print("=" * 60)
    print("金融研报 RAG 系统评估")
    print("=" * 60)

    engine = RAGEngine()
    engine.build_vector_db()

    # ── Step 1：检索指标（Hit Rate & MRR）──────────────────────
    print("\n【检索质量评估】Hit Rate & MRR")
    print("-" * 60)

    hit_rates, mrrs = [], []
    retrieval_results = []

    for item in TEST_DATA:
        query = item["query"]
        # 混合检索 → Reranker（评估用 top-5 排序结果）
        candidates = engine.hybrid_search(query, top_k=20)
        ranked_docs = engine.rerank(query, candidates, top_k=5)

        hit = calculate_hit_rate(ranked_docs, item["reference"])
        mrr = calculate_mrr(ranked_docs, item["reference"])
        hit_rates.append(hit)
        mrrs.append(mrr)
        retrieval_results.append((query, ranked_docs, hit, mrr))

        status = "✅" if hit == 1.0 else "❌"
        print(f"  {status} Hit={hit:.1f} MRR={mrr:.2f}  {query[:35]}")

    avg_hit = sum(hit_rates) / len(hit_rates)
    avg_mrr = sum(mrrs) / len(mrrs)
    print(f"\n  平均 Hit Rate：{avg_hit:.3f}　平均 MRR：{avg_mrr:.3f}")

    # ── Step 2：生成质量评估（RAGAS）──────────────────────────
    print("\n【生成质量评估】RAGAS 四项指标")
    print("-" * 60)

    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_anthropic import ChatAnthropic
        from langchain_huggingface import HuggingFaceEmbeddings

        ragas_llm = LangchainLLMWrapper(ChatAnthropic(model=Config.MODEL_NAME))
        ragas_emb = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
        )

        samples = []
        for item, (_, docs, _, _) in zip(TEST_DATA, retrieval_results):
            response = generate_response(item["query"], docs[:3])
            samples.append(SingleTurnSample(
                user_input=item["query"],
                retrieved_contexts=docs,
                response=response,
                reference=item["reference"],
            ))
            print(f"  [生成完成] {item['query'][:30]}...")

        result = evaluate(
            dataset=EvaluationDataset(samples=samples),
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ragas_llm,
            embeddings=ragas_emb,
        )

        print("\n" + "=" * 60)
        print("综合评估结果")
        print("=" * 60)
        print(f"  检索 Hit Rate：{avg_hit:.3f}　MRR：{avg_mrr:.3f}")
        print(f"  生成 {result}")

        df = result.to_pandas()
        df["hit_rate"] = hit_rates
        df["mrr"] = mrrs
        df.to_csv("ragas_results.csv", index=False, encoding="utf-8-sig")
        print("\n完整结果已保存至：ragas_results.csv")

    except ImportError:
        print("ragas 未安装，跳过生成质量评估。")
        print("\n" + "=" * 60)
        print("评估结果汇总（检索指标）")
        print("=" * 60)
        print(f"  Hit Rate：{avg_hit:.3f}　MRR：{avg_mrr:.3f}")
        _print_retrieval_detail(retrieval_results)


def _print_retrieval_detail(results: list):
    print("\n逐条检索详情：")
    for query, docs, hit, mrr in results:
        print(f"\n  Q: {query}")
        print(f"     Hit={hit:.0f}  MRR={mrr:.2f}  首条片段: {docs[0][:80] if docs else '无'}...")


if __name__ == "__main__":
    run_evaluation()
