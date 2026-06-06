# 基于 RAG 的金融研报问答助手

面向金融投研场景的 RAG 问答系统，加载券商研报与金融机构观点构建本地知识库，围绕文档切分、混合检索、重排精排、效果评估与幻觉控制完成全链路工程优化，提升专业问答的准确率与可信度。

**技术栈**：Python · LangChain · ChromaDB · BM25 · RAGAS · FastAPI · Claude API

## 系统架构

```
用户提问
    │
    ▼  Query 改写（LLM 补全口语/模糊表达）
    │
    ├─── 向量检索（ChromaDB · 稠密）
    ├─── BM25 稀疏检索（jieba 分词）
    │         └── RRF 融合重排
    │
    ▼  CrossEncoder Reranker 精排（bge-reranker-base）
    │
    ▼  Prompt 编排（结构化模板 · 降幻觉约束）
    │
    ▼  Claude 生成回答（附引用来源）
```

## 核心功能

### 数据处理与切片
针对研报文本的章节与段落结构采用递归字符切片并设置重叠窗口，在保持语义连贯性的同时控制 chunk 粒度；对切分结果绑定来源、机构、日期、评级等结构化元数据，支撑后续元数据过滤与引用溯源。

### 混合检索与融合
并行构建向量稠密检索（ChromaDB）与 BM25 稀疏检索两路通道，召回候选后通过 RRF（Reciprocal Rank Fusion）融合重排，兼顾语义相关性与关键词精确匹配，缓解专业术语场景下的漏召问题。

### 查询理解与精排
检索前引入 LLM 查询改写，补全口语化、模糊化提问以提升召回质量；对 RRF 融合后的候选文档通过 CrossEncoder 按相关性二次精排，筛选高置信片段进入上下文，降低噪声干扰。

### Prompt 编排与降幻觉
设计结构化 Prompt 模板，约束模型仅依据给定研报材料作答、标注引用来源（机构名·日期·评级）、无依据时显式拒答，有效抑制编造，显著降低幻觉率。

### 效果评估与迭代
基于 RAGAS 框架结合 Hit Rate、MRR（平均倒数排名）、faithfulness、answer_relevancy 等指标搭建检索与生成的联合评估流程，构建 11 条金融问答测试集对召回数量、融合策略等参数做量化对比，驱动迭代优化。

### 服务化封装
基于 FastAPI 将 RAG 问答能力封装为后端服务接口，通过 session 机制实现多用户会话隔离与多轮对话历史管理，支持元数据过滤参数、历史清除接口与统一异常处理，保障接口稳定可用。

## 项目结构

```
financial-rag-qa/
├── core/
│   ├── config.py          # 配置（模型、chunk 参数、RAG Prompt）
│   ├── llm_client.py      # Claude API 封装
│   └── rag_engine.py      # RAG 引擎（多文档·混合检索·元数据·Reranker·溯源）
├── service/
│   └── rag_service.py     # 业务层（改写→检索→精排→生成，返回答案+来源）
├── utils/
│   └── formatter.py       # CLI 输出格式化
├── knowledge_base/        # 研报文档（4 份，自动多文档入库）
│   ├── catl_research_2024.txt      # 宁德时代（华泰证券·2024-10-15）
│   ├── maotai_research_2024.txt    # 贵州茅台（中信证券·2024-11-08）
│   ├── byd_research_2024.txt       # 比亚迪（国泰君安·2024-11-12）
│   └── new_energy_sector_2024.txt  # 新能源行业策略（申万宏源·2024-11-25）
├── main.py                # CLI 交互入口（含来源标注输出）
├── fapi.py                # FastAPI 服务（会话隔离·元数据过滤·知识库统计）
├── evaluate.py            # 评估脚本（Hit Rate·MRR·RAGAS 四项指标）
└── requirements.txt
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 API Key
export ANTHROPIC_API_KEY=your_key_here

# CLI 模式运行
python main.py

# 启动 FastAPI 服务（接口文档：http://localhost:8000/docs）
uvicorn fapi:app --reload

# 运行评估（RAGAS 完整评估需要 OPENAI_API_KEY）
python evaluate.py
```

## FastAPI 接口

```bash
# 多轮问答
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "宁德时代储能业务增速如何？", "session_id": "user_01"}'

# 限定公司范围检索
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "目标价是多少？", "session_id": "user_01", "company": "宁德时代"}'

# 查看会话历史 / 清空会话
curl http://localhost:8000/sessions/user_01/history
curl -X DELETE http://localhost:8000/sessions/user_01

# 知识库统计
curl http://localhost:8000/knowledge-base/stats
```

## 评估指标说明

| 指标 | 类型 | 含义 |
|------|------|------|
| Hit Rate | 检索 | 召回文档中是否命中参考答案关键信息 |
| MRR | 检索 | 第一条相关文档排名的倒数均值 |
| faithfulness | 生成 | 答案是否忠实于检索到的上下文 |
| answer_relevancy | 生成 | 答案是否切题 |
| context_precision | 生成 | 检索到的片段是否真正有用 |
| context_recall | 生成 | 有用信息是否被充分召回 |

## 技术选型说明

**为什么用混合检索？**
金融研报中大量专有名词（麒麟电池、DM-i、4C超充）纯向量模型难以精准召回；BM25 精确关键词匹配弥补语义模型的不足，RRF 融合兼顾两路优势。

**为什么需要 Reranker？**
向量检索与 BM25 均为粗排（bi-encoder），CrossEncoder 在 query-doc 对上计算交叉注意力，精度显著更高。召回 20 条 → Reranker 精排至 3 条，大幅压缩噪声。

**降幻觉设计**
System Prompt 硬约束：仅依据研报材料作答；无依据时显式声明"研报中未提及"；回答附带具体来源（机构·日期），可追溯核查。
# langgraph-agent
