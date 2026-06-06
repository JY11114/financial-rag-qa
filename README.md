# 基于 RAG 的金融研报问答助手

面向金融投研场景的 RAG 问答系统，加载券商研报构建本地知识库，围绕文档切分、混合检索、重排精排、效果评估与幻觉控制完成全链路工程优化，提升专业问答的准确率与可信度。

**技术栈**：Python · LangChain · ChromaDB · BM25 · RAGAS · FastAPI · Claude API

## 系统架构

```
【离线构建】
knowledge_base/*.txt（4份研报）
    │
    ├── 元数据提取（公司 / 机构 / 日期 / 评级）
    ├── 数据清洗（去页码 / 去页眉页脚 / 去噪 / 归一化空白）
    ├── 递归字符切块
    ├── ChromaDB 向量入库（含元数据）
    └── jieba 分词 → BM25 索引

【在线查询】
用户输入
    │
    ├── rewrite_query()     →  LLM 查询改写
    │
    ├── hybrid_search()
    │     ├── 向量检索（ChromaDB）
    │     ├── BM25 稀疏检索（支持元数据过滤）
    │     └── RRF 融合
    │
    ├── rerank()            →  CrossEncoder 精排
    ├── get_source_info()   →  来源标注（机构 · 日期 · 评级）
    └── send_message()      →  LLM 生成回答

返回：answer + sources  →  CLI / FastAPI（session 隔离）

【效果评估】
evaluate.py（11条测试集）
    ├── 检索：Hit Rate / MRR
    └── 生成：RAGAS → faithfulness / answer_relevancy / context_precision / context_recall
```

## 核心功能

### 数据处理与切片
针对研报文本采用递归字符切片并设置重叠窗口，对切分结果绑定来源、机构、日期、评级等结构化元数据，支撑后续元数据过滤与引用溯源。

### 混合检索与融合
并行构建向量稠密检索（ChromaDB）与 BM25 稀疏检索两路通道，召回候选后通过 RRF 融合重排，兼顾语义相关性与关键词精确匹配，缓解专业术语场景下的漏召问题。

### 查询理解与精排
检索前引入 LLM 查询改写，补全口语化、模糊化提问以提升召回质量；对 RRF 融合后的候选文档通过 CrossEncoder 二次精排，筛选高置信片段进入上下文。

### Prompt 编排与降幻觉
设计结构化 Prompt 模板，约束模型仅依据给定研报材料作答、标注引用来源，无依据时显式拒答，有效抑制编造。

### 效果评估与迭代
基于 RAGAS 框架结合 Hit Rate、MRR 搭建检索与生成的联合评估流程，构建 11 条金融问答测试集做量化对比，驱动迭代优化。

### 服务化封装
基于 FastAPI 将 RAG 问答能力封装为后端服务接口，通过 session 机制实现多用户会话隔离与多轮对话历史管理，支持元数据过滤参数、历史清除与统一异常处理。

## 项目结构

```
financial-rag-qa/
├── core/
│   ├── config.py          # 配置（模型、chunk 参数、RAG Prompt）
│   ├── llm_client.py      # Claude API 封装
│   └── rag_engine.py      # RAG 引擎（多文档·元数据·混合检索·Reranker·溯源）
├── service/
│   └── rag_service.py     # 业务层（改写→检索→精排→生成，返回答案+来源）
├── utils/
│   └── formatter.py       # CLI 输出格式化
├── knowledge_base/        # 4份券商研报
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

# 运行评估
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

## 评估指标

| 指标 | 类型 | 含义 |
|------|------|------|
| Hit Rate | 检索 | 召回文档中是否命中参考答案关键信息 |
| MRR | 检索 | 第一条相关文档排名的倒数均值 |
| faithfulness | 生成 | 答案是否忠实于检索到的上下文 |
| answer_relevancy | 生成 | 答案是否切题 |
| context_precision | 生成 | 检索到的片段是否真正有用 |
| context_recall | 生成 | 有用信息是否被充分召回 |
# financial-rag-qa
