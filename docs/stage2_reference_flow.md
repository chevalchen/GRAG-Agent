# Stage 2 Reference: Full Agentic Flow (supplementary)

> **注意**：本文档是用户提供的参考设计，仅供对照。Stage 2 实施以 `stage2_backend_design.md` 为准。

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 45}, "themeVariables": {"fontSize": "13px"}}}%%
graph TD
    START((START)):::start_node
    END_NODE((END)):::start_node
    ChatEnd[闲聊兜底节点<br/>Chat_Fallback]:::end_node

    subgraph "1. 意图与重写层 (Routing & Rewriting)"
        Router[极速意图路由节点<br/>Intent_Router]:::router
        Rewrite[全能医学重写节点<br/>Medical_Rewrite<br/>指代消解+画像注入]:::router
    end

    subgraph "2. 并发异构检索层 (Retrieval Layer)"
        MilvusDrug[Milvus_Drug<br/>说明书并发检索]:::milvus
        MilvusLit[Milvus_Lit<br/>中医文献并发检索]:::milvus
        MilvusHealth[Milvus_Health<br/>健康科普并发检索]:::milvus
        GraphEnrich[Neo4j 正向穿透节点<br/>提取禁忌红线]:::neo4j
        GraphSearch[Neo4j 反向规则节点<br/>多跳交集圈选]:::neo4j
        MilvusFetch[Milvus 标量拉取节点<br/>提取原文切片]:::milvus
    end

    subgraph "3. 融合与排序层 (Ranking Layer)"
        Rerank[RRF两级加权融合<br/>+ Cross-Encoder精排]:::ranker
    end

    subgraph "4. 生成与安全控制层 (Generation & Safety)"
        Generate[草稿生成节点<br/>Draft_Generator]:::gen
        Safety[安全审查节点<br/>Safety_Guardrail]:::safety
    end

    START --> Router
    Router -- "非医疗/闲聊" --> ChatEnd --> END_NODE
    Router -- "医疗咨询" --> Rewrite

    Rewrite -- "意图: 理论/科普" --> MilvusLit
    Rewrite -.-> MilvusHealth

    Rewrite -- "意图: 模糊寻药" --> MilvusDrug
    Rewrite -.-> MilvusLit
    Rewrite -.-> MilvusHealth
    MilvusDrug -- "传递 kg_anchor_id" --> GraphEnrich

    Rewrite -- "意图: 明确禁忌/多跳条件" --> GraphSearch
    GraphSearch -- "传递 doc_ids" --> MilvusFetch

    MilvusLit --> Rerank
    MilvusHealth --> Rerank
    GraphEnrich --> Rerank
    MilvusFetch --> Rerank

    Rerank --> Generate
    Generate --> Safety
    Safety -- "通过 (Pass)" --> END_NODE
    Safety -- "熔断 (Blocked)" --> Rewrite

    classDef start_node fill:#333,stroke:#fff,stroke-width:2px,color:#fff
    classDef end_node fill:#ccc,stroke:#999,stroke-width:1px
    classDef router fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px
    classDef milvus fill:#e8f5e9,stroke:#4caf50,stroke-width:2px
    classDef neo4j fill:#ffebee,stroke:#f44336,stroke-width:2px
    classDef ranker fill:#fff8e1,stroke:#ffc107,stroke-width:2px
    classDef gen fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
    classDef safety fill:#ffccbc,stroke:#ff5722,stroke-width:2px,stroke-dasharray:5 5
```

## 与原设计的关键 Delta

| 维度 | 原设计（authority） | 参考设计（reference） |
|---|---|---|
| 意图分类数 | 2 类（symptom_to_drug / conditions_exact） | 4 类（+ theory / chat） |
| 查询重写 | 无独立重写节点 | `Medical_Rewrite`（指代消解 + 画像注入） |
| 检索分支 | 症状→Milvus+GraphEnrich；禁忌→Neo4j+MilvusFetch | 理论分支仅 Lit+Health；症状分支 3 路 Milvus+Graph | 
| 充足性回环 | Router 回环（最多 1 次） | 无充足性回环；Safety 熔断后 → Rewrite |
| 安全审查 | 无独立 Safety 节点 | `Safety_Guardrail`（生成后） |
| 用户画像 | 无 | `user_profile` (SQLite 长期记忆) |
| 闲聊处理 | 无（所有 query 走医疗流程） | 独立 `Chat_Fallback` 分支 |

## Stage 2 实施中可低成本采纳的增量（不改主流程）

以下 3 点与原计划兼容，实施时直接加入：

1. **`rewritten_query` 字段** — 在 Router 内部追加一步 LLM prompt 做指代消解，结果写入 state，后续节点优先用 `rewritten_query`；原 `query` 保留供回溯。
2. **`extracted_entities` 字段** — Router 解析 JSON 时顺带提取药名/病名/症状锚点，不增加额外 LLM 调用。
3. **`safety_blocked` + `safety_reason`** — `answer_node` 生成后加一个轻量规则检查（十八反/十九畏关键词列表匹配），命中则设标志位并在回答前追加免责说明；不做回环。
