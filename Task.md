请仔细阅读以下文件，理解当前实现，然后按要求进行改造：

## 需要阅读的文件
- src/app/online_qa/state.py
- src/app/online_qa/nodes/answer_node.py
- src/app/online_qa/cli.py
- src/app/online_qa/checkpointer.py
- src/core/tools/llm/generation_tool.py

## 背景说明
当前多轮对话的 history 通过 .session_history/*.json 文件手动维护，
与 SqliteSaver（.checkpoints/c9.db）并行存在，逻辑割裂。
目标是删除文件方案，让 SqliteSaver 原生承载 history。

## 改造要求

### 1. state.py
- 为 history 字段添加追加语义的 reducer，确保 SqliteSaver 
  恢复 state 后 history 不会被新调用覆盖，而是自动累积

### 2. answer_node.py
当前问题：
- generate() 调用没有传入 history，历史上下文完全未被利用
- 每次返回的 history 是本轮2条，但 state 中缺少 reducer 
  导致无法自动累积

改造要求：
- 从 state 中读取已有 history（SqliteSaver 会自动恢复）
- 将 history 传入 generate() 调用
- 返回值中 history 只包含本轮新增的2条（reducer 负责合并，不要手动拼接全量）

### 3. generation_tool.py
检查 generate() 方法签名是否支持 history 参数：
- 如果不支持，添加 history 参数，并将其拼入 prompt
- history 格式为 [{"role": "user", "content": "..."}, 
                   {"role": "assistant", "content": "..."}]
- 建议拼入方式：在 system prompt 或 user message 前附加近几轮对话（最多取最近5轮，即10条）

### 4. cli.py
- 删除所有 .session_history 相关的文件读写逻辑
  （load_session_history / save_session_history 及其调用）
- ainvoke 调用只传 {"query": args.query}，不传 history
- 确保 config={"configurable": {"thread_id": session_id}} 正确传入

### 5. 验证改造正确性
改造完成后，检查以下陷阱：
- ainvoke 的初始 state 中不能包含 history 字段（否则会覆盖 
  SqliteSaver 恢复的值）
- state.py 的 reducer 函数需要处理 existing 为 None 的情况
- generation_tool 中 history 为空列表时不应拼入任何内容

## 不要改动的内容
- src/legacy/** 下的任何文件
- checkpointer.py 的 SqliteSaver 逻辑
- 其他节点（query_analysis, route, retrieve, fuse）