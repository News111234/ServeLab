# ServeLab 精读路线图：不知道从哪看，就按这份走

> 这不是参考手册（那是 `architecture.md`），是**按顺序执行的精读教程**。
> 主线只有一条：**一个请求的一生**——从 `add_request` 进来，到 token 被吐出去，
> 中间每个数据结构都被它碰一遍。跟着这条线读，就不会迷路。
> 总预算：2 周（每天 1~1.5 小时）。每阶段末尾有"过关标准"，达不到就停在那里补。

---

## 阶段 0 · 跑起来（30 分钟）

目的：先让机器替你"讲"一遍系统在干嘛，建立直觉。

```bash
cd ServeLab
pip install -e ".[all]"          # 或最小安装 numpy tabulate torch safetensors
pytest tests -q                  # 62 passed, 2 skipped —— 记住这个基线
python examples/run_engine.py    # 随机小模型端到端，注意 verbose 每步打印
python examples/run_simulator.py --compare-routers --num-requests 200
python examples/moe_sim.py
```

**观察任务**（现在看不懂没关系，混个眼熟）：
1. `run_engine.py` 的 verbose 输出：`running/waiting/cache` 三个数字怎么变化的
2. 测试名列表扫一遍：`test_block_manager / test_radix / test_scheduler / ...`
   ——名字就是功能清单

---

## 阶段 1 · 一条主线：一个请求的一生（2 天，不精读，先串场）

把这 7 步在编辑器里**跳转着走一遍**（每步只看入口函数，不抠细节）：

| 步 | 发生了什么 | 入口（跳转到这行） |
|---|---|---|
| ① 请求进入 | prompt 变成 Sequence，进 waiting 队列 | `engine/engine.py::add_request` |
| ② 前缀匹配 | 如果缓存里有相同前缀的完整块，直接复用（锁住） | `kv_cache/manager.py::maybe_match_prefix` |
| ③ 准入与分块 | 按预算切 prefill chunk，分配块，进 running | `engine/scheduler.py::schedule`（三阶段注释） |
| ④ 打包批 | 所有序列的 token 摊平 + 算好每个 token 的物理槽位 | `engine/engine.py::step`（tokens/positions/slots/metas） |
| ⑤ 前向 | 逐层：QKV → RoPE → KV 写入池 → 分页注意力 → MLP | `models/decoder.py::forward_packed` |
| ⑥ 采样 | 取目标行的 logits → argmax/采样出新 token | `engine/engine.py::step` 后半 + `engine/sampler.py` |
| ⑦ 落账 | token 追加、停止判断、块表交还前缀树 | `engine/scheduler.py::update_after_exec` → `manager.py::release` |

**本阶段唯一要吃透的概念：物理槽位（slot）**

```
逻辑 token 序列:  t0 t1 t2 t3 | t4 t5 t6 t7 | ...
物理 KV 池(行):   块0占行0~3   块1占行4~7   ...
slot = block_id × block_size + 块内偏移
```
拿笔画一个 4 块 × 块大小 4 的格子图，把 10 个 token 摆进去，标出每块的行号范围。
看不懂后面全是天书，看懂了后面全是常识。

---

## 阶段 2 · 核心三件套精读（4~5 天，最重要的 1000 行）

按顺序精读，每个文件配一个动手实验。**遇到看不懂的先问自己：它在防什么错？**

### 2.1 `kv_cache/manager.py`（约 300 行，半天）

精读顺序：`maybe_match_prefix` → `allocate_slots` → `append_slot` → `release` → `swap_out/swap_in`（跳过 offload 细节）。

**动手实验**：开个 `python -i`，亲手走一遍：

```python
from servelab.kv_cache.manager import BlockManager
bm = BlockManager(num_blocks=16, block_size=4)
print(bm.maybe_match_prefix("s1", [1,2,3,4,5,6,7,8]))   # 0，没有缓存
print(bm.allocate_slots("s1", 8))                        # 分到哪些块？
print(bm.get_block_table("s1"), bm.num_free_blocks())
bm.release("s1", [1,2,3,4,5,6,7,8])                      # 释放后进了哪？
print(bm.maybe_match_prefix("s2", [1,2,3,4,5,6,7,8]))    # 再来一个相同 prompt → 命中几个？
print(bm.get_block_table("s2"))                          # 和 s1 的块表对比！
```

**过关标准**：能回答 `tests/test_block_manager.py` 的前 3 个测试为什么这么断言；
能说清 `release` 时为什么只传 `token_ids[:num_computed_tokens]`（`PROGRESS.md` 缺陷 #3）。

### 2.2 `kv_cache/radix.py` + `policies.py`（约 250 行，1 天）

精读顺序：`match_prefix` → `insert` → `unlock` → `evict_with_info` → `attach`。
配合 `policies.py`（很短）和 `utils/common.py::hash_block_tokens`。

**动手实验**：

```python
from servelab.kv_cache.radix import RadixCache
rc = RadixCache(block_size=4)
rc.insert([1,2,3,4,5,6,7,8], [10,11])          # 两个完整块进树
print(rc.match_prefix([1,2,3,4,9,9,9,9]))      # 第二块不同 → 命中几个？锁了几个？
print(rc.evict(5))                              # 锁着的能被驱逐吗？
rc.unlock(rc.match_prefix([1,2,3,4])[1])
print(rc.evict(5))                              # 解锁后呢？
```

**过关标准**：能解释"为什么驱逐只挑叶子""为什么锁保护的是运行中序列"；
能画出 `[1,2,3,4,5,6,7,8]` 和 `[1,2,3,4,9,9,9,9]` 两条序列在树里的形状。

### 2.3 `engine/scheduler.py`（约 330 行，2 天，全项目最核心）

精读顺序：`schedule()`（就一个函数，三阶段注释已写好）→ `_preempt` /
`_handle_alloc_failure` → `update_after_exec`。

**动手实验**：跑专用测试看调度行为：

```bash
pytest tests/test_scheduler.py::test_chunked_prefill_respects_token_budget -v
pytest tests/test_scheduler.py::test_preemption_under_memory_pressure -v
```
然后把 `tests/test_scheduler.py` 里的 `Driver` 抄进 `python -i`，自己提交两个请求、
手动调 `schedule()` 一步步看 `SchedulerOutput` 里 prefills/decodes 的内容。

**过关标准**：能白板画出 schedule() 三阶段流程图；
能回答：一个 seq 的 `num_computed_tokens` 和 `get_len()` 是什么关系？
（decode 阶段不变量：`computed == len-1`，`PROGRESS.md` 缺陷 #1 就是它崩了）

---

## 阶段 3 · 引擎与模型（3 天）

### 3.1 `engine/engine.py::step`（1 天）

逐行读"打包批"的构建：`tokens / positions / slots / metas / logit_rows` 五个列表
是怎么为每个序列填的（prefill chunk 和 decode 各一遍）。这是 vLLM v0 的打包批思想。

**过关标准**：能回答"为什么 prefill 完成的序列才有 logit 行？"
（chunk 没读完的序列这步不采样——对照 `tests/test_engine_smoke.py` 四配置一致性）

### 3.2 `models/decoder.py::forward_packed`（1 天）

一层循环五件事：layernorm → qkv 投影 → RoPE → **KV 写池 + 读回做注意力** → MLP。
对照 `pool.py::write/read` 看"写进去的行号"和"gather 出来的行号"是同一套 slot。

**动手实验**：`pytest tests/test_engine_smoke.py::test_kv_growth_invariant -v`，
然后试着把 `engine.step` 里 decode 分支的 `seq.num_computed_tokens = seq.get_len()`
注释掉重跑——亲眼看不变量崩掉（`PROGRESS.md` 缺陷 #1 的复现）。

### 3.3 其余引擎件（半天扫过）

`sequence.py`（状态机）、`sampler.py`（temperature/top-p）、`output.py`。
这些是胶水，混个脸熟即可。

---

## 阶段 4 · 进阶模块（按兴趣选读，各 1 天）

到这里主线已经通了，以下是并联模块，**按求职方向挑**：

| 模块 | 文件 | 先跑 | 精读入口 |
|---|---|---|---|
| 张量并行 | `parallel/layers.py` → `tp_model.py` → `engine_group.py` | `pytest tests/test_parallel.py -v` | `shard_state_dict_tp` 的切分规则表 |
| 投机解码 | `engine/speculative.py` | `pytest tests/test_speculative.py -v` | `_verify` 的接受/纠正/奖励三段 |
| SWAP/Offload | `kv_cache/swap.py` + `manager.py` 的 swap 段 | `pytest tests/test_swap_offload.py -v` | `swap_in` 为什么要求块数严格相等 |
| 集群模拟器 | `simulator/cluster.py` | `examples/run_simulator.py --compare-routers` | `Replica.do_step` 与真实引擎 step 的对应关系 |
| 模型加载 | `models/loader.py` + `tests/helpers_tiny_model.py` | 看测试怎么造随机模型 | 状态名与 HF 的映射 |

---

## 阶段 5 · 贯通验证（1 天）

1. **闭卷**做 `docs/self_test.md` 的 50 题，≥85 分
2. **白板**默画两张图：① block table ↔ 物理 KV 格子图；② schedule() 三阶段流程
3. **脱稿**讲 `interview_qa.md` 的 Q0.1（1 分钟项目介绍），录音回听
4. 通不过的，回到对应阶段补

---

## 常见卡点 FAQ

| 卡点 | 去看 |
|---|---|
| slot / flat_rows 算不明白 | 阶段 1 的格子图，动手画；`models/decoder.py::flat_rows` 只有 5 行 |
| `num_computed_tokens` 与 `len(token_ids)` 分不清 | 阶段 2.3 的不变量：decode 时 `computed == len-1`；`test_kv_growth_invariant` |
| radix 树为什么要 lock | 驱逐只能动"没被运行中序列持有的块"，锁就是持有计数（阶段 2.2 实验） |
| 池子为什么是 `[layers, rows, H, D]` 扁平的 | 把 block_size 维折进行数，index_copy/index_select 都是一维索引，简单且够用 |
| 投机解码为什么 rewind | 验证轮会多写 k+1 个 KV，实际只接受 m 个 → 截回 `len-1+m`（阶段 4） |
| 哪里能用 GPU/NCCL | 引擎在 CUDA torch 下自动用卡；TP 的 gloo 后端可换 NCCL；Triton kernel 需 Linux+GPU |
| 代码在哪跑得快/慢 | 参考实现正确性优先（逐序列注意力）；性能路径见 `attention/triton/` 和 roadmap |

## 配套文档导航

- 卡在机制概念 → `docs/architecture.md`（按模块的参考手册）
- 想知道某个"为什么" → `docs/PROGRESS.md` 的设计决策表 + 缺陷记录
- 背题 → `docs/interview_qa.md`；查漏 → `docs/knowledge_checklist.md`；
  验收 → `docs/self_test.md`
