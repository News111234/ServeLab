# 简历写法指南（含可直接粘贴版本）

> 原则：**诚实定位 + 动词有力 + 数字量化 + JD 关键词全覆盖**。
> 定位写成"参考 vLLM/SGLang 设计的研究型系统"——面试官会按"理解深度"来问，
> 这正是本项目的主场；如果吹成"生产级自研框架"，会被按"线上经验"问，必露馅。
> 每条 bullet 后标注了对应的 `interview_qa.md` 防守题号——**写上去的每一句都要能答**。

## 版本 A：完整版（投推理 Infra / 推理优化岗，放项目经历第一条）

**ServeLab —— 研究型 LLM 推理服务系统**（个人项目，github.com/News111234/ServeLab）
技术栈：Python / PyTorch / Triton / CUDA

- 参考 vLLM/SGLang 设计并独立实现 LLM 推理引擎：PagedAttention 块管理、内容寻址的
  Radix 前缀缓存（可插拔淘汰策略）、Continuous Batching、Chunked Prefill、
  Recompute/Swap 双模式抢占 【防守：Q1.1~Q2.3】
- 实现 Megatron 式张量并行（列/行/词表并行，SPMD 架构，模拟与 gloo 双通信后端），
  TP=2/4 与单卡输出逐 token 一致；实现投机解码（提议-验证-拒绝采样），greedy
  输出与普通解码严格一致 【防守：Q3.1~Q4.3】
- 实现 KV Cache FP8/INT8 量化与 W8A8 线性参考实现；编写 Triton PagedAttention /
  RMSNorm kernel 并与 torch 参考实现对拍 【防守：Q5.1/Q5.2】
- 构建 trace 驱动的集群模拟器（路由策略/输出长度预测/PD 分离/MoE 负载均衡），
  含可校准的解析成本模型与批量实验流水线，支撑缓存淘汰与调度方向的论文研究
  【防守：Q6.1/Q6.2】
- 建立"一致性驱动"的测试体系（62 项测试）：缓存/调度配置组合、TP 与单卡、投机与
  普通解码的输出逐 token 一致作为验收标准，曾交叉验证发现 KV 池 dtype 硬编码等
  5 个真实缺陷 【防守：Q7.1/Q7.2】

## 版本 B：精简版（简历空间紧张 / 投通用后端、AI 平台岗）

**ServeLab —— LLM 推理服务研究系统**（github.com/News111234/ServeLab）

- 独立实现 vLLM 式推理引擎（PagedAttention、前缀缓存、Continuous Batching、
  Chunked Prefill、抢占调度）与 Megatron 式张量并行，62 项测试以"TP=2/4、
  投机解码与单卡/普通解码输出逐 token 一致"为验收标准
- 实现 KV Cache FP8/INT8 量化与 Triton PagedAttention kernel；构建 trace 驱动的
  serving 模拟器（路由/预测/PD 分离/MoE）支撑调度与缓存策略的论文研究

## 一句话版（放技能栏或自我介绍）

> ServeLab：手写 vLLM 式 LLM 推理引擎（分页 KV / 前缀缓存 / 连续批处理 / 张量并行 /
> 投机解码）+ serving 策略模拟器，62 项一致性测试。

## 技能栏配合写法（与项目互相印证）

- 语言：Python（熟练）、C/C++（基础）、CUDA/Triton（能编写与对拍简单 kernel）
- 系统：理解 LLM 推理全栈——PagedAttention / Prefix Caching / Continuous Batching /
  Chunked Prefill / Speculative Decoding / TP 通信分析 / KV 量化
- 工具：PyTorch、torch.distributed、Nsight Systems/Compute、pytest

## 投递侧重建议

| 岗位 | 处理方式 |
|---|---|
| 推理 Infra / 推理优化（本 JD） | 版本 A 放项目经历第一条，技能栏配合 |
| 平台 / 框架开发 | 版本 A 保留但砍掉模拟器条，强调引擎与并行 |
| 算法 / 训练侧 | 版本 B 一条带过，重心让给模型/算法经历 |

## 红线（写错必被问倒）

1. 不写"精通 vLLM 源码"——写"理解其核心机制并有复刻实现"
2. 不写"生产环境部署"——没上线过就是没有
3. 不写"性能提升 X 倍"——模拟器数字不能当实测写；GPU 实测数字拿到后再加
4. 每条 bullet 里的名词（Swap 抢占、词表并行、拒绝采样……）都必须能展开讲 1 分钟
   —— 对照 `interview_qa.md` 的防守题号逐条过
