# 大模型安全测试用例集

> 基于「大模型安全实战课」36个文档生成，覆盖测试用例 + LangChain 防护策略
>
> **langchain_guard 框架**：13 个模块 · 104 个防护类 · 七层纵深防御

---

## 目录结构

```
大模型安全测试用例集/
├── README.md                          # 本文件（索引）
├── langchain_guard/                   # LangChain安全防护核心框架
│   ├── __init__.py                    # 统一导出（90+ 个类）
│   ├── prompt_guard.py                # 提示注入检测 + 输入过滤 + 输入控制门
│   ├── output_guard.py                # 输出内容安全审核 + 风险分级 + 流式监控
│   ├── context_guard.py               # 会话隔离 + 角色一致性 + 上下文控制门
│   ├── privacy_guard.py               # PII识别 + 隐私脱敏 + 上下文隐私
│   ├── dos_guard.py                   # Prompt DoS检测 + 速率限制 + 动态降级
│   ├── jailbreak_guard.py             # 越狱检测 + 角色扮演防护 + 多轮意图
│   ├── audit_trail.py                 # 审计日志 + 不可篡改链 + 水印 + 证据导出
│   ├── assessment_engine.py           # 红队测试 + 安全评估 + 持续监控
│   ├── rag_guard.py                   # 安全RAG + 信源验证 + 幻觉检测
│   ├── code_guard.py                  # 代码安全扫描 + 沙箱执行
│   ├── industry_guard.py              # 行业专用防护（31个类）
│   └── guard_chain.py                 # 七层纵深防御编排器
├── 01_安全基础/                        # 6个测试用例
│   ├── 开篇词_大模型安全基本功_测试用例.md
│   ├── 00_课前热身_测试用例.md
│   ├── 01_初识安全_测试用例.md
│   ├── 02_模型机制_测试用例.md
│   ├── 03_风险类型_测试用例.md
│   └── 04_安全架构_测试用例.md
├── 02_攻击手法/                        # 8个测试用例
│   ├── 05_提示注入攻防战上_测试用例.md
│   ├── 06_提示注入攻防战下_测试用例.md
│   ├── 07_数据投毒与后门_测试用例.md
│   ├── 08_模型盗窃_测试用例.md
│   ├── 09_敏感信息防范_测试用例.md
│   ├── 10_有害信息生成_测试用例.md
│   ├── 11_拒绝服务攻击_测试用例.md
│   └── 12_绕过安全防护_测试用例.md
├── 03_防御机制/                        # 6个测试用例
│   ├── 13_提示词过滤净化_测试用例.md
│   ├── 14_输出内容把关_测试用例.md
│   ├── 15_内容治理_测试用例.md
│   ├── 16_舆情与品牌防护_测试用例.md
│   ├── 17_对齐技术_测试用例.md
│   └── 18_数据生命周期安全_测试用例.md
└── 04_行业应用/                        # 16个测试用例
    ├── 19_安全运维_测试用例.md
    ├── 20_红队测试_测试用例.md
    ├── 21_聊天助手安全_测试用例.md
    ├── 22_编程助手安全_测试用例.md
    ├── 23_搜索问答幻觉_测试用例.md
    ├── 24_教育产品保护_测试用例.md
    ├── 25_行业私有大模型_测试用例.md
    ├── 26_可解释性与审计_测试用例.md
    ├── 27_融合模型安全_测试用例.md
    ├── 28_联邦学习安全_测试用例.md
    ├── 29_安全评估体系_测试用例.md
    ├── 30_模拟攻防演练_测试用例.md
    ├── 31_简易安全过滤器_测试用例.md
    ├── 32_企业级安全加固_测试用例.md
    ├── 33_未来趋势演进_测试用例.md
    └── 34_结束语_测试用例.md
```

---

## 快速开始

### 安装依赖

```bash
pip install langchain langchain-openai langchain-core
```

### 使用 langchain_guard 框架

```python
from langchain_guard import SecurityOrchestrator
from langchain_openai import ChatOpenAI

# 1. 创建七层纵深防御编排器
guard = SecurityOrchestrator(
    system_prompt="你是一个专业的客服助手",
    llm=ChatOpenAI(model="gpt-4", temperature=0),
    enable_dos_guard=True,        # 启用DoS防护
    enable_jailbreak_guard=True,  # 启用越狱检测
    enable_audit_trail=True       # 启用审计日志
)

# 2. 创建安全会话
session_id = guard.create_session(user_id="user_001")

# 3. 安全聊天（自动经过七层防护）
result = guard.chat(session_id, "你好，请问有什么可以帮助的？")
print(result["response"])
print("安全检测项:", result["security_checks"].keys())
```

### 单独使用某个防护模块

```python
from langchain_guard import (
    DoSGuard, JailbreakDetector, AuditLogger,
    CodeSecurityScanner, SandboxExecutor,
    MedicalPrivacyGuard, ChildContentGuard
)

# DoS检测
dos = DoSGuard()
result = dos._guard_process({"user_id": "test", "prompt": "用户输入..."})

# 代码安全扫描
scanner = CodeSecurityScanner()
issues = scanner.scan_code("import os; os.system('rm -rf /')")
```

### 配置自定义 API 和模型

langchain_guard 支持三种方式配置自定义 OpenAI 兼容 API（如 DeepSeek、Qwen、Azure OpenAI 等）。

**方式一：环境变量（推荐）**

```bash
# Windows PowerShell
$env:OPENAI_BASE_URL = "https://api.deepseek.com/v1"
$env:OPENAI_MODEL = "deepseek-chat"
$env:OPENAI_API_KEY = "sk-xxx"
```

```bash
# Linux / macOS
export OPENAI_BASE_URL="https://api.deepseek.com/v1"
export OPENAI_MODEL="deepseek-chat"
```

**方式二：代码全局设置**

```python
from langchain_guard import set_llm_config

set_llm_config(
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
    temperature=0
)

# 之后创建的所有防护模块都会自动使用该配置
from langchain_guard import PromptInjectionDetector
detector = PromptInjectionDetector()
```

**方式三：实例级配置（传入自定义 llm）**

```python
from langchain_openai import ChatOpenAI
from langchain_guard import SecurityOrchestrator

llm = ChatOpenAI(
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
    api_key="sk-xxx",
    temperature=0
)

guard = SecurityOrchestrator(
    system_prompt="你是一个客服助手",
    llm=llm
)
```

---

## langchain_guard 模块详解

### 模块总览

| 模块 | 类数 | 核心能力 |
|------|------|---------|
| prompt_guard | 5 | 提示注入检测（规则+LLM双层）、输入净化、输入控制门、日志脱敏 |
| output_guard | 7 | 内容审核（10大类）、风险分级、流式监控、关键词防火墙 |
| context_guard | 5 | 会话隔离、角色一致性检测、上下文控制门、输出审查门 |
| privacy_guard | 5 | PII识别（8类）、脱敏处理、上下文隐私管理 |
| dos_guard | 4 | Prompt DoS多维度检测、速率限制、动态降级 |
| jailbreak_guard | 12 | 越狱检测、角色扮演防护、多轮意图、双签名门禁 |
| audit_trail | 14 | 审计日志、不可篡改哈希链、内容水印、证据包导出 |
| assessment_engine | 9 | 红队测试框架、安全评估引擎、持续监控 |
| rag_guard | 6 | 安全RAG、信源验证、多模型幻觉交叉检测 |
| code_guard | 4 | AST静态扫描、沙箱执行、安全代码运行器 |
| industry_guard | 31 | 医疗/教育/金融/政务/企业/多模态/联邦学习 |
| guard_chain | 2 | 七层防护编排器、安全协同编排 |
| **合计** | **104** | **覆盖大模型安全全场景** |

### 七层纵深防御

```
用户输入
  → 第1层「DoS防护」DoSGuard
    - 速率限制（RateLimiter）
    - Prompt DoS多维度风险评分
    - 动态降级机制
  → 第2层「输入过滤」PromptGuard
    - 正则规则快速检测（SUSPICIOUS_PATTERNS）
    - LLM深度语义检测（llm_based_check）
    - 输入净化（sanitize）
    - 输入控制门（InputControlGate）
  → 第3层「越狱防护」JailbreakDetector
    - 越狱模板匹配
    - 角色扮演检测
    - 多语言输入归一化
    - 双签名门禁验证
  → 第4层「隐私保护」PrivacyGuard
    - PII识别（身份证、手机号、邮箱、银行卡、地址等）
    - 输入脱敏处理
    - 上下文隐私管理
  → 第5层「上下文控制」ContextGuard
    - 多会话隔离（SessionIsolationManager）
    - 角色一致性检测（check_role_consistency）
    - 上下文压缩净化（compact_context + sanitize_context）
    - 上下文可信度评分
  → 第6层「LLM推理」
    - 强化版 System Prompt（安全规则注入）
  → 第7层「输出审核」OutputGuard
    - 关键词+LLM双层内容安全检测
    - PII泄露检测
    - 敏感信息脱敏
    - 输出审查门（OutputReviewGate）
  → 审计追踪层「AuditLogger」
    - 全链路日志记录
    - 不可篡改哈希链
    - 安全事件告警
  → 最终输出
```

### 行业专用防护（industry_guard）

| 行业/领域 | 核心类 | 适用场景 |
|-----------|--------|---------|
| 医疗 | MedicalPrivacyGuard、OutputSensitivityScorer | 医疗咨询、病历分析 |
| 教育 | ChildContentGuard、AlignmentGuard | 教育产品、儿童保护 |
| 金融 | ABACAccessController、SensitiveInfoGuard | 金融客服、投顾助手 |
| 政务 | LogChain、DataLineageTracker | 政务咨询、办事助手 |
| 企业 | BrandIdentityGuard、BrandSafetyGuard | 企业知识库、内部助手 |
| 多模态 | MultiModalSafetyFilter、MultiModalOutputAuditor | 图文音视频多模态 |
| 联邦学习 | DifferentialPrivacyConfig、FederatedGradientProtector | 分布式训练 |
| 数据安全 | DataLifecycleGuard、BackdoorSuppressor | 训练数据安全 |
| 模型保护 | ModelWatermark、APIBehaviorMonitor | 模型知识产权 |

---

## 测试用例一览

| 编号 | 文档主题 | 风险等级 | 核心威胁 |
|------|---------|---------|---------|
| 开篇词 | 大模型安全基本功 | 高 | 越狱、敏感内容生成、合规风险 |
| 00 | 课前热身 | 中 | 安全意识薄弱、误判风险 |
| 01 | 初识安全 | 高 | DAN越狱、幻觉误导、数据泄露 |
| 02 | 模型运行机制 | 高 | Token注入、上下文溢出、RAG投毒 |
| 03 | 8类高频威胁 | 极高 | 提示注入、幻觉、有害言论、隐私泄露、数据投毒、对抗性提示、越狱、模型逆向 |
| 04 | 安全架构边界 | 高 | 输入控制门、上下文控制门、输出审查门 |
| 05 | 提示注入攻防（上） | 极高 | 正向覆盖、逆向诱导、多轮渐进、模板注入 |
| 06 | 提示注入攻防（下） | 极高 | 防御机制、攻防演练、红队测试 |
| 07 | 数据投毒与后门 | 极高 | 污染输出、污染行为、行为后门 |
| 08 | 模型盗窃与逆向 | 极高 | 模型逆向、参数反推、产权侵犯 |
| 09 | 敏感信息防范 | 极高 | 训练数据泄露、上下文泄露、多租户隔离 |
| 10 | 有害信息生成 | 极高 | 违法内容、暴力言论、仇恨言论 |
| 11 | 拒绝服务攻击 | 高 | Prompt DoS、资源耗尽、递归攻击 |
| 12 | 绕过安全防护 | 极高 | 语义绕过、角色扮演、翻译绕过、工具联动 |
| 13 | 提示词过滤净化 | 高 | 提示泄露、间接注入、语义伪装 |
| 14 | 输出内容把关 | 高 | 违规生成、擦边内容、敏感信息 |
| 15 | 内容伦理治理 | 高 | 黄赌毒、伦理边界、价值观冲突 |
| 16 | 舆情与品牌防护 | 高 | 品牌负面、舆情危机、公关损毁 |
| 17 | 对齐技术 | 高 | 价值观偏移、行为失控、奖励欺骗 |
| 18 | 数据生命周期安全 | 高 | 采集污染、存储泄露、销毁不彻底 |
| 19 | 安全运维（SRE） | 高 | 监控盲区、应急响应滞后、日志缺失 |
| 20 | 红队测试与审计 | 高 | 安全盲区、测试不充分、审计缺失 |
| 21 | 聊天助手安全 | 极高 | 越狱、角色漂移、权限绕过 |
| 22 | 编程助手安全 | 极高 | 代码注入、权限漏洞、沙箱逃逸 |
| 23 | 搜索问答幻觉 | 高 | 事实错误、编造信息、权威混淆 |
| 24 | 教育产品保护 | 极高 | 儿童暴露、不当内容、隐私泄露 |
| 25 | 行业私有大模型 | 极高 | 数据隔离、合规要求、供应链攻击 |
| 26 | 可解释性与审计 | 中 | 决策黑箱、责任追溯、合规审计 |
| 27 | 融合模型安全 | 高 | 多模态攻击、跨模态注入、融合漏洞 |
| 28 | 联邦学习安全 | 高 | 梯度泄露、投毒攻击、模型反演 |
| 29 | 安全评估体系 | 高 | 评估标准缺失、指标不统一、误判 |
| 30 | 模拟攻防演练 | 极高 | 实战盲区、响应延迟、方案缺陷 |
| 31 | 简易安全过滤器 | 中 | 关键词绕过、语义模糊化、组合攻击 |
| 32 | 企业安全加固 | 极高 | 身份穿帮、信息泄露、全链条失控 |
| 33 | 未来趋势演进 | 高 | 行为层攻击、角色植入、单点防御断裂 |
| 34 | 结束语 | 中 | 认知滞后、组织缺失、责任追溯 |

---

## 防护策略技术栈

所有防护代码基于以下技术栈：

| 组件 | 库 | 用途 |
|------|-----|------|
| LLM调用 | `langchain_openai.ChatOpenAI` | 安全检测、语义审核 |
| 提示词模板 | `langchain_core.prompts` | 安全检测提示构建 |
| 链式编排 | `langchain_core.runnables` | 多层防护链组合 |
| 输出解析 | `langchain_core.output_parsers` | 结构化结果解析 |
| 嵌入模型 | `langchain_openai.OpenAIEmbeddings` | 语义相似度检测 |
| 回调系统 | `langchain_core.callbacks` | 审计追踪回调 |

---

## 文件统计

| 类别 | 数量 |
|------|------|
| 测试用例文档 | 36 个（覆盖全部课程内容） |
| langchain_guard 模块 | 13 个 Python 文件 |
| 防护类总数 | 104 个 |
| 总攻击测试用例 | 150+ 个（含具体 Prompt） |
| 无不适宜内容被跳过 | 所有文档均为正常安全技术内容 |

---

## 注意事项

1. 运行测试用例需设置 `OPENAI_API_KEY` 环境变量
2. 部分测试用例的 Prompt 仅用于安全测试，请勿在生产环境对真实用户使用
3. langchain_guard 模块可直接集成到实际项目中
4. 建议定期更新各模块的规则库（正则模式、关键词列表等）
5. `code_guard.py` 中的 `SandboxExecutor` 在 Windows 上部分功能受限（如 `resource` 模块不可用）
6. 生产环境部署建议结合实际业务场景选择启用哪些防护层，平衡安全与性能
