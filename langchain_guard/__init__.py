"""
langchain_guard - 大模型安全防护框架

基于 LangChain + OpenAI 技术栈的大模型安全防护框架
整合自《大模型安全实战课》36个测试用例文档的完整防护策略

模块总览：
    prompt_guard      - 提示注入检测、输入净化、输入控制门
    output_guard      - 输出内容审核、风险分级、流式监控
    context_guard     - 会话隔离、上下文控制、角色一致性
    privacy_guard     - PII识别脱敏、上下文隐私管理
    dos_guard         - Prompt DoS检测、速率限制、动态降级
    jailbreak_guard   - 越狱检测、角色扮演防护、多轮意图分析
    audit_trail       - 审计日志、不可篡改链、水印溯源、证据导出
    assessment_engine - 红队测试、安全评估、持续监控
    rag_guard         - 安全RAG、信源验证、幻觉检测
    code_guard        - 代码安全扫描、沙箱执行
    industry_guard    - 行业专用防护（医疗/教育/金融/政务/企业/多模态/联邦学习）
    guard_chain       - 防护编排器，组合多层防护
    config            - 统一配置（自定义 API URL、模型名等）
"""

# === 统一配置 ===
from .config import set_llm_config, get_llm_config, create_chat_openai

# === 提示注入防护 ===
from .prompt_guard import (
    PromptGuard,
    PromptInjectionDetector,
    LogSanitizer,
    InputControlGate,
    SessionContextManager,
)

# === 输出内容安全 ===
from .output_guard import (
    OutputGuard,
    ContentSafetyChecker,
    RiskLevel,
    ContentSafetyReviewer,
    RiskResponseHandler,
    StreamingSafetyMonitor,
    KeywordFirewall,
)

# === 上下文安全管理 ===
from .context_guard import (
    ContextGuard,
    SessionIsolationManager,
    ContextControlGate,
    OutputReviewGate,
    SecureSessionManager,
)

# === 隐私保护 ===
from .privacy_guard import (
    PrivacyGuard,
    PIIMasker,
    PIIEntity,
    ExtendedPIIMasker,
    ContextualPrivacyGuard,
)

# === DoS防护 ===
from .dos_guard import (
    PromptDoSDetector,
    RateLimiter,
    DoSGuard,
    PromptDoSGuard,
)

# === 越狱与角色扮演检测 ===
from .jailbreak_guard import (
    MultiTurnIntentDetector,
    RolePlayDetector,
    MultiLanguageInputNormalizer,
    MultiTurnGuard,
    RoleFrozenGuard,
    JailbreakDetector,
    DualSignatureGuard,
    RolePlayGuardrail,
    BehaviorChainAnalyzer,
    SecurityClosedLoop,
    RefusalConsistencyGuard,
    StructuredPromptIsolator,
)

# === 审计追踪 ===
from .audit_trail import (
    AuditLogger,
    AuditCallbackHandler,
    FullChainAuditLogger,
    CircuitBreaker,
    ImmutableAuditChain,
    ContentWatermark,
    ContentWatermarker,
    IncidentEvidenceExporter,
    LogChain,
    OutputSafetyAuditor,
    ImmutableLogChain,
    SecurityRole,
    OrganizationalSecurityOrchestrator,
    RedTeamSimulator,
)

# === 安全评估 ===
from .assessment_engine import (
    RedTeamTester,
    SecurityTestCase,
    TestResult,
    SecurityAssessmentEngine,
    MultiModelSecurityBenchmark,
    ContinuousSecurityMonitor,
    PromptInjectionFilter,
    ContextAwareSecurityGuard,
    TieredSecurityResponse,
)

# === RAG与幻觉防护 ===
from .rag_guard import (
    SecureRAG,
    FunctionCallGuard,
    SourceVerifier,
    RAGWithSourceBinding,
    CrossModelHallucinationDetector,
)

# === 代码安全 ===
from .code_guard import (
    CodeSecurityScanner,
    SandboxExecutor,
    SafeCodeRunner,
)

# === 行业专用防护 ===
from .industry_guard import (
    # 隐私保护扩展
    PrivacyProtectionChain,
    MedicalPrivacyGuard,
    # 有害信息防护
    OutputSensitivityScorer,
    HarmfulContentGuard,
    PublicInfoBoundaryGuard,
    # 内容治理
    ContentGovernanceGuard,
    # 品牌防护
    BrandProtectionGuard,
    BrandIdentityGuard,
    SensitiveInfoGuard,
    BrandSafetyGuard,
    # 对齐技术
    AlignmentGuard,
    AlignmentGuardrail,
    # 数据生命周期
    DataLifecycleGuard,
    DataGuardChain,
    # 教育产品
    ChildContentGuard,
    # 私有大模型
    DataLineageTracker,
    ABACAccessController,
    # 多模态安全
    MultiModalSafetyFilter,
    ModalityIsolationHandler,
    MultiModalOutputAuditor,
    # 联邦学习
    DifferentialPrivacyConfig,
    FederatedGradientProtector,
    NodeReputationSystem,
    SecureAggregationEngine,
    # 模型盗窃防护
    APIBehaviorMonitor,
    ModelWatermark,
    AccessControlGuard,
    # 数据投毒防护
    BackdoorSuppressor,
    DataProvenanceTracker,
)

# === 防护编排 ===
from .guard_chain import GuardChain, SecurityOrchestrator

__all__ = [
    # config
    "set_llm_config", "get_llm_config", "create_chat_openai",
    # prompt_guard
    "PromptGuard", "PromptInjectionDetector", "LogSanitizer",
    "InputControlGate", "SessionContextManager",
    # output_guard
    "OutputGuard", "ContentSafetyChecker", "RiskLevel",
    "ContentSafetyReviewer", "RiskResponseHandler",
    "StreamingSafetyMonitor", "KeywordFirewall",
    # context_guard
    "ContextGuard", "SessionIsolationManager",
    "ContextControlGate", "OutputReviewGate", "SecureSessionManager",
    # privacy_guard
    "PrivacyGuard", "PIIMasker", "PIIEntity",
    "ExtendedPIIMasker", "ContextualPrivacyGuard",
    # dos_guard
    "PromptDoSDetector", "RateLimiter", "DoSGuard", "PromptDoSGuard",
    # jailbreak_guard
    "MultiTurnIntentDetector", "RolePlayDetector", "MultiLanguageInputNormalizer",
    "MultiTurnGuard", "RoleFrozenGuard", "JailbreakDetector",
    "DualSignatureGuard", "RolePlayGuardrail", "BehaviorChainAnalyzer",
    "SecurityClosedLoop", "RefusalConsistencyGuard", "StructuredPromptIsolator",
    # audit_trail
    "AuditLogger", "AuditCallbackHandler", "FullChainAuditLogger",
    "CircuitBreaker", "ImmutableAuditChain", "ContentWatermark",
    "ContentWatermarker", "IncidentEvidenceExporter", "LogChain",
    "OutputSafetyAuditor", "ImmutableLogChain", "SecurityRole",
    "OrganizationalSecurityOrchestrator", "RedTeamSimulator",
    # assessment_engine
    "RedTeamTester", "SecurityTestCase", "TestResult",
    "SecurityAssessmentEngine", "MultiModelSecurityBenchmark",
    "ContinuousSecurityMonitor", "PromptInjectionFilter",
    "ContextAwareSecurityGuard", "TieredSecurityResponse",
    # rag_guard
    "SecureRAG", "FunctionCallGuard", "SourceVerifier",
    "RAGWithSourceBinding", "CrossModelHallucinationDetector",
    # code_guard
    "CodeSecurityScanner", "SandboxExecutor", "SafeCodeRunner",
    # industry_guard
    "PrivacyProtectionChain", "MedicalPrivacyGuard",
    "OutputSensitivityScorer", "HarmfulContentGuard", "PublicInfoBoundaryGuard",
    "ContentGovernanceGuard", "BrandProtectionGuard", "BrandIdentityGuard",
    "SensitiveInfoGuard", "BrandSafetyGuard", "AlignmentGuard", "AlignmentGuardrail",
    "DataLifecycleGuard", "DataGuardChain", "ChildContentGuard",
    "DataLineageTracker", "ABACAccessController",
    "MultiModalSafetyFilter", "ModalityIsolationHandler", "MultiModalOutputAuditor",
    "DifferentialPrivacyConfig", "FederatedGradientProtector",
    "NodeReputationSystem", "SecureAggregationEngine",
    "APIBehaviorMonitor", "ModelWatermark", "AccessControlGuard",
    "BackdoorSuppressor", "DataProvenanceTracker",
    # guard_chain
    "GuardChain", "SecurityOrchestrator",
]
