"""
拒绝服务攻击（DoS）防护模块
基于 LangChain 实现的 Prompt DoS 检测与防御

本模块整合了来自多个测试用例的 DoS 防护组件：
- PromptDoSDetector: Prompt DoS 多维度风险评分检测器（循环/嵌套/角色劫持/超长输出）
- RateLimiter: 用户级请求速率限制器（并发/分钟/日配额管控）
- DoSGuard: DoS 防护网关，整合检测器与限流器，支持拦截与降级
- PromptDoSGuard: Prompt DoS 防护守卫，五维检测（长度/嵌套/重复/特殊字符/速率）
"""
import time
import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, deque
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from .config import create_chat_openai


class PromptDoSDetector:
    """Prompt DoS 检测器 - 多维度风险评分"""

    DOS_PATTERNS = {
        "infinite_loop": [
            r"不断.*?重新", r"一直.*?直到", r"不停.*?继续",
            r"反复.*?重复", r"无限.*?生成", r"循环.*?执行",
            r"直到.*?说停", r"一直.*?下去"
        ],
        "nested_tasks": [
            r"第.+步骤", r"每.*个.*?再", r"每个.*?下面",
            r"分别.*?每个", r"依次.*?再", r"分层.*?逐级"
        ],
        "role_hijack": [
            r"你现在是.*?任务引擎", r"你是一个.*?执行器",
            r"作为.*?处理器", r"你的工作就是.*?执行"
        ],
        "massive_output": [
            r"五万字", r"十万字", r"一万字以上",
            r"越多越好", r"尽可能详细", r"非常详细"
        ]
    }

    def __init__(self, max_input_tokens: int = 4000,
                 max_output_tokens: int = 2000,
                 risk_threshold: float = 7.0):
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.risk_threshold = risk_threshold

    def check_input_length(self, prompt: str) -> Dict:
        """检查输入长度"""
        char_count = len(prompt)
        estimated_tokens = char_count // 2

        return {
            "char_count": char_count,
            "estimated_tokens": estimated_tokens,
            "exceeds_limit": estimated_tokens > self.max_input_tokens,
            "limit": self.max_input_tokens
        }

    def check_structural_analysis(self, prompt: str) -> Dict:
        """结构分析 - 检测嵌套任务复杂度"""
        score = 0.0
        findings = []

        for category, patterns in self.DOS_PATTERNS.items():
            category_hits = 0
            for pattern in patterns:
                if re.search(pattern, prompt):
                    category_hits += 1
                    findings.append(f"{category}:{pattern}")

            if category_hits > 0:
                if category == "infinite_loop":
                    score += category_hits * 2.5
                elif category == "nested_tasks":
                    score += category_hits * 2.0
                elif category == "role_hijack":
                    score += category_hits * 1.5
                elif category == "massive_output":
                    score += category_hits * 2.0

        nesting_depth = self._count_nesting_depth(prompt)
        if nesting_depth >= 3:
            score += (nesting_depth - 2) * 1.0
            findings.append(f"嵌套层级过深: {nesting_depth}层")

        return {
            "structural_score": min(10.0, score),
            "findings": findings,
            "nesting_depth": nesting_depth
        }

    def _count_nesting_depth(self, prompt: str) -> int:
        """估算任务嵌套深度"""
        depth = 0
        max_depth = 0
        lines = prompt.split('\n')

        for line in lines:
            stripped = line.strip()
            if re.match(r'^[0-9]+[.、]', stripped) or re.match(r'^[一二三四五六七八九十]+[.、]', stripped):
                depth += 1
                max_depth = max(max_depth, depth)
            elif stripped and depth > 0:
                depth = max(0, depth - 1)

        return max_depth

    def calculate_risk_score(self, prompt: str) -> Dict:
        """综合DoS风险评分"""
        length_check = self.check_input_length(prompt)
        structural_check = self.check_structural_analysis(prompt)

        total_score = 0.0

        if length_check["exceeds_limit"]:
            ratio = length_check["estimated_tokens"] / self.max_input_tokens
            total_score += min(4.0, ratio * 3.0)

        total_score += structural_check["structural_score"]

        task_count = len(re.findall(r'[0-9]+[.、]', prompt))
        if task_count >= 5:
            total_score += min(2.0, (task_count - 4) * 0.5)

        risk_level = "low"
        should_block = False
        should_degrade = False

        if total_score >= self.risk_threshold:
            risk_level = "high"
            should_block = True
        elif total_score >= 4.0:
            risk_level = "medium"
            should_degrade = True

        return {
            "total_score": min(10.0, round(total_score, 1)),
            "risk_level": risk_level,
            "should_block": should_block,
            "should_degrade": should_degrade,
            "length_check": length_check,
            "structural_check": structural_check
        }


class RateLimiter:
    """速率限制器 - 用户级请求管控"""

    def __init__(self, max_requests_per_minute: int = 10,
                 max_concurrent: int = 3,
                 max_daily_tokens: int = 100000):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_concurrent = max_concurrent
        self.max_daily_tokens = max_daily_tokens

        self.user_request_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.user_concurrent: Dict[str, int] = defaultdict(int)
        self.user_daily_tokens: Dict[str, int] = defaultdict(int)
        self.user_last_reset: Dict[str, float] = defaultdict(float)
        self.blocked_users: Dict[str, float] = {}

    def check_rate_limit(self, user_id: str) -> Dict:
        """检查速率限制"""
        now = time.time()
        one_minute_ago = now - 60

        recent = [t for t in self.user_request_times[user_id] if t > one_minute_ago]

        minute_count = len(recent)

        daily_tokens = self.user_daily_tokens[user_id]
        concurrent = self.user_concurrent[user_id]

        is_blocked = user_id in self.blocked_users and self.blocked_users[user_id] > now

        return {
            "is_blocked": is_blocked,
            "requests_last_minute": minute_count,
            "minute_limit": self.max_requests_per_minute,
            "concurrent_requests": concurrent,
            "concurrent_limit": self.max_concurrent,
            "daily_tokens_used": daily_tokens,
            "daily_token_limit": self.max_daily_tokens,
            "exceeds_minute_limit": minute_count >= self.max_requests_per_minute,
            "exceeds_concurrent_limit": concurrent >= self.max_concurrent,
            "exceeds_daily_limit": daily_tokens >= self.max_daily_tokens
        }

    def record_request(self, user_id: str, estimated_tokens: int = 100) -> bool:
        """记录请求，返回是否允许"""
        check = self.check_rate_limit(user_id)
        if check["exceeds_minute_limit"] or check["exceeds_concurrent_limit"]:
            return False

        now = time.time()
        self.user_request_times[user_id].append(now)
        self.user_concurrent[user_id] += 1
        self.user_daily_tokens[user_id] += estimated_tokens
        return True

    def complete_request(self, user_id: str):
        """完成请求，释放并发槽位"""
        if self.user_concurrent[user_id] > 0:
            self.user_concurrent[user_id] -= 1

    def block_user(self, user_id: str, duration_seconds: int = 300):
        """临时封禁用户"""
        self.blocked_users[user_id] = time.time() + duration_seconds


class DoSGuard:
    """DoS防护网关"""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or create_chat_openai()
        self.detector = PromptDoSDetector()
        self.rate_limiter = RateLimiter()

        self.block_responses = [
            "请求过于复杂，请简化您的问题，以便我更好地帮助您。",
            "这个任务规模较大，建议您拆分成几个小问题分别提问。",
            "为了保证服务质量，单次请求内容不宜过长，请精简后重试。"
        ]

        self.degrade_responses = [
            "我先给您一个简要版本：\n",
            "由于内容较多，这里是摘要：\n",
            "考虑到篇幅，这里简要回复如下：\n"
        ]

    def build_guard_runnable(self):
        """构建防护组件"""
        return RunnableLambda(self._guard_process)

    def _guard_process(self, inputs: Dict) -> Dict:
        """防护处理"""
        user_id = inputs.get("user_id", "unknown")
        prompt = inputs.get("prompt", "")

        rate_check = self.rate_limiter.check_rate_limit(user_id)
        if rate_check["is_blocked"]:
            return {
                "blocked": True,
                "reason": "用户已被临时封禁，请稍后再试"
            }

        if rate_check["exceeds_minute_limit"]:
            return {
                "blocked": True,
                "reason": "请求过于频繁，请稍后再试"
            }

        dos_score = self.detector.calculate_risk_score(prompt)

        if dos_score["should_block"]:
            return {
                "blocked": True,
                "reason": f"检测到高资源消耗风险（风险分：{dos_score['total_score']}/10）",
                "risk_score": dos_score,
                "suggestion": "请简化问题或拆分为多个小问题"
            }

        if dos_score["should_degrade"]:
            return {
                "blocked": False,
                "degraded": True,
                "risk_score": dos_score,
                "max_tokens": 500,
                "mode": "summary"
            }

        self.rate_limiter.record_request(user_id)

        return {
            "blocked": False,
            "degraded": False,
            "risk_score": dos_score,
            "proceed": True,
            "max_tokens": self.detector.max_output_tokens
        }


class PromptDoSGuard:
    """Prompt DoS 防护守卫"""

    def __init__(self):
        self.max_input_length = 8000      # 最大输入长度
        self.max_nesting_depth = 5        # 最大嵌套深度
        self.max_repetition_ratio = 0.6   # 最大重复率
        self.max_special_char_ratio = 0.3 # 最大特殊字符比例
        self.rate_limit_window = 60       # 速率限制窗口（秒）
        self.max_requests_per_window = 30 # 窗口内最大请求数
        self.request_history = {}         # 请求历史记录

    def check_input_length(self, user_input: str) -> Tuple[bool, str]:
        """检查输入长度"""
        if len(user_input) > self.max_input_length:
            return False, f"输入长度超过限制 ({len(user_input)} > {self.max_input_length})"
        return True, "OK"

    def check_nesting_depth(self, user_input: str) -> Tuple[bool, str]:
        """检查嵌套深度"""
        # 检测引号嵌套、括号嵌套等
        depth = 0
        max_depth = 0
        for char in user_input:
            if char in '([{（【「':
                depth += 1
                max_depth = max(max_depth, depth)
            elif char in ')]}）】」':
                depth -= 1
        if max_depth > self.max_nesting_depth:
            return False, f"嵌套深度超过限制 ({max_depth} > {self.max_nesting_depth})"
        return True, "OK"

    def check_repetition(self, user_input: str) -> Tuple[bool, str]:
        """检查重复内容比例"""
        if len(user_input) < 100:
            return True, "OK"
        # 将文本分块，检查重复块比例
        chunk_size = 50
        chunks = [user_input[i:i+chunk_size] for i in range(0, len(user_input)-chunk_size, chunk_size)]
        unique_chunks = set(chunks)
        if len(chunks) > 0:
            repetition_ratio = 1 - (len(unique_chunks) / len(chunks))
            if repetition_ratio > self.max_repetition_ratio:
                return False, f"内容重复率过高 ({repetition_ratio:.2%} > {self.max_repetition_ratio:.2%})"
        return True, "OK"

    def check_special_chars(self, user_input: str) -> Tuple[bool, str]:
        """检查特殊字符比例"""
        if len(user_input) < 50:
            return True, "OK"
        special_count = len(re.findall(r'[^\w\s\u4e00-\u9fff]', user_input))
        ratio = special_count / len(user_input)
        if ratio > self.max_special_char_ratio:
            return False, f"特殊字符比例过高 ({ratio:.2%} > {self.max_special_char_ratio:.2%})"
        return True, "OK"

    def check_rate_limit(self, user_id: str) -> Tuple[bool, str]:
        """检查请求速率"""
        now = time.time()
        if user_id not in self.request_history:
            self.request_history[user_id] = []
        # 清理过期记录
        self.request_history[user_id] = [
            t for t in self.request_history[user_id]
            if now - t < self.rate_limit_window
        ]
        self.request_history[user_id].append(now)
        if len(self.request_history[user_id]) > self.max_requests_per_window:
            return False, "请求频率过高，请稍后再试"
        return True, "OK"

    def full_check(self, user_input: str, user_id: str = "default") -> Dict:
        """完整检查"""
        checks = [
            ("长度检查", self.check_input_length(user_input)),
            ("嵌套深度", self.check_nesting_depth(user_input)),
            ("重复率", self.check_repetition(user_input)),
            ("特殊字符", self.check_special_chars(user_input)),
            ("速率限制", self.check_rate_limit(user_id)),
        ]
        failed = [(name, msg) for name, (ok, msg) in checks if not ok]
        return {
            "passed": len(failed) == 0,
            "failed_checks": failed,
            "message": "安全检查通过" if len(failed) == 0 else f"安全检查失败: {'; '.join(f'{n}:{m}' for n, m in failed)}"
        }


__all__ = [
    "PromptDoSDetector",
    "RateLimiter",
    "DoSGuard",
    "PromptDoSGuard",
]
