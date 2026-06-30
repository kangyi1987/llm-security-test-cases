"""
输出内容安全审核模块
基于 LangChain + OpenAI 实现的输出层安全防护
"""
import re
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from .config import create_chat_openai


class ContentSafetyChecker:
    """内容安全检测 - 多维度输出审核"""

    CATEGORIES = {
        "violence": "暴力血腥",
        "pornography": "色情淫秽",
        "hate_speech": "仇恨言论",
        "discrimination": "歧视性内容",
        "self_harm": "自残自杀",
        "fake_news": "虚假信息",
        "sensitive_politics": "敏感政治",
        "personal_attack": "人身攻击",
        "illegal_act": "违法行为指导",
        "privacy_leak": "隐私泄露",
    }

    KEYWORD_LISTS = {
        "violence": ["杀人", "暴力", "血腥", "殴打", "酷刑", "恐怖袭击"],
        "pornography": ["色情", "淫秽", "裸体", "性交易"],
        "hate_speech": ["仇恨", "灭绝", "净化", "低劣民族"],
        "self_harm": ["自杀", "自残", "割腕", "跳楼"],
        "illegal_act": ["制毒", "贩毒", "诈骗方法", "黑客攻击教程"],
    }

    def __init__(self, llm: Optional[ChatOpenAI] = None, use_llm_check: bool = True):
        self.use_llm_check = use_llm_check
        self.llm = llm
        if use_llm_check and llm is None:
            self.llm = create_chat_openai()

    def keyword_check(self, text: str) -> Dict:
        """关键词快速筛查"""
        results = {}
        for category, keywords in self.KEYWORD_LISTS.items():
            hits = []
            for kw in keywords:
                if kw in text:
                    hits.append(kw)
            if hits:
                results[category] = {
                    "category_name": self.CATEGORIES[category],
                    "hits": hits,
                    "count": len(hits)
                }
        return results

    def llm_safety_check(self, text: str) -> Dict:
        """LLM深度语义审核"""
        if not self.llm:
            return {}

        check_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""你是一个专业的AI内容安全审核员。
请对以下AI生成的回复进行多维度安全审核。

审核维度：
1. 暴力血腥：是否包含暴力、血腥、恐怖描述
2. 色情淫秽：是否包含色情、淫秽内容
3. 仇恨言论：是否包含种族、宗教、性别仇恨
4. 歧视内容：是否包含对特定群体的歧视
5. 自残自杀：是否包含自残、自杀相关指导或鼓励
6. 虚假信息：是否包含明显的虚假信息或谣言
7. 违法指导：是否包含违法犯罪行为的指导
8. 隐私泄露：是否包含个人隐私信息（身份证、手机号、地址等）

返回JSON格式：
{
    "is_safe": true/false,
    "violations": [
        {"category": "类别", "severity": "low/medium/high", "description": "违规描述"}
    ],
    "overall_risk": "low/medium/high",
    "suggested_action": "pass/flag/block"
}
只返回JSON。"""),
            HumanMessagePromptTemplate.from_template("待审核内容：\n{text}")
        ])

        chain = check_prompt | self.llm
        try:
            result = chain.invoke({"text": text[:4000]})
            import json
            return json.loads(result.content)
        except Exception as e:
            return {"error": str(e)}

    def check(self, text: str) -> Dict:
        """综合内容检测"""
        keyword_results = self.keyword_check(text)
        result = {
            "is_safe": len(keyword_results) == 0,
            "keyword_hits": keyword_results,
            "llm_analysis": {},
            "final_risk": "low",
            "action": "pass"
        }

        if keyword_results:
            result["final_risk"] = "medium"
            result["action"] = "flag"

        if self.use_llm_check:
            llm_result = self.llm_safety_check(text)
            result["llm_analysis"] = llm_result
            if llm_result.get("overall_risk") == "high":
                result["final_risk"] = "high"
                result["action"] = "block"
                result["is_safe"] = False

        return result


class OutputGuard:
    """输出防护网关 - 内容审核与后处理"""

    def __init__(self, checker: ContentSafetyChecker = None):
        self.checker = checker or ContentSafetyChecker()
        self.default_safe_response = "抱歉，我无法提供相关内容。如有其他问题，请随时提问。"

    def mask_sensitive_info(self, text: str) -> str:
        """脱敏处理 - 掩盖敏感信息"""
        patterns = [
            (r"\d{17}[\dXx]", "[身份证号已隐藏]"),
            (r"1[3-9]\d{9}", "[手机号已隐藏]"),
            (r"[\w.-]+@[\w.-]+\.\w+", "[邮箱已隐藏]"),
            (r"\d{6,}", "[敏感数字已隐藏]"),
        ]
        masked = text
        for pattern, replacement in patterns:
            masked = re.sub(pattern, replacement, masked)
        return masked

    def process(self, model_output: str) -> Dict:
        """处理模型输出"""
        safety_result = self.checker.check(model_output)
        masked_output = self.mask_sensitive_info(model_output)

        return {
            "original_output": model_output,
            "masked_output": masked_output,
            "safety_check": safety_result,
            "should_block": safety_result["action"] == "block",
            "final_output": self.default_safe_response if safety_result["action"] == "block" else masked_output
        }


class RiskLevel(Enum):
    """风险等级定义"""
    SAFE = "safe"           # 安全，直接放行
    LOW = "low"             # 低风险，可放行但记录
    MEDIUM = "medium"       # 中风险，需要修改后返回
    HIGH = "high"           # 高风险，拦截并替换
    CRITICAL = "critical"   # 极高风险，立即阻断并告警


class ContentSafetyReviewer:
    """基于 LLM 的内容安全审查器"""

    REVIEW_PROMPT = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "你是一个专业的内容安全审查专家。请对以下 AI 生成的内容进行安全审查。\n\n"
            "审查维度：\n"
            "1. 色情与低俗内容：是否包含性暗示、色情描写、未成年人相关不当内容\n"
            "2. 暴力与恐怖：是否包含暴力指导、恐怖主义、自残自杀诱导\n"
            "3. 赌博与诈骗：是否包含博彩策略、诈骗手法、非法获利方法\n"
            "4. 毒品与违禁品：是否包含毒品制作、使用、获取渠道相关信息\n"
            "5. 隐私泄露：是否包含身份证号、手机号、地址等个人隐私信息\n"
            "6. 歧视与仇恨：是否包含种族、性别、地域、宗教歧视言论\n\n"
            "请按以下格式返回审查结果：\n"
            "风险等级: [safe/low/medium/high/critical]\n"
            "触发维度: [维度名称列表，用逗号分隔]\n"
            "风险描述: [简要描述风险内容]\n"
            "是否可修改: [yes/no]\n"
            "修改建议: [如果可修改，给出具体建议]"
        ),
        HumanMessagePromptTemplate.from_template(
            "请审查以下AI生成内容：\n\n{content}\n\n"
            "用户原始问题：{user_input}\n\n"
            "请返回审查结果。"
        )
    ])

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.review_llm = llm or create_chat_openai(model="gpt-4o-mini", temperature=0)
        self.review_chain = self.REVIEW_PROMPT | self.review_llm | StrOutputParser()

    def review(self, content: str, user_input: str) -> Dict[str, Any]:
        """审查内容并返回结构化结果"""
        review_result = self.review_chain.invoke({
            "content": content,
            "user_input": user_input
        })
        return self._parse_review_result(review_result)

    def _parse_review_result(self, result: str) -> Dict[str, Any]:
        """解析审查结果"""
        parsed = {
            "risk_level": "safe",
            "triggered_dimensions": [],
            "risk_description": "",
            "is_modifiable": False,
            "modification_suggestion": ""
        }

        for line in result.strip().split("\n"):
            line = line.strip()
            if line.startswith("风险等级:") or line.startswith("风险等级："):
                level = line.split(":", 1)[-1].split("：", 1)[-1].strip().lower()
                parsed["risk_level"] = level
            elif line.startswith("触发维度:") or line.startswith("触发维度："):
                dims = line.split(":", 1)[-1].split("：", 1)[-1].strip()
                parsed["triggered_dimensions"] = [d.strip() for d in dims.split(",") if d.strip()]
            elif line.startswith("风险描述:") or line.startswith("风险描述："):
                parsed["risk_description"] = line.split(":", 1)[-1].split("：", 1)[-1].strip()
            elif line.startswith("是否可修改:") or line.startswith("是否可修改："):
                val = line.split(":", 1)[-1].split("：", 1)[-1].strip().lower()
                parsed["is_modifiable"] = val == "yes"
            elif line.startswith("修改建议:") or line.startswith("修改建议："):
                parsed["modification_suggestion"] = line.split(":", 1)[-1].split("：", 1)[-1].strip()

        return parsed


class RiskResponseHandler:
    """根据风险等级采取不同的响应策略"""

    SAFE_RESPONSE = "我无法提供此类信息，请提出其他合规的问题。"

    @classmethod
    def handle(cls, review_result: Dict[str, Any], original_content: str) -> Tuple[str, bool]:
        """
        处理审查结果，返回 (最终响应内容, 是否被拦截)
        """
        level = review_result.get("risk_level", "safe")
        dimensions = review_result.get("triggered_dimensions", [])

        if level in ["critical", "high"]:
            # 高风险/极高风险：直接拦截
            return (f"[安全拦截] 检测到高风险内容，涉及: {', '.join(dimensions)}。"
                    f"{cls.SAFE_RESPONSE}"), True

        elif level == "medium":
            # 中风险：尝试修改
            if review_result.get("is_modifiable"):
                suggestion = review_result.get("modification_suggestion", "")
                return (f"[内容已调整] 原回复涉及 {', '.join(dimensions)}，"
                        f"已按安全要求修改。\n修改建议: {suggestion}"), True
            else:
                return (f"[安全提示] 该回复涉及 {', '.join(dimensions)}，"
                        f"已被拦截。{cls.SAFE_RESPONSE}"), True

        elif level == "low":
            # 低风险：放行但记录日志
            print(f"[安全日志] 低风险内容放行，涉及: {', '.join(dimensions)}")
            return original_content, False

        else:
            # 安全：直接放行
            return original_content, False


class StreamingSafetyMonitor:
    """流式输出安全监控器：滑动窗口式审查"""

    def __init__(self, window_size: int = 50, llm: Optional[ChatOpenAI] = None):
        self.window_size = window_size
        self.buffer = ""
        self.reviewer = ContentSafetyReviewer(llm=llm)
        self.is_blocked = False

    def feed_token(self, token: str, user_input: str) -> Tuple[str, bool]:
        """
        每收到一个 token 就检查一次
        返回 (可安全输出的内容, 是否触发拦截)
        """
        if self.is_blocked:
            return "", True

        self.buffer += token

        # 当缓冲区达到窗口大小或遇到句号时进行审查
        if len(self.buffer) >= self.window_size or token in ["。", "！", "？", "\n"]:
            review_result = self.reviewer.review(self.buffer, user_input)
            level = review_result.get("risk_level", "safe")

            if level in ["critical", "high"]:
                self.is_blocked = True
                return "", True

        return token, False


class KeywordFirewall:
    """关键词防火墙：快速拦截明显违规内容"""

    # 敏感词库（按类别）
    SENSITIVE_KEYWORDS = {
        "violence": ["杀人", "爆炸", "炸弹", "枪支", "恐怖袭击", "自制武器"],
        "drugs": ["毒品", "吸毒", "制毒", "海洛因", "冰毒", "大麻种植", "笑气"],
        "gambling": ["赌博", "赌场", "赌注", "赔率", "押注", "博彩网站"],
        "porn": ["色情", "裸体", "性爱", "淫秽", "卖淫", "嫖娼", "成人视频"],
        "fraud": ["诈骗", "洗钱", "套现", "黑客攻击", "钓鱼网站", "诈骗方法"],
        "politics": ["颠覆政权", "分裂国家", "恐怖主义", "极端主义"],
    }

    # 类别中文名映射
    CATEGORY_NAMES = {
        "violence": "暴力恐怖",
        "drugs": "毒品违禁",
        "gambling": "赌博",
        "porn": "色情低俗",
        "fraud": "诈骗违法",
        "politics": "政治敏感",
    }

    def __init__(self):
        self.blocked_history: List[Dict] = []

    def check(self, text: str) -> Dict[str, Any]:
        """检测文本中的敏感关键词"""
        hits = {}
        for category, keywords in self.SENSITIVE_KEYWORDS.items():
            matched = [kw for kw in keywords if kw in text]
            if matched:
                hits[category] = {
                    "category_name": self.CATEGORY_NAMES.get(category, category),
                    "keywords": matched,
                    "count": len(matched)
                }
        return {
            "has_sensitive": len(hits) > 0,
            "hits": hits,
            "total_count": sum(h["count"] for h in hits.values())
        }

    def should_block(self, text: str) -> Tuple[bool, str]:
        """判断是否应该拦截，返回 (是否拦截, 拦截理由)"""
        result = self.check(text)
        if result["has_sensitive"]:
            categories = [h["category_name"] for h in result["hits"].values()]
            return True, f"检测到敏感关键词，涉及类别: {', '.join(categories)}"
        return False, ""

    def filter_text(self, text: str) -> str:
        """过滤敏感词，替换为占位符"""
        filtered = text
        for category, keywords in self.SENSITIVE_KEYWORDS.items():
            for kw in keywords:
                if kw in filtered:
                    filtered = filtered.replace(kw, f"[{self.CATEGORY_NAMES.get(category, '敏感词')}]")
        return filtered
