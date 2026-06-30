"""
隐私保护模块
基于 LangChain 实现的PII识别与脱敏
"""
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


@dataclass
class PIIEntity:
    """PII实体"""
    type: str
    value: str
    start: int
    end: int
    confidence: float


class PIIMasker:
    """PII脱敏器 - 识别并掩盖个人敏感信息"""

    PII_PATTERNS = {
        "id_card": {
            "pattern": r"\d{17}[\dXx]",
            "mask": "[身份证号]",
            "name": "身份证号"
        },
        "phone": {
            "pattern": r"1[3-9]\d{9}",
            "mask": "[手机号]",
            "name": "手机号"
        },
        "email": {
            "pattern": r"[\w.-]+@[\w.-]+\.\w+",
            "mask": "[邮箱]",
            "name": "邮箱"
        },
        "bank_card": {
            "pattern": r"\d{16,19}",
            "mask": "[银行卡号]",
            "name": "银行卡号"
        },
        "address": {
            "pattern": r"[\u4e00-\u9fa5]{2,}(省|市|区|县|街道|路|号|小区|大厦)",
            "mask": "[地址]",
            "name": "地址"
        },
        "name": {
            "pattern": r"(先生|女士|老师|总|经理)\s*[\u4e00-\u9fa5]{2,3}",
            "mask": "[姓名]",
            "name": "姓名"
        },
    }

    def __init__(self, llm=None):
        self.llm = llm

    def detect(self, text: str) -> List[PIIEntity]:
        """检测文本中的PII实体"""
        entities = []
        for pii_type, config in self.PII_PATTERNS.items():
            for match in re.finditer(config["pattern"], text):
                entities.append(PIIEntity(
                    type=pii_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9
                ))
        return entities

    def mask(self, text: str, mask_types: Optional[List[str]] = None) -> str:
        """脱敏处理"""
        masked = text
        types_to_mask = mask_types or list(self.PII_PATTERNS.keys())
        for pii_type in types_to_mask:
            if pii_type in self.PII_PATTERNS:
                config = self.PII_PATTERNS[pii_type]
                masked = re.sub(config["pattern"], config["mask"], masked)
        return masked

    def llm_deep_detect(self, text: str) -> List[PIIEntity]:
        """LLM深度检测 - 识别更复杂的隐私信息"""
        if not self.llm:
            return []

        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的隐私信息检测助手。
请从以下文本中识别出所有可能的个人隐私信息（PII）。

需要识别的类型：
- 个人身份信息：姓名、年龄、生日
- 联系方式：手机号、邮箱、微信号
- 位置信息：住址、工作单位、学校
- 财务信息：银行卡、支付宝、微信支付账号
- 敏感属性：健康状况、宗教信仰、政治面貌

返回JSON格式：
{"entities": [{"type": "类型", "value": "值", "confidence": 0.9}]}
只返回JSON。"""),
            ("human", "文本：{text}")
        ])

        try:
            result = self.llm.invoke(prompt.format(text=text[:2000]))
            import json
            data = json.loads(result.content)
            entities = []
            for e in data.get("entities", []):
                idx = text.find(e["value"])
                if idx >= 0:
                    entities.append(PIIEntity(
                        type=e["type"],
                        value=e["value"],
                        start=idx,
                        end=idx + len(e["value"]),
                        confidence=e.get("confidence", 0.8)
                    ))
            return entities
        except Exception:
            return []


class PrivacyGuard:
    """隐私防护网关 - 输入输出双向隐私保护"""

    def __init__(self, masker: PIIMasker = None):
        self.masker = masker or PIIMasker()
        self.input_pii_log = []
        self.output_pii_log = []

    def protect_input(self, user_input: str) -> Dict:
        """输入隐私保护 - 检测并记录"""
        entities = self.masker.detect(user_input)
        masked_input = self.masker.mask(user_input)
        self.input_pii_log.append({
            "input": user_input,
            "detected_count": len(entities),
            "entities": [{"type": e.type, "value": e.value} for e in entities]
        })
        return {
            "original": user_input,
            "masked": masked_input,
            "pii_detected": len(entities) > 0,
            "pii_count": len(entities),
            "entities": entities
        }

    def protect_output(self, model_output: str) -> Dict:
        """输出隐私保护 - 确保不泄露"""
        entities = self.masker.detect(model_output)
        masked_output = self.masker.mask(model_output)
        self.output_pii_log.append({
            "output": model_output,
            "detected_count": len(entities)
        })
        return {
            "original": model_output,
            "masked": masked_output,
            "has_leakage": len(entities) > 0,
            "leak_count": len(entities),
            "entities": entities
        }


class ExtendedPIIMasker(PIIMasker):
    """扩展版PII脱敏器 - 增加更多PII类型和检测方式"""

    def __init__(self, llm=None):
        super().__init__(llm)
        self.PII_PATTERNS.update({
            "wechat_id": {
                "pattern": r"微信[：:]\s*[a-zA-Z0-9_-]{5,}",
                "mask": "[微信号]",
                "name": "微信号"
            },
            "qq_number": {
                "pattern": r"QQ[：:]\s*\d{5,12}",
                "mask": "[QQ号]",
                "name": "QQ号"
            },
            "company_name": {
                "pattern": r"[\u4e00-\u9fa5]{2,}(有限公司|股份有限公司|集团|科技公司)",
                "mask": "[企业名称]",
                "name": "企业名称"
            },
            "medical_record": {
                "pattern": r"(病历号|病案号|住院号)[：:]\s*\d+",
                "mask": "[病历号]",
                "name": "病历号"
            },
        })

    def detect_contextual_pii(self, text: str, context: List[BaseMessage] = None) -> List[Dict]:
        """上下文相关PII检测 - 检测多轮对话中泄露的隐私"""
        findings = []
        if not context:
            return findings

        context_text = "\n".join([m.content for m in context if isinstance(m, HumanMessage)])

        pii_in_context = self.detect(context_text)
        current_pii = self.detect(text)

        for pii in current_pii:
            if pii.value not in context_text:
                findings.append({
                    "entity": pii,
                    "source": "current_input",
                    "risk_level": "medium"
                })

        return findings


class ContextualPrivacyGuard:
    """上下文隐私防护器 - 管理多轮对话中的隐私边界"""

    def __init__(self, max_pii_retention_turns: int = 3):
        self.masker = ExtendedPIIMasker()
        self.max_pii_retention_turns = max_pii_retention_turns
        self.session_pii: Dict[str, List[Dict]] = {}

    def check_output_leakage(self, session_id: str, output: str) -> Dict:
        """检查模型输出是否泄露了用户输入的敏感信息"""
        session_pii_list = self.session_pii.get(session_id, [])
        leaked_items = []

        for pii_record in session_pii_list:
            if pii_record["value"] in output:
                leaked_items.append(pii_record)

        return {
            "has_leakage": len(leaked_items) > 0,
            "leaked_count": len(leaked_items),
            "leaked_items": leaked_items
        }

    def register_session_pii(self, session_id: str, user_input: str) -> None:
        """记录会话中出现的PII，用于后续输出检查"""
        pii_entities = self.masker.detect(user_input)
        if session_id not in self.session_pii:
            self.session_pii[session_id] = []

        for entity in pii_entities:
            self.session_pii[session_id].append({
                "type": entity.type,
                "value": entity.value,
                "registered_at": __import__("time").time()
            })

    def clean_expired_pii(self, session_id: str, current_turn: int) -> int:
        """清理过期的PII记录（超过保留轮次的）"""
        if session_id not in self.session_pii:
            return 0

        original_count = len(self.session_pii[session_id])
        if current_turn > self.max_pii_retention_turns:
            self.session_pii[session_id] = []

        return original_count - len(self.session_pii[session_id])
