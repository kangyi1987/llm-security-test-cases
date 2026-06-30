"""
统一配置模块 - 支持自定义 API URL 和模型名

环境变量：
    OPENAI_BASE_URL   - 自定义 API 端点（兼容 OpenAI 格式的服务）
    OPENAI_MODEL     - 默认使用的模型名称
    OPENAI_API_KEY   - API Key

使用方式：
    方式一：环境变量（推荐，全局生效）
        export OPENAI_BASE_URL="https://api.example.com/v1"
        export OPENAI_MODEL="gpt-4o-mini"

    方式二：代码设置（仅当前进程）
        from langchain_guard.config import set_llm_config
        set_llm_config(base_url="https://api.example.com/v1", model="gpt-4o-mini")

    方式三：实例参数（仅当前实例）
        detector = PromptInjectionDetector(
            llm=ChatOpenAI(base_url="https://api.example.com/v1", model="gpt-4o-mini")
        )
"""
import os
from typing import Optional


_LLM_CONFIG = {
    "base_url": os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE"),
    "model": os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
    "temperature": 0,
}


def set_llm_config(
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
):
    """设置全局 LLM 配置

    Args:
        base_url: API 基础 URL（如 https://api.deepseek.com/v1）
        model: 模型名称（如 gpt-4o-mini、deepseek-chat 等）
        temperature: 采样温度
    """
    if base_url is not None:
        _LLM_CONFIG["base_url"] = base_url
    if model is not None:
        _LLM_CONFIG["model"] = model
    if temperature is not None:
        _LLM_CONFIG["temperature"] = temperature


def get_llm_config() -> dict:
    """获取当前 LLM 配置"""
    return dict(_LLM_CONFIG)


def create_chat_openai(model: Optional[str] = None, temperature: Optional[float] = None, **kwargs):
    """创建 ChatOpenAI 实例，自动应用全局配置

    Args:
        model: 模型名，不传则使用全局配置
        temperature: 温度，不传则使用全局配置
        **kwargs: 其他传递给 ChatOpenAI 的参数

    Returns:
        ChatOpenAI 实例
    """
    from langchain_openai import ChatOpenAI

    config = get_llm_config()
    params = {
        "model": model or config["model"],
        "temperature": temperature if temperature is not None else config["temperature"],
    }
    if config["base_url"]:
        params["base_url"] = config["base_url"]
    params.update(kwargs)

    return ChatOpenAI(**params)
