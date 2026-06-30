"""
代码安全扫描与沙箱执行模块
基于 AST 静态分析与隔离执行实现 AI 生成代码的安全防护
包含危险函数检测、命令注入模式扫描、沙箱隔离执行等核心能力

来源：
- 04_行业应用/22_编程助手安全_测试用例.md（CodeSecurityScanner、SandboxExecutor、SafeCodeRunner）

注意：
- SandboxExecutor 依赖 `resource` 模块，该模块仅在 Unix/Linux 平台可用，
  在 Windows 上导入本模块会抛出 ImportError。如需在 Windows 使用，请自行
  移除 `import resource` 或替换为跨平台实现。
"""
import os
import ast
import re
import subprocess
import tempfile
import resource
from typing import List, Dict, Tuple, Optional
from langchain_openai import ChatOpenAI
from .config import create_chat_openai
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class CodeSecurityScanner:
    """AI代码安全扫描器 - 基于AST静态分析"""

    # 危险函数列表
    DANGEROUS_FUNCTIONS = [
        "eval", "exec", "compile",
        "os.system", "os.popen", "os.execl", "os.execle",
        "os.execlp", "os.execlpe", "os.execv", "os.execve",
        "os.execvp", "os.execvpe",
        "subprocess.call", "subprocess.Popen", "subprocess.run",
        "subprocess.check_call", "subprocess.check_output",
        "pickle.load", "pickle.loads",
        "marshal.loads",
        "__import__",
    ]

    # 危险命令模式
    DANGEROUS_PATTERNS = [
        (r"rm\s+-rf\s+/", "CRITICAL", "极端清空指令"),
        (r"rm\s+-rf\s+~", "CRITICAL", "删除用户目录"),
        (r"wget\s+http.*\|\s*(ba)?sh", "HIGH", "下载并执行远程脚本"),
        (r"curl\s+.*\|\s*(ba)?sh", "HIGH", "下载并执行远程脚本"),
        (r"chmod\s+777", "MEDIUM", "设置全权限"),
        (r"chmod\s+[0-7]*7[0-7]*7", "MEDIUM", "宽松权限设置"),
        (r"DROP\s+TABLE", "CRITICAL", "删除数据库表"),
        (r"DELETE\s+FROM\s+.*WHERE", "HIGH", "数据库删除操作"),
        (r"dd\s+if=", "HIGH", "磁盘直接读写"),
        (r"mkfs\.", "CRITICAL", "格式化文件系统"),
        (r">\s*/dev/sd", "CRITICAL", "写入磁盘设备"),
        (r"iptables\s+-F", "HIGH", "清空防火墙规则"),
        (r"useradd\s+-o\s+-u\s+0", "CRITICAL", "创建root权限用户"),
        (r"nc\s+-[lL].*-[eE]", "HIGH", "Netcat反向Shell"),
        (r"bash\s+-i\s*>&.*/dev/tcp", "CRITICAL", "反向Shell"),
    ]

    # 安全加载的模块白名单
    SAFE_MODULES = {
        "os", "sys", "json", "csv", "datetime", "math", "random",
        "collections", "itertools", "functools", "pathlib", "re",
        "typing", "dataclasses", "enum", "copy", "hashlib", "base64",
        "logging", "argparse", "configparser", "shutil", "tempfile",
        "uuid", "textwrap", "string", "numbers", "decimal", "fractions",
        "statistics", "time", "calendar", "zoneinfo", "html", "xml",
        "urllib", "http", "socket", "ssl", "email", "smtplib",
    }

    def __init__(self):
        self.issues: List[Dict] = []
        self.risk_score: int = 0

    def scan_ast(self, code: str) -> List[Dict]:
        """AST级别扫描"""
        issues = []
        try:
            tree = ast.parse(code)
            visitor = self._CodeVisitor(self)
            visitor.visit(tree)
            issues.extend(visitor.issues)
        except SyntaxError as e:
            issues.append({
                "line": e.lineno or 0,
                "severity": "ERROR",
                "type": "语法错误",
                "message": f"代码语法错误: {str(e)}",
                "code_snippet": ""
            })
        return issues

    class _CodeVisitor(ast.NodeVisitor):
        def __init__(self, scanner):
            self.scanner = scanner
            self.issues = []

        def visit_Call(self, node):
            # 检查函数调用
            fname = ""
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    fname = f"{node.func.value.id}.{node.func.attr}"
                elif isinstance(node.func.value, ast.Attribute):
                    fname = f"*.{node.func.attr}"
            elif isinstance(node.func, ast.Name):
                fname = node.func.id

            if fname in self.scanner.DANGEROUS_FUNCTIONS:
                severity = "CRITICAL" if fname in ["eval", "exec", "os.system"] else "HIGH"
                self.issues.append({
                    "line": node.lineno,
                    "severity": severity,
                    "type": "危险函数调用",
                    "message": f"调用危险函数: {fname}()",
                    "code_snippet": f"Line {node.lineno}: {fname}(...)"
                })

            self.generic_visit(node)

        def visit_Import(self, node):
            for alias in node.names:
                module_name = alias.name.split('.')[0]
                if module_name not in self.scanner.SAFE_MODULES:
                    self.issues.append({
                        "line": node.lineno,
                        "severity": "LOW",
                        "type": "未知模块导入",
                        "message": f"导入未在白名单中的模块: {module_name}",
                        "code_snippet": f"import {alias.name}"
                    })
            self.generic_visit(node)

        def visit_Try(self, node):
            # 标记有异常处理
            self.generic_visit(node)

    def scan_patterns(self, code: str) -> List[Dict]:
        """正则模式扫描"""
        issues = []
        for i, line in enumerate(code.splitlines(), 1):
            for pattern, severity, desc in self.DANGEROUS_PATTERNS:
                if re.search(pattern, line):
                    issues.append({
                        "line": i,
                        "severity": severity,
                        "type": "危险命令模式",
                        "message": f"检测到{desc}: {line.strip()[:80]}",
                        "code_snippet": line.strip()
                    })
        return issues

    def scan_missing_safety(self, code: str) -> List[Dict]:
        """检查缺失的安全措施"""
        issues = []

        # 检查是否有异常处理
        if "try" not in code and "except" not in code:
            # 但如果有subprocess/os.system等危险调用，需要异常处理
            if any(f in code for f in ["os.system", "subprocess", "eval", "exec"]):
                issues.append({
                    "line": 0,
                    "severity": "MEDIUM",
                    "type": "缺少异常处理",
                    "message": "代码包含危险操作但缺少try-except异常处理结构",
                    "code_snippet": ""
                })

        # 检查是否有输入验证
        if "input(" in code and "strip" not in code and "validate" not in code:
            issues.append({
                "line": 0,
                "severity": "MEDIUM",
                "type": "缺少输入验证",
                "message": "代码接收用户输入但未进行验证或清洗",
                "code_snippet": ""
            })

        return issues

    def full_scan(self, code: str) -> Dict:
        """完整扫描"""
        self.issues = []

        # AST扫描
        self.issues.extend(self.scan_ast(code))
        # 模式扫描
        self.issues.extend(self.scan_patterns(code))
        # 安全检查
        self.issues.extend(self.scan_missing_safety(code))

        # 计算风险分数
        severity_weights = {
            "CRITICAL": 10, "HIGH": 6, "MEDIUM": 3, "LOW": 1, "ERROR": 5
        }
        self.risk_score = sum(
            severity_weights.get(issue["severity"], 1)
            for issue in self.issues
        )

        return {
            "total_issues": len(self.issues),
            "risk_score": self.risk_score,
            "risk_level": self._get_risk_level(),
            "issues": self.issues
        }

    def _get_risk_level(self) -> str:
        if self.risk_score >= 20:
            return "CRITICAL - 代码极度危险，禁止执行"
        elif self.risk_score >= 10:
            return "HIGH - 代码存在高风险，需要人工审查"
        elif self.risk_score >= 5:
            return "MEDIUM - 代码存在中风险，建议审查"
        elif self.risk_score > 0:
            return "LOW - 代码存在低风险"
        return "SAFE - 未检测到风险"


class SandboxExecutor:
    """沙箱执行器 - 安全隔离执行AI生成的代码"""

    def __init__(self,
                 timeout: int = 5,
                 max_memory_mb: int = 128,
                 allow_network: bool = False,
                 allow_file_write: bool = False,
                 working_dir: Optional[str] = None):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.allow_network = allow_network
        self.allow_file_write = allow_file_write
        self.working_dir = working_dir or tempfile.mkdtemp(prefix="sandbox_")

    def _build_sandbox_wrapper(self, code: str) -> str:
        """构建沙箱包装代码"""
        wrapper = f'''
import sys
import os
import builtins

# 限制模块导入
_original_import = builtins.__import__

BLOCKED_MODULES = {{
    "os", "subprocess", "shutil", "socket", "requests",
    "http", "urllib", "ftplib", "smtplib", "telnetlib",
    "ctypes", "multiprocessing", "signal",
}}

def _safe_import(name, *args, **kwargs):
    module_name = name.split(".")[0]
    if module_name in BLOCKED_MODULES and not {self.allow_network}:
        raise ImportError(f"模块 '{{module_name}}' 在沙箱中被禁止")
    return _original_import(name, *args, **kwargs)

builtins.__import__ = _safe_import

# 限制文件操作
if not {self.allow_file_write}:
    _original_open = builtins.open
    def _safe_open(file, mode="r", *args, **kwargs):
        if "w" in mode or "a" in mode or "+" in mode:
            raise PermissionError("沙箱禁止写入文件")
        return _original_open(file, mode, *args, **kwargs)
    builtins.open = _safe_open

# 设置工作目录
os.chdir("{self.working_dir}")

# 用户代码
{code}
'''
        return wrapper

    def execute(self, code: str) -> Tuple[str, str, bool]:
        """在沙箱中执行代码，返回(stdout, stderr, success)"""
        wrapper_code = self._build_sandbox_wrapper(code)

        try:
            result = subprocess.run(
                ["python3", "-c", wrapper_code],
                capture_output=True,
                timeout=self.timeout,
                text=True,
                cwd=self.working_dir,
                env={
                    "PATH": os.environ.get("PATH", "/usr/bin"),
                    "HOME": self.working_dir,
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )

            stdout = result.stdout
            stderr = result.stderr
            success = result.returncode == 0

            # 截断过长输出
            if len(stdout) > 5000:
                stdout = stdout[:5000] + "\n...[输出已截断]"
            if len(stderr) > 5000:
                stderr = stderr[:5000] + "\n...[错误输出已截断]"

            return stdout, stderr, success

        except subprocess.TimeoutExpired:
            return "", f"执行超时 (>{self.timeout}秒)", False
        except Exception as e:
            return "", f"沙箱执行异常: {str(e)}", False


class SafeCodeRunner:
    """安全代码运行器 - 扫描 + 沙箱执行"""

    def __init__(self):
        self.scanner = CodeSecurityScanner()
        self.executor = SandboxExecutor(
            timeout=5,
            max_memory_mb=128,
            allow_network=False,
            allow_file_write=False
        )

    def run(self, code: str, force: bool = False) -> Dict:
        """安全运行代码"""
        result = {
            "code": code,
            "scan": None,
            "execution": None,
            "allowed": False
        }

        # 第一步：安全扫描
        scan_result = self.scanner.full_scan(code)
        result["scan"] = scan_result

        if scan_result["risk_level"].startswith("CRITICAL"):
            result["allowed"] = False
            result["execution"] = {
                "stdout": "",
                "stderr": f"代码安全扫描未通过: {scan_result['risk_level']}",
                "success": False
            }
            return result

        if scan_result["risk_level"].startswith("HIGH") and not force:
            result["allowed"] = False
            result["execution"] = {
                "stdout": "",
                "stderr": f"代码风险较高，需要人工确认: {scan_result['risk_level']}",
                "success": False
            }
            return result

        # 第二步：沙箱执行
        result["allowed"] = True
        stdout, stderr, success = self.executor.execute(code)
        result["execution"] = {
            "stdout": stdout,
            "stderr": stderr,
            "success": success
        }

        return result


__all__ = [
    "CodeSecurityScanner",
    "SandboxExecutor",
    "SafeCodeRunner",
]
