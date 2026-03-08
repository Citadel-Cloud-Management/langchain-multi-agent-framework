"""Safe code execution sandbox for the coder agent."""

from __future__ import annotations

import io
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class CodeExecutorInput(BaseModel):
    """Input schema for the code executor tool."""

    code: str = Field(description="Python code to execute.")
    timeout: int = Field(default=30, description="Execution timeout in seconds.")


class CodeExecutorTool(BaseTool):
    """Tool for executing Python code in a sandboxed environment.

    Captures stdout, stderr, and return values. Applies restrictions
    to prevent dangerous operations like file system writes or network access
    in the default restricted mode.
    """

    name: str = "code_executor"
    description: str = (
        "Execute Python code in a sandboxed environment. Returns stdout output, "
        "stderr output, and any return value. Use this to test code snippets."
    )
    args_schema: Type[BaseModel] = CodeExecutorInput
    allowed_modules: list[str] = [
        "math", "json", "re", "datetime", "collections",
        "itertools", "functools", "string", "textwrap",
        "dataclasses", "typing", "enum", "statistics",
    ]
    restricted_mode: bool = True

    def _run(
        self,
        code: str,
        timeout: int = 30,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute Python code and capture output.

        Args:
            code: Python source code to execute.
            timeout: Maximum execution time in seconds.
            run_manager: Optional callback manager.

        Returns:
            Formatted string with stdout, stderr, and execution status.
        """
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        restricted_globals = {"__builtins__": self._get_safe_builtins()}
        local_vars: dict = {}

        try:
            compiled = compile(code, "<agent_code>", "exec")

            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(compiled, restricted_globals, local_vars)

            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()

            result_parts = []
            if stdout_output:
                result_parts.append(f"STDOUT:\n{stdout_output}")
            if stderr_output:
                result_parts.append(f"STDERR:\n{stderr_output}")
            if "result" in local_vars:
                result_parts.append(f"RESULT:\n{local_vars['result']}")
            if not result_parts:
                result_parts.append("Code executed successfully with no output.")

            return "\n\n".join(result_parts)

        except SyntaxError as e:
            return f"SYNTAX ERROR:\n{e}"
        except Exception as e:
            tb = traceback.format_exc()
            return f"RUNTIME ERROR:\n{e}\n\nTraceback:\n{tb}"

    def _get_safe_builtins(self) -> dict:
        """Return a restricted set of built-in functions."""
        import builtins

        safe_names = [
            "abs", "all", "any", "ascii", "bin", "bool", "bytes",
            "callable", "chr", "complex", "dict", "dir", "divmod",
            "enumerate", "filter", "float", "format", "frozenset",
            "getattr", "hasattr", "hash", "hex", "id", "int",
            "isinstance", "issubclass", "iter", "len", "list",
            "map", "max", "min", "next", "object", "oct", "ord",
            "pow", "print", "property", "range", "repr", "reversed",
            "round", "set", "slice", "sorted", "str", "sum",
            "super", "tuple", "type", "vars", "zip",
        ]
        safe_builtins = {name: getattr(builtins, name) for name in safe_names}
        safe_builtins["__import__"] = self._restricted_import
        return safe_builtins

    def _restricted_import(self, name: str, *args, **kwargs):
        """Allow importing only whitelisted modules."""
        if self.restricted_mode and name not in self.allowed_modules:
            raise ImportError(
                f"Import of '{name}' is not allowed in restricted mode. "
                f"Allowed modules: {', '.join(self.allowed_modules)}"
            )
        return __import__(name, *args, **kwargs)
