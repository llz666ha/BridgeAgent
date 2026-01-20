#!/usr/bin/env python3
"""
BNERC Server - 基于 LangGraph 的 Plan & Execute Agent 生产服务器
符合 ASGI 规范
"""

import json
import os
import sys
import re
import logging
import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 加载环境变量
load_dotenv()

# ================================
# 日志配置
# ================================
def setup_logging():
    """设置生产环境日志配置"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 移除默认处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（带轮转）- 使用绝对路径
    log_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(log_dir, "bnerc_server.log")
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 降低第三方库日志级别
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    return logger

logger = setup_logging()

# ================================
# LangGraph 相关导入
# ================================
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.types import StreamWriter
# PostgreSQL checkpointer 支持
POSTGRES_AVAILABLE = False
PostgresSaver = None
AsyncPostgresSaver = None
AsyncConnectionPool = None

try:
    # 尝试导入 PostgreSQL checkpointer（异步版本，推荐）
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    # 尝试导入连接池
    try:
        from psycopg_pool import AsyncConnectionPool
        logger.info("psycopg_pool 可用，将使用连接池")
    except ImportError:
        logger.warning("psycopg_pool 不可用，将使用直接连接方式")
    POSTGRES_AVAILABLE = True
    logger.info("PostgreSQL checkpointer 可用（异步版本）")
except ImportError:
    try:
        # 尝试同步版本
        from langgraph.checkpoint.postgres import PostgresSaver
        POSTGRES_AVAILABLE = True
        logger.info("PostgreSQL checkpointer 可用（同步版本）")
    except ImportError:
        POSTGRES_AVAILABLE = False
        logger.warning("PostgresSaver 不可用，将使用内存存储（重启后状态会丢失）")

# 始终导入 MemorySaver 作为回退
from langgraph.checkpoint.memory import MemorySaver

from utils.file import load_prompt_from_yaml, load_prompt_from_md
from utils.draw import draw_workflow
from utils.utils import get_llm_instance
from agent.mcp.mcp_client import client

# ================================
# 全局变量
# ================================
llm = get_llm_instance()
llm_with_tools = None
app_graph = None
checkpointer = None
db_pool = None  # PostgreSQL 连接池

# 加载提示词模板（使用 md 版本）
planner_prompt_data = load_prompt_from_md('planner2.md')
checker_prompt_data = load_prompt_from_md('checker.md')
summarizer_prompt_data = load_prompt_from_md('summarizer.md')

# ================================
# State 定义
# ================================
class State(MessagesState):
    run_id: str
    plan: List[Dict[str, Any]]
    current_step: int
    current_tool: Optional[Dict[str, Any]]
    files: List[Dict[str, Any]]
    result: Dict[str, Any]
    replan: bool
    # reply 字段已移除，最终回复可以从 messages 的最后一个 AIMessage 中获取

# ================================
# 工具函数
# ================================
def extract_text(content):
    """从消息内容中提取文本（支持多种格式）"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # 处理列表格式，如 [{"type": "text", "text": "..."}]
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    text_parts.append(item["text"])
                elif "content" in item:
                    text_parts.append(item["content"])
            elif isinstance(item, str):
                text_parts.append(item)
        return "\n".join(text_parts)
    elif isinstance(content, dict):
        return content.get("text", content.get("content", str(content)))
    else:
        return str(content)

async def safe_get_tools() -> list:
    """安全获取 MCP 工具列表"""
    try:
        return await client.get_tools()
    except Exception as e:
        logger.warning(f"MCP 连接失败，使用空工具列表: {e}")
        return []

async def initialize_llm_with_tools():
    """初始化绑定工具的 LLM"""
    global llm_with_tools
    if llm_with_tools is None:
        mcp_tools = await safe_get_tools()
        llm_with_tools = llm.bind_tools(mcp_tools)
        logger.info(f"LLM 已绑定 {len(mcp_tools)} 个工具")
    return llm_with_tools

def extract_text(content):
    """提取文本内容"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        return "\n".join(texts)
    return str(content)

def extract_tool_results(messages):
    """提取工具执行结果"""
    results = []
    for m in messages:
        if isinstance(m, ToolMessage):
            results.append({
                "tool_call_id": m.tool_call_id,
                "content": m.content,
            })
    return results

def get_recent_run_tool_messages(messages, run_id):
    """获取当前 run 的工具消息"""
    results = []
    i = len(messages) - 1
    
    while i >= 0:
        m = messages[i]
        if isinstance(m, ToolMessage):
            if i >= 1:
                prev_msg = messages[i-1]
                if isinstance(prev_msg, AIMessage) and prev_msg.additional_kwargs.get("run_id") == run_id:
                    results.append(m)
                else:
                    break
            else:
                break
            i -= 1
        elif isinstance(m, AIMessage) and m.additional_kwargs.get("run_id") == run_id:
            i -= 1
        else:
            break
    
    return list(reversed(results))

# ================================
# LangGraph 节点定义
# ================================
async def planner(state: State, writer: StreamWriter):
    """生成执行计划"""
    messages = state.get("messages", [])
    run_id = state.get("run_id", str(uuid.uuid4()))
    
    # 获取用户输入（当前用户输入）
    user_input = ""
    files = state.get("files", [])
    current_user_msg_index = -1
    
    # 从后往前找最后一个用户消息
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, HumanMessage):
            user_input = extract_text(msg.content)
            current_user_msg_index = i
            break
    
    # 构建对话历史：排除当前用户输入及其之后的所有消息
    conversation_messages = []
    if current_user_msg_index > 0:
        conversation_messages = messages[:current_user_msg_index]
    else:
        conversation_messages = []
    
    conversation = "\n".join(
        f"{m.type}: {extract_text(m.content)}" for m in conversation_messages
        if m.type in ['human', 'ai'] and hasattr(m, 'content') and extract_text(m.content).strip()
    )
    
    logger.info(f"Planner: 用户输入: {user_input}")
    
    # 发送计划开始事件
    writer({"event": "plan_start", "text": "🎯 正在生成计划...\n"})
    
    # 确保 LLM 已绑定工具
    global llm_with_tools
    if llm_with_tools is None:
        await initialize_llm_with_tools()
    
    # 格式化 files 为字符串
    files_str = ""
    if files:
        files_str = "\n".join(
            f"- {f.get('name', '')} ({f.get('path', '')}, {f.get('type', '')})" 
            for f in files if isinstance(f, dict)
        )
    
    # md 文件返回的是字符串，直接格式化
    system_prompt = planner_prompt_data.format(
        user_input=user_input,
        conversation=conversation,
        files=files_str if files_str else "无"
    )
    
    try:
        response = await llm_with_tools.ainvoke([SystemMessage(content=system_prompt)])
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        # 记录原始响应（用于调试）
        logger.debug(f"Planner: LLM 原始响应: {response_content[:500]}...")  # 只记录前500字符
        
        # 检查响应是否为空
        if not response_content or not response_content.strip():
            logger.error("Planner: LLM 返回空响应")
            return {
                "plan": [],
                "current_step": 0,
                "messages": messages,
                "run_id": run_id
            }
        
        # 尝试解析 JSON
        parsed_response = None
        try:
            # 首先尝试直接解析
            parsed_response = json.loads(response_content)
        except json.JSONDecodeError:
            # 如果直接解析失败，尝试提取 JSON 块
            logger.warning("Planner: 直接 JSON 解析失败，尝试提取 JSON 块")
            
            # 尝试提取 ```json ... ``` 代码块
            json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
            json_match = re.search(json_block_pattern, response_content, re.DOTALL)
            if json_match:
                try:
                    parsed_response = json.loads(json_match.group(1))
                    logger.info("Planner: 从代码块中成功提取 JSON")
                except json.JSONDecodeError:
                    pass
            
            # 如果还是失败，尝试提取第一个 { ... } 块
            if parsed_response is None:
                json_obj_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                json_match = re.search(json_obj_pattern, response_content, re.DOTALL)
                if json_match:
                    try:
                        parsed_response = json.loads(json_match.group(0))
                        logger.info("Planner: 从文本中成功提取 JSON 对象")
                    except json.JSONDecodeError:
                        pass
            
            # 如果仍然失败，记录错误并返回空计划
            if parsed_response is None:
                logger.error(f"Planner: 无法解析 JSON，响应内容: {response_content[:200]}")
                raise json.JSONDecodeError("无法从响应中提取有效的 JSON", response_content, 0)
        
        if not isinstance(parsed_response, dict) or "plan" not in parsed_response:
            logger.warning(f"Planner: LLM 返回格式错误，缺少 plan 字段。解析结果: {parsed_response}")
            return {
                "plan": [],
                "current_step": 0,
                "messages": messages,  # 不添加错误消息到对话历史
                "run_id": run_id
            }
        
        # 处理执行计划
        processed_plan = []
        for step in parsed_response["plan"]:
            tool_name = step.get("tool", "") if isinstance(step, dict) else str(step).strip()
            if tool_name:
                processed_plan.append({"tool": tool_name})
        
        logger.info(f"Planner: 执行计划: {processed_plan}")
        
        # 发送计划完成事件
        writer({"event": "plan_final", "plan": processed_plan})
        writer({"event": "plan_end", "text": "\n"})
        
        # 绘制工作流程图
        try:
            draw_workflow(parsed_response["plan"])
        except Exception:
            pass
        
        return {
            "plan": processed_plan,
            "current_step": 0,
            "messages": messages,
            "files": files,
            "run_id": run_id
        }
    except json.JSONDecodeError as e:
        logger.error(f"Planner: JSON 解析失败: {e}", exc_info=True)
        # 不添加错误消息到对话历史，只记录日志
        return {
            "plan": [],
            "current_step": 0,
            "messages": messages,  # 保持 messages 不变，不污染对话历史
            "run_id": run_id
        }
    except Exception as e:
        logger.error(f"Planner: 计划生成过程中发生错误: {e}", exc_info=True)
        # 不添加错误消息到对话历史，只记录日志
        return {
            "plan": [],
            "current_step": 0,
            "messages": messages,  # 保持 messages 不变，不污染对话历史
            "run_id": run_id
        }

async def executor(state: State):
    """执行计划步骤"""
    global llm_with_tools
    
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    messages = state.get("messages", [])
    run_id = state.get("run_id", str(uuid.uuid4()))
    
    # 空计划处理
    if not plan:
        return {
            "next": "summarizer",
            "plan": plan,
            "messages": messages,
        }
    
    # 解析上一个工具执行结果（如果有）
    is_retry = False
    retry_count = state.get("retry_count", 0)
    
    if messages and isinstance(messages[-1], ToolMessage):
        try:
            # 处理不同的 content 格式
            content = messages[-1].content
            if isinstance(content, list) and len(content) > 0:
                raw_text = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
            else:
                raw_text = str(content)
            
            result_json = json.loads(raw_text)
            success = result_json.get("status") == "success"
            # tool_failed = not success
            
            # 如果工具执行失败且可以重试
            if not success and retry_count < 1:
                logger.info(f"Executor: 工具执行失败，准备重试 ({retry_count + 1}/1)")
                is_retry = True
                
                # 移除失败的 ToolMessage 和对应的 AIMessage（包含 tool_call）
                cleaned_messages = messages[:-1]  # 移除 ToolMessage
                if cleaned_messages and isinstance(cleaned_messages[-1], AIMessage) and cleaned_messages[-1].tool_calls:
                    cleaned_messages = cleaned_messages[:-1]  # 移除包含 tool_call 的 AIMessage
                    logger.info(f"Executor: 已移除失败的 ToolMessage 和 AIMessage，准备重新生成 tool_call")
                
                # 更新 messages，保持 current_step 不变，继续执行后面的代码重新生成 tool_call
                messages = cleaned_messages
                
            # 如果重试次数已用完仍然失败，跳过当前步骤
            elif not success and retry_count >= 1:
                logger.warning(f"Executor: 工具执行失败且重试次数已用完，跳过当前步骤")
                return {
                    "messages": messages,
                    "current_step": current_step + 1,  # 跳过当前步骤
                    "plan": plan,
                    "retry_count": 0,  # 重置重试计数
                    "next": "check" if current_step + 1 >= len(plan) else "tools",
                }
                
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            logger.error(f"Executor: 解析工具结果失败: {e}, content: {messages[-1].content}")
            # 解析失败也视为工具执行失败
            if retry_count < 1:
                logger.info(f"Executor: 解析失败，准备重试 ({retry_count + 1}/1)")
                is_retry = True
                cleaned_messages = messages[:-1]
                if cleaned_messages and isinstance(cleaned_messages[-1], AIMessage) and cleaned_messages[-1].tool_calls:
                    cleaned_messages = cleaned_messages[:-1]
                messages = cleaned_messages
            else:
                logger.warning(f"Executor: 解析失败且重试次数已用完，跳过当前步骤")
                return {
                    "messages": messages,
                    "current_step": current_step + 1,
                    "plan": plan,
                    "retry_count": 0,
                    "next": "check" if current_step + 1 >= len(plan) else "tools",
                }
    
    # 计划执行完成
    if current_step >= len(plan):
        return {
            "current_step": current_step,
            "next": "check",
            "messages": messages,
        }
    
    # 当前步骤
    step = plan[current_step]
    tool_name = step["tool"]
    logger.info(f"Executor: Step {current_step + 1}/{len(plan)} → {tool_name}")
    
    # 构建文件上下文
    files = state.get("files", [])
    logger.info(f"Executor: 当前文件列表: {files}")
    file_context = ""
    if files:
        # 更明确地提供文件信息，特别是路径
        file_list = []
        for f in files:
            file_name = f.get('name', '未知文件')
            file_path = f.get('path', '')
            file_type = f.get('type', '')
            
            # 记录原始路径
            logger.info(f"Executor: 处理文件 - 名称: {file_name}, 原始路径: {file_path}, 类型: {file_type}")
            
            # 确保路径是绝对路径
            if file_path:
                if not os.path.isabs(file_path):
                    # 如果是相对路径，尝试转换为绝对路径
                    file_path = os.path.abspath(file_path)
                    logger.info(f"Executor: 转换为绝对路径: {file_path}")
                
                # 验证文件是否存在
                if os.path.exists(file_path):
                    logger.info(f"Executor: 文件存在: {file_path}")
                else:
                    logger.warning(f"Executor: 文件不存在: {file_path}")
            else:
                logger.warning(f"Executor: 文件路径为空: {file_name}")
            
            file_list.append(f"文件名: {file_name}, 路径: {file_path}, 类型: {file_type}")
        
        file_context = "当前可用文件信息：\n" + "\n".join(file_list)
        file_context += "\n\n重要：如果工具需要文件路径参数（如 image_path），请使用上面提供的完整路径。"
        logger.info(f"Executor: 文件上下文: {file_context}")
    else:
        logger.info("Executor: 没有可用文件")
    
    # 让 LLM 生成 tool_call
    llm_messages = messages + [
        SystemMessage(content=file_context) if file_context else SystemMessage(content=""),
        HumanMessage(content=f"请调用工具 `{tool_name}`，自动从上下文中提取所需参数。如果工具需要文件路径，请使用上面提供的完整文件路径。不要输出任何解释文本。")
    ]
    
    if llm_with_tools is None:
        await initialize_llm_with_tools()
    
    response = await llm_with_tools.ainvoke(
        llm_messages,
        tool_choice={"type": "function", "function": {"name": tool_name}},
    )
    
    if not response.tool_calls:
        raise RuntimeError(f"Executor: LLM 未生成 tool_call: {tool_name}")
    
    response.additional_kwargs["run_id"] = run_id
    
    # 如果是重试，更新 retry_count 并保持 current_step 不变；否则重置为 0 并前进步骤
    if is_retry:
        new_retry_count = retry_count + 1
        new_current_step = current_step  # 重试时保持步骤不变，重新执行当前步骤
    else:
        new_retry_count = 0
        new_current_step = current_step + 1  # 正常执行时前进步骤
    
    return {
        "messages": messages + [response],
        "current_step": new_current_step,
        "plan": plan,
        "run_id": run_id,
        "next": "tools",
        "retry_count": new_retry_count
    }

async def checker(state: State):
    """检查执行结果"""
    messages = state["messages"]
    run_id = state.get("run_id", "")
    
    # 获取用户查询（当前用户输入）
    user_query = None
    current_user_msg_index = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, HumanMessage):
            user_query = extract_text(msg.content) if hasattr(msg, 'content') else ""
            current_user_msg_index = i
            break
    
    # 提取工具执行结果
    recent_tool_messages = get_recent_run_tool_messages(messages, run_id)
    tool_results = extract_tool_results(recent_tool_messages)
    final_result = json.dumps(tool_results, ensure_ascii=False, indent=2) if tool_results else ""
    
    # 构建对话历史：排除当前用户输入及其之后的所有消息
    conversation_messages = []
    if current_user_msg_index > 0:
        conversation_messages = messages[:current_user_msg_index]
    else:
        conversation_messages = []
    
    conversation = "\n".join(
        f"{m.type}: {extract_text(m.content)}" for m in conversation_messages
        if hasattr(m, 'type') and m.type in ['human', 'ai'] and hasattr(m, 'content') and extract_text(m.content).strip()
    )
    
    logger.info(f"Checker: 用户问题: {user_query}")
    logger.info(f"Checker: 执行结果: {final_result[:200]}...")
    
    # md 文件返回的是字符串，直接格式化
    system_prompt = checker_prompt_data.format(
        user_query=user_query,
        final_result=final_result,
        conversation=conversation
    )
    evaluation_prompt = [
        SystemMessage(content=system_prompt)
    ]
    
    try:
        response = await llm.ainvoke(evaluation_prompt)
        response_content = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        
        # 检查响应内容是否为空
        if not response_content:
            logger.warning("Checker: LLM 返回空内容，使用默认评估结果")
            evaluation_result = {
                "satisfies_needs": True,
                "reason": "LLM 返回空内容，默认认为满足需求",
                "needs_replan": False
            }
        else:
            # 尝试解析 JSON
            try:
                evaluation_result = json.loads(response_content)
            except json.JSONDecodeError as json_err:
                logger.error(f"Checker: JSON 解析失败，响应内容: {response_content[:200]}")
                logger.error(f"Checker: JSON 解析错误详情: {json_err}")
                
                # 尝试提取 JSON 部分（可能包含其他文本）
                import re
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    try:
                        evaluation_result = json.loads(json_match.group())
                        logger.info("Checker: 从响应中提取并解析 JSON 成功")
                    except json.JSONDecodeError:
                        logger.error("Checker: 无法从响应中提取有效 JSON，使用默认结果")
                        evaluation_result = {
                            "satisfies_needs": True,
                            "reason": "无法解析 LLM 响应，默认认为满足需求",
                            "needs_replan": False
                        }
                else:
                    logger.error("Checker: 响应中未找到 JSON 格式，使用默认结果")
                    evaluation_result = {
                        "satisfies_needs": True,
                        "reason": "响应格式错误，默认认为满足需求",
                        "needs_replan": False
                    }
        
        # 确保必要的字段存在
        if "satisfies_needs" not in evaluation_result:
            evaluation_result["satisfies_needs"] = True
        if "needs_replan" not in evaluation_result:
            evaluation_result["needs_replan"] = False
        
        logger.info(f"Checker: 满足需求: {evaluation_result['satisfies_needs']}, 需要重规划: {evaluation_result.get('needs_replan', False)}")
        
        # 暂不重规划
        evaluation_result["needs_replan"] = False
        
        return {
            "replan": evaluation_result["needs_replan"],
            "next": "replan" if evaluation_result["needs_replan"] else "end",
            "plan": state.get("plan", []),
            "messages": messages
        }
    except Exception as e:
        logger.error(f"Checker 错误: {e}", exc_info=True)
        return {
            "replan": False,
            "next": "end",
            "plan": state.get("plan", []),
            "messages": messages
        }

async def summarizer(state: State, writer: StreamWriter):
    """生成最终回复 - 流式输出"""
    messages = state["messages"]
    run_id = state.get("run_id", "")
    
    # 获取用户查询（当前用户输入）
    user_query = ""
    current_user_msg_index = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if hasattr(msg, 'type') and msg.type == "human":
            user_query = extract_text(msg.content) if hasattr(msg, 'content') else ""
            current_user_msg_index = i
            break
    
    # 提取工具执行结果
    recent_tool_messages = get_recent_run_tool_messages(messages, run_id)
    tool_results = extract_tool_results(recent_tool_messages)
    final_result = json.dumps(tool_results, ensure_ascii=False, indent=2) if tool_results else ""
    
    # 构建对话历史：排除当前用户输入及其之后的所有消息（包括工具调用、AI回复等）
    # 只保留当前用户输入之前的对话历史
    conversation_messages = []
    if current_user_msg_index > 0:
        # 只取当前用户输入之前的消息
        conversation_messages = messages[:current_user_msg_index]
    else:
        # 如果没有找到当前用户输入，或者当前用户输入是第一条消息，则没有历史对话
        conversation_messages = []
    
    # 格式化对话历史，只包含 human 和 ai 类型的消息
    conversation = "\n".join(
        f"{m.type}: {extract_text(m.content)}" for m in conversation_messages
        if hasattr(m, 'type') and m.type in ['human', 'ai'] and hasattr(m, 'content') and extract_text(m.content).strip()
    )
    
    logger.info(f"Summarizer: 当前用户输入: {user_query}")
    logger.info(f"Summarizer: 对话历史长度: {len(conversation_messages)} 条消息")
    logger.info(f"Summarizer: 对话历史: {conversation[:200]}..." if conversation else "Summarizer: 无对话历史")
    
    logger.info("Summarizer: 生成最终回复")
    
    # md 文件返回的是字符串，直接格式化
    system_prompt = summarizer_prompt_data.format(
        user_query=user_query,
        final_result=final_result,
        conversation=conversation
    )
    summary_prompt = [
        SystemMessage(content=system_prompt)
    ]
    
    accumulated_reply = ""
    final_messages = messages.copy()
    
    try:
        if hasattr(llm, "astream"):
            async for chunk in llm.astream(summary_prompt):
                if hasattr(chunk, 'content') and chunk.content:
                    accumulated_reply += chunk.content
                    # 使用 StreamWriter 发送流式数据（同步调用）
                    writer({
                        "event_type": "custom_stream",
                        "reply": chunk.content,
                        "is_partial": True
                    })
        else:
            response = await llm.ainvoke(summary_prompt)
            accumulated_reply = response.content if hasattr(response, 'content') else str(response)
            writer({
                "event_type": "custom_stream",
                "reply": accumulated_reply,
                "is_partial": False
            })
    except Exception as e:
        logger.error(f"Summarizer 错误: {e}")
        accumulated_reply = final_result or "抱歉，生成回复时出错。"
        writer({
            "event_type": "custom_stream",
            "reply": accumulated_reply,
            "is_partial": False
        })
    
    final_messages.append(AIMessage(content=accumulated_reply))
    
    # reply 字段已移除，最终回复已包含在 messages 的最后一个 AIMessage 中
    return {
        "messages": final_messages,
        "plan": state.get("plan", [])
    }

# ================================
# 图初始化
# ================================
async def init_checkpointer():
    """初始化 checkpointer（支持 PostgreSQL 或内存）"""
    global checkpointer, db_pool
    
    if checkpointer is not None:
        return checkpointer
    
    # 从环境变量读取数据库配置
    db_conn_string = os.getenv("DB_CONN_STRING", "")
    use_postgres = os.getenv("USE_POSTGRES", "false").lower() == "true"
    
    if use_postgres and POSTGRES_AVAILABLE and db_conn_string:
        try:
            logger.info("使用 PostgreSQL 作为 checkpointer")
            
            # 如果支持连接池，使用连接池方式（推荐）
            if AsyncPostgresSaver is not None and AsyncConnectionPool is not None:
                # 使用连接池方式
                logger.info("使用连接池方式初始化 PostgreSQL checkpointer")
                
                # 从环境变量读取连接池配置
                pool_max_size = int(os.getenv("DB_POOL_MAX_SIZE", "20"))
                pool_min_size = int(os.getenv("DB_POOL_MIN_SIZE", "5"))
                pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
                
                # 处理连接字符串格式（psycopg 需要 postgresql:// 格式，不支持 postgresql+asyncpg://）
                # 如果连接字符串包含 postgresql+asyncpg://，需要转换为 postgresql://
                conninfo = db_conn_string.replace("postgresql+asyncpg://", "postgresql://")
                
                # 先使用临时连接执行 setup（因为 CREATE INDEX CONCURRENTLY 不能在事务中执行）
                # 使用单独的连接在 autocommit 模式下执行 setup
                logger.info("使用临时连接初始化数据库表...")
                
                # 检查是否需要强制重新创建表
                force_recreate = os.getenv("DB_FORCE_RECREATE", "false").lower() == "true"
                if force_recreate:
                    logger.warning("⚠️  DB_FORCE_RECREATE=true，将删除并重新创建所有表（会丢失数据）")
                    try:
                        from psycopg import AsyncConnection
                        async with await AsyncConnection.connect(conninfo, autocommit=True) as temp_conn:
                            async with temp_conn.cursor() as cur:
                                # 删除所有相关表
                                await cur.execute("DROP TABLE IF EXISTS checkpoint_writes CASCADE")
                                await cur.execute("DROP TABLE IF EXISTS checkpoint_blobs CASCADE")
                                await cur.execute("DROP TABLE IF EXISTS checkpoints CASCADE")
                                await cur.execute("DROP TABLE IF EXISTS checkpoint_migrations CASCADE")
                                logger.info("已删除旧表")
                    except Exception as drop_error:
                        logger.warning(f"删除旧表时出错（可能表不存在）: {drop_error}")
                
                setup_success = False
                try:
                    from psycopg import AsyncConnection
                    # 创建临时连接，使用 autocommit 模式
                    # 在 psycopg 3.x 中，autocommit 通过连接参数设置
                    async with await AsyncConnection.connect(conninfo, autocommit=True) as temp_conn:
                        # 创建临时 checkpointer 来执行 setup
                        temp_checkpointer = AsyncPostgresSaver(temp_conn)
                        await temp_checkpointer.setup()
                        setup_success = True
                        logger.info("数据库表初始化完成（使用 autocommit 模式）")
                except Exception as setup_error:
                    # 如果 setup 失败，尝试使用 from_conn_string 方式（可能表已经存在）
                    logger.warning(f"使用 autocommit 模式 setup 失败: {setup_error}，尝试其他方式...")
                    try:
                        # 尝试使用 from_conn_string 方式（如果表已存在，不会报错）
                        temp_checkpointer = await AsyncPostgresSaver.from_conn_string(conninfo).asetup()
                        setup_success = True
                        logger.info("数据库表初始化完成（使用 from_conn_string 方式）")
                    except Exception as e2:
                        # 如果还是失败，记录错误
                        logger.error(f"数据库表初始化失败: {e2}")
                        # 验证表是否存在，如果存在但结构不完整，可能需要手动修复
                        try:
                            from psycopg import AsyncConnection
                            async with await AsyncConnection.connect(conninfo) as verify_conn:
                                async with verify_conn.cursor() as cur:
                                    await cur.execute("""
                                        SELECT table_name 
                                        FROM information_schema.tables 
                                        WHERE table_schema = 'public' 
                                        AND table_name = 'checkpoints'
                                    """)
                                    table_exists = await cur.fetchone()
                                    if table_exists:
                                        logger.warning("checkpoints 表已存在，但可能结构不完整。建议运行初始化脚本修复。")
                                        logger.warning("可以运行: python3 scripts/init_postgres_tables.py")
                        except Exception:
                            pass
                        # 不抛出异常，让代码继续，但会回退到内存存储
                        raise Exception(f"数据库表初始化失败，无法使用 PostgreSQL checkpointer: {e2}")
                
                if not setup_success:
                    raise Exception("数据库表初始化失败")
                
                # 创建连接池
                db_pool = AsyncConnectionPool(
                    conninfo=conninfo,
                    max_size=pool_max_size,
                    min_size=pool_min_size,
                    timeout=pool_timeout,
                    # 可选：添加连接参数
                    kwargs={
                        # 可以在这里添加额外的连接参数
                    }
                )
                
                # 打开连接池（确保连接池在使用前已准备好）
                await db_pool.open()
                
                # 使用连接池创建 checkpointer（表已经初始化，不需要再次 setup）
                checkpointer = AsyncPostgresSaver(db_pool)
                
                # 验证表结构是否正确（检查主键约束是否存在）
                try:
                    from psycopg import AsyncConnection
                    async with await AsyncConnection.connect(conninfo) as verify_conn:
                        async with verify_conn.cursor() as cur:
                            # 检查 checkpoints 表的主键约束
                            await cur.execute("""
                                SELECT constraint_name, constraint_type
                                FROM information_schema.table_constraints
                                WHERE table_schema = 'public' 
                                AND table_name = 'checkpoints'
                                AND constraint_type = 'PRIMARY KEY'
                            """)
                            pk_exists = await cur.fetchone()
                            if not pk_exists:
                                logger.warning("⚠️  checkpoints 表缺少主键约束，可能导致 ON CONFLICT 错误")
                                logger.warning("建议运行初始化脚本修复: python3 scripts/init_postgres_tables.py")
                except Exception as verify_error:
                    logger.warning(f"验证表结构时出错: {verify_error}")
                
                logger.info(f"PostgreSQL checkpointer 初始化成功（连接池大小: {pool_min_size}-{pool_max_size}）")
                
            elif AsyncPostgresSaver is not None:
                # 回退到直接连接字符串方式
                logger.info("使用直接连接字符串方式初始化 PostgreSQL checkpointer")
                checkpointer = await AsyncPostgresSaver.from_conn_string(db_conn_string).asetup()
                logger.info("PostgreSQL checkpointer 初始化成功")
                
            elif PostgresSaver is not None:
                # 使用同步版本
                checkpointer = PostgresSaver.from_conn_string(db_conn_string)
                # 同步版本可能需要手动 setup
                if hasattr(checkpointer, 'asetup'):
                    await checkpointer.asetup()
                elif hasattr(checkpointer, 'setup'):
                    # 如果是同步方法，需要在同步上下文中调用
                    import asyncio
                    await asyncio.to_thread(checkpointer.setup)
                logger.info("PostgreSQL checkpointer 初始化成功（同步版本）")
            else:
                raise ImportError("PostgresSaver 类不可用")
                
        except Exception as e:
            logger.error(f"PostgreSQL checkpointer 初始化失败: {e}，回退到内存存储", exc_info=True)
            # 如果连接池已创建，需要清理
            if db_pool is not None:
                try:
                    await db_pool.close()
                except Exception:
                    pass
                db_pool = None
            
            # 如果是表结构问题，提供更详细的错误信息
            error_str = str(e).lower()
            if "constraint" in error_str or "conflict" in error_str or "index" in error_str:
                logger.error("=" * 60)
                logger.error("数据库表结构可能不完整！")
                logger.error("建议执行以下操作修复：")
                logger.error("1. 运行初始化脚本: python3 scripts/init_postgres_tables.py")
                logger.error("2. 或者删除旧表后重新启动服务（会丢失数据）")
                logger.error("=" * 60)
            
            checkpointer = MemorySaver()
    else:
        if use_postgres:
            logger.warning("PostgreSQL 配置存在但 PostgresSaver 不可用，使用内存存储")
        else:
            logger.info("使用内存存储（MemorySaver），重启后状态会丢失")
        checkpointer = MemorySaver()
    
    return checkpointer

async def init_graph():
    """延迟初始化 LangGraph 图"""
    global app_graph
    
    # 初始化 checkpointer
    await init_checkpointer()
    
    # 初始化工具
    tools = await safe_get_tools()
    logger.info(f"初始化工具列表: {len(tools)} 个工具")
    tool_node = ToolNode(tools)
    
    # 构建图
    graph = StateGraph(State)
    
    # 添加节点
    graph.add_node("planner", planner, aflow=True)
    graph.add_node("executor", executor)
    graph.add_node("tools", tool_node)
    graph.add_node("checker", checker, aflow=True)
    graph.add_node("summarizer", summarizer, aflow=True)
    
    # 设置入口点
    graph.set_entry_point("planner")
    
    # 连接节点
    graph.add_edge("planner", "executor")
    
    graph.add_conditional_edges(
        "executor",
        lambda x: x.get("next", "end"),
        {
            "tools": "tools",
            "check": "checker",
            "summarizer": "summarizer",
            "end": END
        }
    )
    
    graph.add_edge("tools", "executor")
    
    graph.add_conditional_edges(
        "checker",
        lambda x: x.get("next", "end"),
        {
            "replan": "planner",
            "end": "summarizer"
        }
    )
    
    # summarizer 完成后结束
    graph.add_edge("summarizer", END)
    
    # 编译图（使用 checkpointer 实现持久化记忆）
    app_graph = graph.compile(name="bnerc_agent", checkpointer=checkpointer)
    
    logger.info("LangGraph 图编译完成（已启用持久化记忆）")
    return app_graph

# ================================
# FastAPI 生命周期管理
# ================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """ASGI 生命周期事件处理 - 启动和关闭"""
    # 启动时执行
    logger.info("应用启动中...")
    try:
        # 预初始化图（可选，也可以延迟初始化）
        # await init_graph()
        logger.info("应用启动完成")
    except Exception as e:
        logger.error(f"应用启动失败: {e}", exc_info=True)
        raise
    
    yield
    
    # 关闭时执行
    logger.info("应用关闭中...")
    try:
        # 清理资源
        global app_graph, llm_with_tools, checkpointer, db_pool
        app_graph = None
        llm_with_tools = None
        
        # 关闭连接池（如果使用连接池）
        if db_pool is not None:
            try:
                logger.info("正在关闭 PostgreSQL 连接池...")
                await db_pool.close()
                db_pool = None
                logger.info("PostgreSQL 连接池已关闭")
            except Exception as e:
                logger.error(f"关闭连接池时出错: {e}", exc_info=True)
        
        checkpointer = None
        logger.info("应用关闭完成")
    except Exception as e:
        logger.error(f"应用关闭时出错: {e}", exc_info=True)

# ================================
# FastAPI 应用
# ================================
# 从环境变量读取 CORS 配置，生产环境应限制域名
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",") if os.getenv("CORS_ORIGINS") else ["*"]

app = FastAPI(
    title="BNERC Server",
    description="基于 LangGraph 的 Plan & Execute Agent 服务",
    version="1.0.0",
    lifespan=lifespan  # 添加生命周期管理
)

# 中间件
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,  # 从环境变量读取，生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # 限制允许的方法
    allow_headers=["*"],
    max_age=3600,  # 预检请求缓存时间
)

# 请求模型
class RunRequest(BaseModel):
    assistant_id: str
    input: dict
    stream_mode: Optional[str] = "custom"

# ================================
# API 路由
# ================================
@app.post("/threads")
async def create_thread():
    """创建新的对话线程"""
    thread_id = str(uuid.uuid4())
    logger.info(f"创建新线程: {thread_id}")
    return {"thread_id": thread_id}

@app.get("/threads/{thread_id}")
async def get_thread(thread_id: str):
    """获取线程的历史状态（用于恢复对话）"""
    try:
        global app_graph
        if app_graph is None:
            app_graph = await init_graph()
        
        config = {"configurable": {"thread_id": thread_id}}
        state = await app_graph.aget_state(config)
        
        if state and state.values:
            # 提取消息历史
            messages = state.values.get("messages", [])
            message_list = []
            for msg in messages:
                if hasattr(msg, 'type') and hasattr(msg, 'content'):
                    message_list.append({
                        "role": "user" if msg.type == "human" else "assistant",
                        "content": extract_text(msg.content) if hasattr(msg, 'content') else str(msg.content)
                    })
            
            return {
                "thread_id": thread_id,
                "exists": True,
                "messages": message_list,
                "message_count": len(message_list)
            }
        else:
            return {
                "thread_id": thread_id,
                "exists": False,
                "messages": [],
                "message_count": 0
            }
    except Exception as e:
        logger.error(f"获取线程状态失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取线程状态失败: {str(e)}")

@app.post("/threads/{thread_id}/runs/stream")
async def run_stream(thread_id: str, run_request: RunRequest, request: Request):
    """流式运行对话"""
    # 生成请求ID用于追踪
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] 线程 {thread_id} 开始流式运行")
    
    if run_request.assistant_id != "bridge_bindtools":
        logger.warning(f"[{request_id}] 无效的 assistant_id: {run_request.assistant_id}")
        raise HTTPException(status_code=400, detail="Invalid assistant_id")
    
    try:
        return StreamingResponse(
            stream_handler(run_request, thread_id, request_id),
            media_type="text/event-stream",
            headers={
                "X-Request-ID": request_id,
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        logger.error(f"[{request_id}] 流式处理启动失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Stream initialization failed")

async def stream_handler(run_request: RunRequest, thread_id: str, request_id: str):
    """流式响应处理 - 使用后台任务 + 队列"""
    queue = asyncio.Queue()
    
    async def run_graph_task():
        """后台任务：执行图并将事件放入队列"""
        try:
            global app_graph
            if app_graph is None:
                logger.info(f"[{request_id}] 初始化 LangGraph 图...")
                app_graph = await init_graph()
            
            # 构建输入
            input_data = run_request.input
            messages = input_data.get("messages", [])
            files = input_data.get("files", [])
            
            # 尝试从 checkpointer 恢复历史状态
            config = {"configurable": {"thread_id": thread_id}}
            existing_state = None
            
            try:
                # 获取现有状态（如果存在）
                existing_state = await app_graph.aget_state(config)
                if existing_state and existing_state.values:
                    logger.info(f"[{request_id}] 恢复线程 {thread_id} 的历史状态，已有 {len(existing_state.values.get('messages', []))} 条消息")
                    # 使用历史状态中的 messages
                    existing_messages = existing_state.values.get("messages", [])
                    # 只添加新的用户消息（避免重复）
                    new_user_messages = [msg for msg in messages if msg["role"] == "user"]
                    for msg in new_user_messages:
                        existing_messages.append(HumanMessage(content=msg["content"]))
                    
                    initial_state = {
                        "messages": existing_messages,
                        "run_id": existing_state.values.get("run_id", str(uuid.uuid4())),
                        "files": files or existing_state.values.get("files", [])
                    }
                else:
                    # 没有历史状态，创建新的
                    logger.info(f"[{request_id}] 线程 {thread_id} 是新对话，创建初始状态")
                    langchain_messages = []
                    for msg in messages:
                        if msg["role"] == "user":
                            langchain_messages.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant":
                            langchain_messages.append(AIMessage(content=msg["content"]))
                    
                    initial_state = {
                        "messages": langchain_messages,
                        "run_id": str(uuid.uuid4()),
                        "files": files
                    }
            except Exception as e:
                logger.warning(f"[{request_id}] 恢复状态失败: {e}，使用新状态")
                # 恢复失败，使用新状态
                langchain_messages = []
                for msg in messages:
                    if msg["role"] == "user":
                        langchain_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        langchain_messages.append(AIMessage(content=msg["content"]))
                
                initial_state = {
                    "messages": langchain_messages,
                    "run_id": str(uuid.uuid4()),
                    "files": files
                }
            
            logger.info(f"[{request_id}] 开始执行 Plan & Execute 工作流")
            
            # 使用 astream 执行图
            async for event in app_graph.astream(
                initial_state,
                config,
                stream_mode=["custom", "updates"]
            ):
                if isinstance(event, tuple) and len(event) == 2:
                    mode, data = event
                    
                    if mode == "custom":
                        logger.debug(f"[{request_id}] [custom] {data}")
                        await queue.put(data)
                    elif mode == "updates" and data:
                        node_name = list(data.keys())[0]
                        logger.info(f"[{request_id}] [updates] 节点 [{node_name}] 完成")
                        
                        # 只发送节点完成事件，plan_final 由 planner 节点的 writer 发送
                        await queue.put({'event': 'node_complete', 'node': node_name})
                else:
                    if isinstance(event, dict):
                        await queue.put(event)
            
            logger.info(f"[{request_id}] 工作流执行完成")
            
        except asyncio.CancelledError:
            logger.warning(f"[{request_id}] 图执行被取消")
            await queue.put({'error': 'Cancelled'})
        except Exception as e:
            logger.error(f"[{request_id}] 图执行失败: {e}", exc_info=True)
            await queue.put({'error': str(e)})
        finally:
            await queue.put(None)  # 结束标记
    
    # 启动后台任务
    task = asyncio.create_task(run_graph_task())
    
    try:
        while True:
            try:
                data = await asyncio.wait_for(queue.get(), timeout=120.0)
                
                if data is None:
                    yield f"data: {json.dumps({'event': 'stream_end'})}\n\n"
                    break
                
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                
            except asyncio.TimeoutError:
                logger.warning(f"[{request_id}] 队列超时")
                yield f"data: {json.dumps({'error': 'timeout', 'request_id': request_id})}\n\n"
                break
    except asyncio.CancelledError:
        logger.info(f"[{request_id}] 流式处理被取消")
        yield f"data: {json.dumps({'error': 'cancelled', 'request_id': request_id})}\n\n"
    except Exception as e:
        logger.error(f"[{request_id}] 流式处理失败: {e}", exc_info=True)
        yield f"data: {json.dumps({'error': str(e), 'request_id': request_id})}\n\n"
    finally:
        if not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=5.0)  # 等待最多5秒
            except (asyncio.CancelledError, asyncio.TimeoutError):
                logger.warning(f"[{request_id}] 后台任务清理超时")
                pass

@app.get("/threads")
async def list_threads(limit: int = 50, offset: int = 0):
    """列出所有线程（仅支持 PostgreSQL）"""
    try:
        global checkpointer
        
        # 检查是否使用 PostgreSQL
        is_postgres = False
        if checkpointer is not None:
            checkpointer_class_name = type(checkpointer).__name__
            is_postgres = "Postgres" in checkpointer_class_name
        
        if not is_postgres:
            # 检查是否是 MemorySaver
            from langgraph.checkpoint.memory import MemorySaver
            if isinstance(checkpointer, MemorySaver):
                return {
                    "error": "内存存储不支持列出所有线程",
                    "hint": "请配置 PostgreSQL 以查看所有历史线程"
                }
        
        # 如果使用 PostgreSQL，直接查询数据库
        db_conn_string = os.getenv("DB_CONN_STRING", "")
        if db_conn_string and POSTGRES_AVAILABLE:
            import asyncpg
            from urllib.parse import urlparse
            
            # 解析连接字符串
            parsed = urlparse(db_conn_string.replace("postgresql+asyncpg://", "postgresql://"))
            if parsed.scheme == "":
                parsed = urlparse("postgresql://" + db_conn_string)
            
            conn = await asyncpg.connect(
                host=parsed.hostname or "localhost",
                port=parsed.port or 5432,
                user=parsed.username or "postgres",
                password=parsed.password or "",
                database=parsed.path.lstrip("/") if parsed.path else "postgres"
            )
            
            try:
                # 查询所有唯一的 thread_id
                query = """
                    SELECT DISTINCT thread_id, 
                           MAX(checkpoint_id) as latest_checkpoint,
                           COUNT(*) as checkpoint_count
                    FROM checkpoints
                    GROUP BY thread_id
                    ORDER BY latest_checkpoint DESC
                    LIMIT $1 OFFSET $2
                """
                rows = await conn.fetch(query, limit, offset)
                
                # 查询总数
                count_query = "SELECT COUNT(DISTINCT thread_id) as total FROM checkpoints"
                total_row = await conn.fetchrow(count_query)
                total = total_row['total'] if total_row else 0
                
                threads = []
                for row in rows:
                    threads.append({
                        "thread_id": row['thread_id'],
                        "checkpoint_count": row['checkpoint_count'],
                        "latest_checkpoint": row['latest_checkpoint']
                    })
                
                return {
                    "threads": threads,
                    "total": total,
                    "limit": limit,
                    "offset": offset
                }
            finally:
                await conn.close()
        else:
            return {
                "error": "PostgreSQL 未配置",
                "hint": "请设置 USE_POSTGRES=true 和 DB_CONN_STRING 环境变量"
            }
    except Exception as e:
        logger.error(f"列出线程失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"列出线程失败: {str(e)}")

@app.get("/threads/{thread_id}/history")
async def get_thread_history(thread_id: str, limit: int = 100):
    """获取线程的完整历史（包括所有 checkpoints）"""
    try:
        global app_graph
        if app_graph is None:
            app_graph = await init_graph()
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # 获取当前状态
        state = await app_graph.aget_state(config)
        
        if not state or not state.values:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        # 提取完整信息
        messages = state.values.get("messages", [])
        message_list = []
        for idx, msg in enumerate(messages):
            msg_info = {
                "index": idx,
                "type": getattr(msg, 'type', 'unknown'),
                "content": extract_text(msg.content) if hasattr(msg, 'content') else str(msg.content),
            }
            
            # 添加额外信息
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                msg_info["tool_calls"] = [
                    {
                        "name": tc.get("name", ""),
                        "args": tc.get("args", {})
                    } for tc in msg.tool_calls
                ]
            
            if hasattr(msg, 'tool_call_id'):
                msg_info["tool_call_id"] = msg.tool_call_id
            
            message_list.append(msg_info)
        
        # 获取其他状态信息
        state_info = {
            "thread_id": thread_id,
            "run_id": state.values.get("run_id"),
            "plan": state.values.get("plan", []),
            "current_step": state.values.get("current_step", 0),
            "files": state.values.get("files", []),
            "message_count": len(message_list),
            "messages": message_list[:limit]  # 限制返回数量
        }
        
        return state_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取线程历史失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取线程历史失败: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查 - 生产环境应检查关键依赖"""
    import time
    global checkpointer, db_pool
    
    # 检查 checkpointer 类型
    checkpointer_type = "未初始化"
    checkpointer_status = "unknown"
    is_postgres = False
    db_connection_status = "unknown"
    checkpoint_count = 0
    thread_count = 0
    
    if checkpointer is not None:
        checkpointer_type = type(checkpointer).__name__
        is_postgres = "Postgres" in checkpointer_type or "PostgresSaver" in checkpointer_type
        
        if is_postgres:
            checkpointer_status = "active"
            # 检查连接池状态
            if db_pool is not None:
                try:
                    # 检查连接池是否打开
                    if hasattr(db_pool, '_pool') and db_pool._pool:
                        db_connection_status = "connected"
                    else:
                        db_connection_status = "pool_not_ready"
                except Exception:
                    db_connection_status = "unknown"
            
            # 尝试查询数据库统计信息
            try:
                db_conn_string = os.getenv("DB_CONN_STRING", "")
                if db_conn_string:
                    from psycopg import AsyncConnection
                    conninfo = db_conn_string.replace("postgresql+asyncpg://", "postgresql://")
                    async with await AsyncConnection.connect(conninfo) as conn:
                        async with conn.cursor() as cur:
                            # 查询 checkpoint 数量
                            await cur.execute("SELECT COUNT(*) FROM checkpoints")
                            result = await cur.fetchone()
                            checkpoint_count = result[0] if result else 0
                            
                            # 查询 thread 数量
                            await cur.execute("SELECT COUNT(DISTINCT thread_id) FROM checkpoints")
                            result = await cur.fetchone()
                            thread_count = result[0] if result else 0
            except Exception as e:
                logger.debug(f"查询数据库统计信息失败: {e}")
        else:
            from langgraph.checkpoint.memory import MemorySaver
            if isinstance(checkpointer, MemorySaver):
                checkpointer_status = "memory_only"
            else:
                checkpointer_status = "active"
    else:
        checkpointer_status = "not_initialized"
    
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "bnerc_server",
        "version": "1.0.0",
        "checkpointer": {
            "type": checkpointer_type,
            "status": checkpointer_status,
            "is_postgres": is_postgres,
        "postgres_available": POSTGRES_AVAILABLE
        },
        "database": {
            "connection_status": db_connection_status,
            "checkpoint_count": checkpoint_count,
            "thread_count": thread_count
        }
    }
    
    # 检查关键组件（可选）
    try:
        # 检查图是否已初始化（如果未初始化，延迟初始化也可以）
        if app_graph is None:
            health_status["graph_initialized"] = False
        else:
            health_status["graph_initialized"] = True
        
        # 可以添加更多健康检查，如数据库连接、MCP 连接等
        return health_status
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

# ================================
# ASGI 入口点
# ================================
asgi_app = app

# ================================
# 主函数
# ================================
if __name__ == "__main__":
    import uvicorn
    import signal
    import sys
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "2026"))
    workers = int(os.getenv("WORKERS", "1"))  # 工作进程数
    timeout_keep_alive = int(os.getenv("TIMEOUT_KEEP_ALIVE", "60"))  # 保持连接超时
    timeout_graceful_shutdown = int(os.getenv("TIMEOUT_GRACEFUL_SHUTDOWN", "30"))  # 优雅关闭超时
    
    logger.info(f"启动 BNERC Server: http://{host}:{port}")
    logger.info(f"工作进程数: {workers}")
    logger.info(f"保持连接超时: {timeout_keep_alive}s")
    logger.info(f"优雅关闭超时: {timeout_graceful_shutdown}s")
    
    # 生产环境配置
    uvicorn_config = {
        "app": "bnerc_server:asgi_app",
        "host": host,
        "port": port,
        "reload": False,  # 生产环境禁用自动重载
        "workers": workers,
        "log_level": "info",
        "access_log": True,
        "timeout_keep_alive": timeout_keep_alive,
        "timeout_graceful_shutdown": timeout_graceful_shutdown,
        "limit_max_requests": 100000,  # 每个工作进程处理的最大请求数
        "limit_concurrency": None,  # 最大并发数（None 表示无限制，生产环境可设置）
        "backlog": 2048,  # 连接队列大小
    }
    
    try:
        uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        logger.info("收到键盘中断，退出服务")
    except Exception as e:
        logger.error(f"服务启动失败: {e}", exc_info=True)
        sys.exit(1)
