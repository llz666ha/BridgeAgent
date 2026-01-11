import json
from typing import Dict, List, Any
from langgraph.checkpoint.memory import MemorySaver
# import sys
# sys.path.append('/root/langgraph')
from utils.file import load_prompt_from_yaml

from langgraph.graph import StateGraph, END
from tools.tools import TOOLS, TOOL_DESCRIPTIONS
from typing import Any, Optional
from typing_extensions import TypedDict
from utils.draw import draw_workflow
import logging
import inspect
import asyncio
from dotenv import load_dotenv
load_dotenv()
# 配置logging基本设置
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为INFO，这样INFO及以上级别的日志都会显示
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 日志格式
    handlers=[
        logging.StreamHandler()  # 输出到控制台的处理器
    ]
)
logger = logging.getLogger(__name__)

# ================================
# 2. LLM 定义
# ================================

# 从utils导入全局LLM实例
from utils.utils import get_llm_instance
llm = get_llm_instance()

# 工具函数已移至tools/tools.py文件中
# 创建checkpointer，还没用
memory = MemorySaver()

# 导入MCP客户端
from .mcp.mcp_client import MCP_CLIENT
# 导入时间模块用于缓存
import time

# MCP工具列表缓存
mcp_tools_cache = None
cache_timestamp = 0
cache_duration = 60  # 缓存有效期，单位：秒

# ================================
# 3. State 定义
# ================================

class State(TypedDict,total=False):
    messages: List[Dict[str, Any]]          # 对话 / 任务进度
    plan: List[Dict[str, Any]]              # Planner 生成的计划
    current_step: int                          # 当前执行到第几步 
    current_tool: Optional[Dict[str, Any]]     # 当前要调用的工具
    current_input: Optional[Dict[str, Any]]    # 当前工具的输入参数
    result: Dict[str, Any]              # 工具执行结果
    replan: bool                            # 是否需要重新规划
    reply: Optional[str]                    # 最终回复

# ================================
# 4. 加载提示词模板
# ================================

# 加载提示词模板
planner_prompt_data = load_prompt_from_yaml('planner.yaml')
router_prompt_data = load_prompt_from_yaml('router.yaml')
checker_prompt_data = load_prompt_from_yaml('checker.yaml')
summarizer_prompt_data = load_prompt_from_yaml('summarizer.yaml')

# ====================================
# 1. Planner（生成工具执行计划）
# ====================================

async def get_mcp_tools():
    """获取MCP工具列表，使用缓存机制"""
    global mcp_tools_cache, cache_timestamp
    current_time = time.time()
    
    # 检查缓存是否有效
    if mcp_tools_cache and (current_time - cache_timestamp < cache_duration):
        logger.info(f"使用缓存的MCP工具列表，缓存时间: {cache_timestamp}")
        return mcp_tools_cache
    
    # 缓存无效，重新获取
    try:
        mcp_tools_list = await MCP_CLIENT.list_tools()
        mcp_tools = {tool['name']: tool for tool in mcp_tools_list}
        mcp_tools_cache = mcp_tools
        cache_timestamp = current_time
        logger.info(f"获取到新的MCP工具列表，缓存时间: {cache_timestamp}")
        return mcp_tools
    except Exception as e:
        logger.error(f"获取MCP工具失败: {e}")
        # 返回空字典，确保程序继续执行
        return {}

def process_tool_result(tool_result, is_mcp_tool, current_step, retry_count, status="success", error_info=None):
    """处理工具执行结果，确保结果可序列化并添加必要信息"""
    extracted_data = tool_result
    
    # 处理CallToolResult对象
    if hasattr(tool_result, 'content') and hasattr(tool_result, 'is_error'):
        # 这是一个CallToolResult对象
        if tool_result.is_error:
            status = "error"
            extracted_data = f"工具调用失败: {tool_result}"
        else:
            # 提取ContentBlock中的实际内容
            if hasattr(tool_result, 'content') and tool_result.content:
                # 获取第一个ContentBlock
                content_block = tool_result.content[0]
                if hasattr(content_block, 'text'):
                    # 获取text内容
                    text_content = content_block.text
                    try:
                        # 尝试解析JSON字符串
                        extracted_data = json.loads(text_content)
                    except json.JSONDecodeError:
                        # 如果不是有效的JSON，直接使用text内容
                        extracted_data = text_content
    
    # 构建结果字典
    result_dict = {
        "status": status,
        "data": extracted_data,
        "retries": retry_count,
        "tool_type": "mcp" if is_mcp_tool else "local",
        "step": current_step + 1
    }
    
    # 添加错误信息（如果有）
    if status == "error" and error_info:
        result_dict["message"] = f"工具调用错误: {str(error_info)}"
        # 分类错误类型
        error_type = "unknown_error"
        error_str = str(error_info).lower()
        if "filenotfounderror" in str(type(error_info)).lower():
            error_type = "file_not_found"
        elif "timeout" in error_str:
            error_type = "timeout"
        elif "permission" in error_str:
            error_type = "permission_denied"
        elif "api" in error_str:
            error_type = "api_error"
        result_dict["error_type"] = error_type
    
    return result_dict

def get_result_key(tool_name, existing_results):
    """获取工具执行结果的键名，处理重复调用情况"""
    # 处理多次调用同一工具的情况，为结果添加步骤索引
    tool_index = 1
    result_key = tool_name
    
    # 如果已存在该工具的结果，添加索引
    while result_key in existing_results:
        tool_index += 1
        result_key = f"{tool_name}_{tool_index}"
    
    return result_key

async def planner(state: State):
    """生成执行计划，写入 plan 和初始化状态，并将图片路径加入 result"""
    messages = state.get("messages", [])
    
    # 获取用户输入
    user_input = messages[-1]['content'] if messages else ""
    
    # 获取MCP工具列表（使用缓存）
    mcp_tools = await get_mcp_tools()
    
    # 合并本地工具描述和MCP工具描述
    combined_tool_descriptions = TOOL_DESCRIPTIONS
    if mcp_tools:
        mcp_tool_descriptions = "\n".join([
            f"- {name}: {tool['description']}"
            for name, tool in mcp_tools.items()
        ])
        combined_tool_descriptions += "\n" + mcp_tool_descriptions
    
    # 使用从YAML加载的提示词，传入合并后的工具描述
    system_prompt = planner_prompt_data['system'].format(TOOL_DESCRIPTIONS=combined_tool_descriptions)
    human_prompt = planner_prompt_data['human'].format(user_input=user_input)
    
    # 使用LLM调用生成计划
    response = await llm.ainvoke(
            [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": human_prompt
                }
            ]
        )
    # # 检查是否有ainvoke方法（异步调用）
    # if hasattr(llm, "ainvoke"):
    #     response = await llm.ainvoke(
    #         [
    #             {
    #                 "role": "system",
    #                 "content": system_prompt,
    #             },
    #             {
    #                 "role": "user",
    #                 "content": human_prompt
    #             }
    #         ]
    #     )
    # else:
    #     response = llm.invoke(
    #         [
    #             {
    #                 "role": "system",
    #                 "content": system_prompt,
    #             },
    #             {
    #                 "role": "user",
    #                 "content": human_prompt
    #             }
    #         ]
    #     )
    
    try:
        # 解析响应为JSON
        response_content = response.content
        # logger.info(response_content)
        parsed_response = json.loads(response_content)
        logger.info(parsed_response)
        
        # 验证响应格式是否正确
        if not isinstance(parsed_response, dict) or "plan" not in parsed_response or "resources" not in parsed_response:
            error_message = "计划格式错误，缺少必要字段"
            return {
                "plan": [],
                "current_step": 0,
                "messages": messages + [
                    {"role": "assistant", "content": error_message}
                ],
            }
        logger.info(f"Planner: {parsed_response['plan']}")

        # 绘制工作流程图并保存到workflow目录
        draw_workflow(parsed_response["plan"])
        
        # 获取用户问题（从messages中提取最后一条用户消息）
        user_question = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_question = msg.get("content", "")
                break
                
        # 处理执行计划，确保只包含有效的tool字段
        processed_plan = []
        for step in parsed_response["plan"]:
            tool_name = step.get("tool", "")
            if tool_name:  # 只保留有有效tool名称的步骤
                processed_step = {"tool": tool_name}
                processed_plan.append(processed_step)
        logger.info(f"Planner: 处理后的执行计划: {processed_plan}")
        
        # 返回更新后的状态
        # 计划更新到State里
        # 资源路径以键值对形式加入State的result里
        # 用户问题以键值对形式（键为"query"）加入State的result里
        return {
            "plan": processed_plan,
            "current_step": 0,
            "result": {
                "query": user_question,
                **parsed_response["resources"]  # 将所有资源路径直接添加到result中
            },
            "messages": messages + [
                {"role": "assistant", "content": response_content}
            ],
        }
    except json.JSONDecodeError:
        # 如果解析失败，返回错误消息
        error_message = "计划生成失败，请重试"
        return {
            "plan": [],
            "current_step": 0,
            "messages": messages + [
                {"role": "assistant", "content": error_message}
            ],
        }
    except Exception as e:
        # 捕获其他异常
        error_message = f"计划生成过程中发生错误: {str(e)}"
        return {
            "plan": [],
            "current_step": 0,
            "messages": messages + [
                {"role": "assistant", "content": error_message}
            ],
        }

# ====================================
# 2. Executor（决定当前执行哪个步骤）
# ====================================

def executor(state: State):
    """根据计划决定下一个要执行的工具（不负责输入构造）"""
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)

    # ---------- 空计划检查：如果计划为空，直接完成 ----------
    if not plan:
        logger.info("Executor: 计划为空，无需执行工具，直接进入检查阶段")
        return {
            "is_plan_completed": True,
            "next": "check",
            "current_step": current_step,
            "plan": plan  # 保留执行计划
        }

    # ---------- 是否执行完 ----------
    if current_step >= len(plan):
        logger.info(f"Executor: 计划已完成，共执行 {current_step} 步")
        return {
            "is_plan_completed": True,
            "next": "check",
            "current_step": current_step,
            "plan": plan  # 保留执行计划
        }

    # ---------- 执行当前步骤 ----------
    step = plan[current_step]
    logger.info(f"当前步骤: {step}")
    current_tool = step.get("tool", "")
    
    logger.info(f"Executor: 准备执行步骤 {current_step + 1}/{len(plan)} → 工具 {current_tool}")
    # 返回当前工具和步骤信息，交给router处理
    return {
        "current_tool": current_tool,
        "current_step": current_step,
        "next": "route",
        "plan": plan  # 保留执行计划
    }

# ====================================
# 3. Router（真正执行工具 + 统一构造输入 + 写入结果）
# ====================================

async def router(state: State):
    """统一的工具执行器（所有工具调用都走这里）"""
    tool_name = state.get("current_tool")
    current_step = state.get("current_step", 0)
    if not tool_name:
        logger.info("工具名错误")
        return {
            "tool_result": "错误：没有工具可执行",
            "current_step": current_step + 1,
            "next": "executor"
        }

    logger.info(f"Router: 调用工具 {tool_name}")
    tool_fn = TOOLS.get(tool_name)
    is_mcp_tool = False

    # 获取工具函数的参数schema和required参数
    tool_params = []
    
    # 一次性获取工具的schema信息
    tool_description = ""
    tool_schema = ""
    if tool_fn and hasattr(tool_fn, "schema"):
        # 通过@tool装饰器获取schema
        schema = tool_fn.args_schema.model_json_schema()
        tool_params = schema.get("required", [])
        logger.info(f"Router: 本地工具 {tool_name} 需要的参数: {tool_params}")
        
        # 对于本地工具@tool，从schema提取出描述。从schema中提取description作为tool_description
        tool_description = schema.pop("description", f"{tool_name}工具")
        # 简化schema，只保留核心字段，与MCP工具schema保持一致
        tool_schema = {
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
            "title": schema.get("title", tool_name),
            "type": schema.get("type", "object")
        }
        
        logger.info(f"Router: 本地工具 {tool_name} 的schema: {tool_schema}")
        logger.info(f"Router: 本地工具生成的描述: {tool_description}")
    else:
        # 检查是否为MCP工具
        try:
            # 获取MCP工具列表（使用缓存）
            mcp_tools_dict = await get_mcp_tools()
            
            # 查找指定的MCP工具
            if tool_name in mcp_tools_dict:
                is_mcp_tool = True
                tool_info = mcp_tools_dict[tool_name]
                
                # mcp工具的描述和schema是分开的，获取工具描述和inputSchema
                tool_description = tool_info.get('description', '')
                logger.info(f"Router: MCP工具 {tool_name} 的描述: {tool_description}")
                tool_schema = tool_info.get('inputSchema', {})
                
                logger.info(f"Router: MCP工具 {tool_name} 的schema: {tool_schema}")
        except Exception as e:
            logger.info(f"Router: 检查MCP工具失败: {str(e)}")

    # =============== 使用大模型提取工具参数 ===============
    tool_input = {}
    
    result_content = state["result"]
    messages = state.get("messages", [])
    
    # # 构建包含完整对话历史的上下文
    # context_content = {
    #     "messages": messages,
    #     "result": result_content,
    # }
    
    # 使用router.yaml中的提示词，传递工具信息和schema
    prompt = router_prompt_data['human'].format(
        result_content=result_content,
        tool_description=tool_description,
        tool_schema=tool_schema
    )
    
    # logger.info(f"Router: 构建的提示词: {prompt}")
    
    try:
        # 使用大模型提取参数，使用异步方式避免阻塞
        import asyncio
        if hasattr(llm, "ainvoke"):
            # 如果llm支持异步调用，直接使用
            response = await llm.ainvoke(prompt)
        else:
            # 如果llm不支持异步调用，使用asyncio.to_thread避免阻塞
            response = await asyncio.to_thread(llm.invoke, prompt)
        
        extracted_json = response.content.strip()
        
        # 清理可能的格式标记
        if extracted_json.startswith('```json'):
            extracted_json = extracted_json[7:]
        if extracted_json.endswith('```'):
            extracted_json = extracted_json[:-3]
        
        # 清理JSON字符串
        extracted_json = extracted_json.strip()
        extracted_params = json.loads(extracted_json)
        
        # 处理大模型可能返回的嵌套参数格式（如{'tool_name': {'param1': 'value1'}}）
        if isinstance(extracted_params, dict):
            # 检查是否包含工具名作为根键
            if tool_name in extracted_params and isinstance(extracted_params[tool_name], dict):
                # 如果是嵌套格式，提取内部的参数
                extracted_params = extracted_params[tool_name]
                logger.info(f"Router: 处理嵌套参数格式，提取内部参数: {extracted_params}")
        
        # 将提取的参数添加到tool_input中
        # 如果有明确的参数列表，只使用这些参数；否则使用所有提取的参数
        if tool_params:
            for param in tool_params:
                if param in extracted_params:
                    tool_input[param] = extracted_params[param]
        else:
            # 使用所有提取的参数
            for param, value in extracted_params.items():
                tool_input[param] = value
        
        logger.info(f"Router: 提取参数: {tool_input}")
        
    except Exception as e:
        logger.info(f"Router: 大模型提取参数失败: {str(e)}")
        # 失败时，只记录日志，不做特殊处理
        pass
    
    # logger.info(f"Router: 工具 {tool_name} 最终输入参数: {tool_input}")

    # 最大重试次数
    MAX_RETRIES = 1
    retry_count = 0
    tool_executed = False
    error_info = None
    
    if not tool_fn and not is_mcp_tool:
        logger.info(f"未知工具：{tool_name}")
        tool_result = {"status": "error", "message": f"未知工具：{tool_name}", "details": None}
        tool_executed = True
    else:
        # 尝试执行工具，支持重试
        while retry_count <= MAX_RETRIES and not tool_executed:
            try:
                # 记录重试信息
                if retry_count > 0:
                    logger.info(f"Router: 工具 {tool_name} 第 {retry_count} 次重试...")
                
                if is_mcp_tool:
                    logger.info(f"Router: MCP工具 {tool_name} 的参数: {tool_input}")
                    
                    # 执行MCP工具，直接使用原始参数
                    tool_result = await MCP_CLIENT.call_tool(tool_name, **tool_input)
                    logger.info(f"Router: MCP工具 {tool_name} 执行成功,结果: {tool_result}")
                else:
                    logger.info(f"Router: 本地工具 {tool_name} 的参数: {tool_input}")
                    
                    # 执行本地异步工具，直接使用原始参数
                    tool_result = await tool_fn.ainvoke(tool_input)
                    logger.info(f"Router: 本地工具 {tool_name} 执行成功,结果: {tool_result}")
                
                # 创建新的result字典，保留原有内容并添加新结果
                new_result = state["result"].copy() if isinstance(state["result"], dict) else {}
                
                # 处理工具执行结果
                result_dict = process_tool_result(tool_result, is_mcp_tool, current_step, retry_count)
                # 获取结果键名
                result_key = get_result_key(tool_name, new_result)
                # 添加到结果字典
                new_result[result_key] = result_dict                
                tool_executed = True
                
            except Exception as e:
                retry_count += 1
                error_info = e
                logger.info(f"Router: 工具 {tool_name} 执行失败 ({retry_count}/{MAX_RETRIES}): {error_info}")
                
                # 处理失败情况
                new_result = state["result"].copy() if isinstance(state["result"], dict) else {}
                # 处理工具执行结果（错误情况）
                result_dict = process_tool_result(str(error_info), is_mcp_tool, current_step, retry_count - 1, "error", error_info)
                # 获取结果键名
                result_key = get_result_key(tool_name, new_result)
                # 添加到结果字典
                new_result[result_key] = result_dict
    
    # 记录错误日志（如果有）
    if not tool_executed and error_info:
        logger.error(f"Router: 工具 {tool_name} 执行失败，已达到最大重试次数: {error_info}")
    
    # 确保返回值包含plan字段，以便执行计划能够正确传递
    return {
        "current_step": current_step + 1,
        "result": new_result,
        "next": "executor",
        "tool_execution_status": "success" if tool_executed else "failed",
        "plan": state.get("plan", [])  # 保留执行计划
    }

# ====================================
# 4. Checker（检查是否满足用户需求）
# ====================================

async def checker(state: State):
    """检查最终执行结果，使用LLM评估是否需要重规划"""
    user_query = state["messages"][0]["content"]
    # 确保state["result"]是可序列化的
    try:
        final_result = json.dumps(state["result"], ensure_ascii=False)
    except TypeError:
        # 如果直接序列化失败，转换为字符串
        final_result = str(state["result"])

    logger.info("\nChecker: 评估执行结果是否满足需求")
    logger.info(f"用户问题: {user_query}")
    logger.info(f"执行结果: {final_result}")

    # 使用LLM评估执行结果是否满足用户需求
    # 使用从YAML加载的提示词
    system_prompt = checker_prompt_data['system']
    evaluation_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": checker_prompt_data['human'].format(
            user_query=user_query,
            final_result=final_result
        )}
    ]
    
    try:
        # 使用异步方式调用llm，避免阻塞
        import asyncio
        if hasattr(llm, "ainvoke"):
            response = await llm.ainvoke(evaluation_prompt)
        else:
            response = await asyncio.to_thread(llm.invoke, evaluation_prompt)
        
        evaluation_result = json.loads(response.content.strip())
        
        logger.info(f"Checker: 评估结果 - 满足需求: {evaluation_result['satisfies_needs']}, 需要重新规划: {evaluation_result['needs_replan']}")
        logger.info(f"Checker: 评估理由: {evaluation_result['reason']}")
        # 暂不重规划
        evaluation_result["needs_replan"]=False
        # 如果需要重新规划，将当前执行结果添加到messages中，用于重新规划
        if evaluation_result["needs_replan"]:
            state["messages"].append({
                "role": "assistant",
                "content": f"当前执行结果不满足需求: {evaluation_result['reason']}\n\n当前执行结果: {final_result}"
            })
            
            return {
                "replan": True,
                "next": "replan",  # 修改为与条件边配置匹配的值
                "plan": state.get("plan", [])  # 保留执行计划
            }
        else:
            # 检查通过，将进入summarizer节点生成最终回复
            if not evaluation_result["satisfies_needs"]:
                # 如果不满足需求但也不重新规划，预先设置一个基本回复
                state["reply"] = final_result
            return {
                "replan": False,
                "next": "end",
                "plan": state.get("plan", [])  # 保留执行计划
            }
    except Exception as e:
        logger.info(f"Checker: 评估过程出错: {str(e)}")
        # 出错时默认不重新规划
        return {
            "replan": False,
            "next": "end",
            "plan": state.get("plan", [])  # 保留执行计划
        }

# ====================================
# 5. Summarizer（生成最终回复）
# ====================================
async def summarizer(state: State):
    """使用LLM生成最终回复"""
    user_query = state["messages"][0]["content"]
    
    # 确保执行结果存在，使用更通用的处理方式
    result = state["result"]
    
    # 智能提取工具执行结果中的有用信息
    if isinstance(result, dict):
        # 尝试提取所有工具的执行结果
        tool_results = []
        for tool_name, tool_data in result.items():
            # 跳过资源信息和非工具结果
            if tool_name == "query" or isinstance(tool_data, str):
                continue
                
            if isinstance(tool_data, dict):
                # 处理工具执行结果
                if tool_data.get("status") == "success":
                    tool_data_content = tool_data.get('data', '')
                    # 确保tool_data_content是可序列化的
                    try:
                        serializable_data = json.dumps(tool_data_content, ensure_ascii=False)
                    except TypeError:
                        # 如果直接序列化失败，转换为字符串
                        serializable_data = str(tool_data_content)
                    tool_results.append(f"{tool_name}: {serializable_data}")
                else:
                    tool_results.append(f"{tool_name}: 执行失败 - {tool_data.get('message', '')}")
            else:
                # 确保tool_data是可序列化的
                try:
                    serializable_data = json.dumps(tool_data, ensure_ascii=False)
                except TypeError:
                    # 如果直接序列化失败，转换为字符串
                    serializable_data = str(tool_data)
                tool_results.append(f"{tool_name}: {serializable_data}")
        
        if tool_results:
            final_result = "\n".join(tool_results)
        else:
            # 确保result是可序列化的
            try:
                final_result = json.dumps(result, ensure_ascii=False)
            except TypeError:
                # 如果直接序列化失败，转换为字符串
                final_result = str(result)
    else:
        # 确保result是可序列化的
        try:
            final_result = json.dumps(result, ensure_ascii=False)
        except TypeError:
            # 如果直接序列化失败，转换为字符串
            final_result = str(result)

    logger.info("\nSummarizer: 生成最终回复")
    
    # 使用从YAML加载的提示词
    system_prompt = summarizer_prompt_data['system']
    summary_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": summarizer_prompt_data['human'].format(
            user_query=user_query,
            final_result=final_result
        )}
    ]
    
    try:
        # 使用异步方式调用llm，避免阻塞
        import asyncio
        if hasattr(llm, "ainvoke"):
            response = await llm.ainvoke(summary_prompt)
        else:
            response = await asyncio.to_thread(llm.invoke, summary_prompt)
        
        state["reply"] = response.content.strip()
        logger.info(f"Summarizer: 生成的最终回复: {state['reply']}")
        return {
            **state,
            "next": "end",
            "plan": state.get("plan", [])  # 保留执行计划
        }
    except Exception as e:
        logger.error(f"Summarizer: 生成回复时出错: {str(e)}")
        # 出错时使用执行结果作为回复
        state["reply"] = final_result
        return {
            **state,
            "next": "end",
            "plan": state.get("plan", [])  # 保留执行计划
        }

# ================================
# 8. 构建 LangGraph
# ================================

# graph = StateGraph[State, None, State, State](State)
graph = StateGraph(State)

# 添加节点
graph.add_node("planner", planner, aflow=True)  # Planner现在是异步节点
graph.add_node("executor", executor)  # 执行计划管理步骤
graph.add_node("router", router, aflow=True)     # 执行工具处理输入输出（异步节点）
graph.add_node("checker", checker, aflow=True)   # 检查结果是否满足要求（异步节点）
graph.add_node("summarizer", summarizer, aflow=True)  # 生成最终回复（异步节点）

# 设置入口点
graph.set_entry_point("planner")

# 连接节点
graph.add_edge("planner", "executor")  # Planner生成计划后交给Executor

# 添加条件边，根据executor返回的状态决定下一步
graph.add_conditional_edges(
    "executor",
    # 条件函数：根据next字段决定下一个节点
    lambda x: x.get("next", "end"),
    {
        "route": "router",      # 需要执行工具，交给router
        "check": "checker",     # 计划完成，检查结果
        "end": END
    }
)

# Router执行工具后返回给Executor继续执行下一步
graph.add_edge("router", "executor")



# 检查器的条件边
graph.add_conditional_edges(
    "checker",
    # 条件函数：根据next字段决定下一个节点
    lambda x: x.get("next", "end"),
    {
        "replan": "planner",   # 需要重新规划，返回planner
        "end": "summarizer"   # 满足需求，生成最终回复
    }
)

# 编译图
# app = graph.compile(name="bridge", checkpointer=memory)
app = graph.compile(name="bridge")

# ================================
# 8. 运行
# ================================

# 确保app可以作为模块被正确导入
# 主函数部分，仅在直接运行时执行
async def run_agent(user_input, state=None):
    """运行agent处理用户输入"""
    if state is None:
        # 初始化状态
        state = {
            "messages": [{"role": "user", "content": user_input}],
            "plan": [],
            "current_step": 0,
            "result": {},
            "current_tool": "",
            "replan": False
        }
    else:
        # 在现有状态上添加新的用户消息
        state["messages"].append({"role": "user", "content": user_input})
        # 重置部分状态
        state["current_step"] = 0
        state["current_tool"] = ""
        state["replan"] = False
    
    logger.info(f"=== User: {user_input}")
    
    # 运行工作流
    if inspect.iscoroutinefunction(app.invoke):
        final_state = await app.invoke(state)
    else:
        if hasattr(app, 'ainvoke'):
            final_state = await app.ainvoke(state)
        else:
            # 如果必须同步调用，使用loop.run_in_executor
            loop = asyncio.get_event_loop()
            final_state = await loop.run_in_executor(None, lambda: app.invoke(state))
    
    # 优先使用state["reply"]作为最终回复
    logger.info(f"=== Final State: {final_state}")
    if "reply" in final_state:
        agent_reply = final_state["reply"]
        logger.info(f"=== Assistant: {agent_reply}")
        print(f"\n助手: {agent_reply}\n")
    else:
        # 备用方案：从messages中查找最后一条assistant消息
        agent_reply = "抱歉，无法生成回复。"
        for message in reversed(final_state["messages"]):
            if message["role"] == "assistant":
                agent_reply = message["content"]
                break
        logger.info(f"=== Assistant (备用): {agent_reply}")
        print(f"\n助手: {agent_reply}\n")
    
    return final_state

if __name__ == "__main__":
    print("=== LangGraph Agent 对话系统 ===")
    print("请输入您的问题或指令，输入'退出'或'exit'结束对话\n")
    
    state = None  # 用于维护对话历史状态
    
    # 进入对话循环
    while True:
        try:
            # 获取用户输入
            user_input = input("你: ")
            
            # 检查是否退出
            if user_input.lower() in ['退出', 'exit', 'quit', 'q']:
                print("感谢使用，再见！")
                break
            
            # 运行agent处理用户输入
            state = asyncio.run(run_agent(user_input, state))
            
        except KeyboardInterrupt:
            print("\n对话已中断，感谢使用！")
            break
        except Exception as e:
            logger.error(f"处理请求时出错: {str(e)}")
            print(f"发生错误: {str(e)}")
            # 继续对话，不退出