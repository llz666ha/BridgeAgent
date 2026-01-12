import json
from typing import Dict, List, Any
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import ToolNode
# import sys
# sys.path.append('/root/langgraph')
from utils.file import load_prompt_from_yaml
# 导入时间模块用于缓存
import time
from langgraph.graph import StateGraph, END, START,MessagesState
from langgraph.types import StreamWriter
from tools.tools import TOOLS
from typing import Any, Optional,Union
from typing_extensions import TypedDict
from utils.draw import draw_workflow
import logging
import asyncio
import uuid
from dotenv import load_dotenv
load_dotenv()
# 配置logging基本设置
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为INFO
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
from agent.mcp.mcp_client import client

llm = get_llm_instance()
# 创建一个全局变量来存储绑定了工具的LLM实例
llm_with_tools = None
# 工具函数已移至tools/tools.py文件中
# 注意：在LangGraph API环境中，自定义checkpointer会被忽略，平台会自动处理持久性
# MCP工具列表缓存
mcp_tools_cache = None
cache_timestamp = 0
cache_duration = 60  # 缓存有效期，单位：秒

# ================================
# 3. State 定义
# ================================

class State(MessagesState):
    # messages: Union[List[BaseMessage], List[dict]]          # 对话 / 任务进度
    run_id: str
    plan: List[Dict[str, Any]]              # Planner 生成的计划
    current_step: int                          # 当前执行到第几步 
    current_tool: Optional[Dict[str, Any]]     # 当前要调用的工具
    files: List[Dict[str, Any]]   # ← 文件元信息
    result: Dict[str, Any]              # 工具执行结果
    replan: bool                            # 是否需要重新规划
    reply: Optional[str]                    # 最终回复

# ================================
# 4. 加载提示词模板
# ================================

# 加载提示词模板
planner_prompt_data = load_prompt_from_yaml('planner2.yaml')
# router_prompt_data = load_prompt_from_yaml('router.yaml')
checker_prompt_data = load_prompt_from_yaml('checker.yaml')
summarizer_prompt_data = load_prompt_from_yaml('summarizer.yaml')

# ====================================
# 1. Planner（生成工具执行计划）
# ====================================

async def safe_get_tools() -> list:
    try:
        return await client.get_tools()
    except Exception as e:
        logger.warning(f"MCP 连接失败，使用空工具列表降级: {e}")
        return []

# 异步初始化函数，用于绑定工具到LLM
async def initialize_llm_with_tools():
    global llm_with_tools
    if llm_with_tools is None:
        mcp_tools = await safe_get_tools()
        llm_with_tools = llm.bind_tools(mcp_tools)
        logger.info("LLM已成功绑定工具")
    return llm_with_tools
# 初始化一个空的工具列表，稍后在应用启动时填充
tools = []
# 异步获取工具列表
async def get_tools_for_toolnode():
    return await safe_get_tools()

# 不能直接 await client.get_tools()
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

def extract_tool_results(messages):
    results = []
    for m in messages:
        if isinstance(m, ToolMessage):
            results.append({
                "tool_call_id": m.tool_call_id,
                "content": m.content,
            })
    return results

# conn   = aiosqlite.connect("checkpoints.db", isolation_level=None)
# saver  = SqliteSaver(conn)

def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # 常见格式：[{type: "text", text: "..."}]
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        return "\n".join(texts)
    return str(content)

def parse_tool_result(content) -> bool:
    """
    判断工具是否执行成功
    兼容 ToolMessage.content 的所有常见结构
    """
    try:
        # ToolMessage.content 是 list（LangChain 新结构）
        if isinstance(content, list) and content:
            block = content[0]
            if isinstance(block, dict) and "text" in block:
                raw = block["text"]
            else:
                return False

        # 直接是字符串
        elif isinstance(content, str):
            raw = content

        # 直接是 dict
        elif isinstance(content, dict):
            return content.get("status") == "success"

        else:
            return False

        # 解析 JSON 字符串
        data = json.loads(raw)
        return data.get("status") == "success"

    except Exception as e:
        logger.error(f"parse_tool_result failed: {e}")
        return False

def get_recent_run_tool_messages(messages, run_id):
    results = []
    message_count = len(messages)
    i = message_count - 1

    # 从尾部向前扫描，查找匹配的 ToolMessage
    while i >= 0:
        m = messages[i]
        if isinstance(m, ToolMessage):
            # 检查前一条消息是否为 AIMessage（工具调用），并验证 run_id
            if i >= 1:
                prev_msg = messages[i-1]
                if isinstance(prev_msg, AIMessage) and prev_msg.additional_kwargs.get("run_id") == run_id:
                    results.append(m)
                else:
                    # 遇到不属于本次 run 的 ToolMessage，终止扫描
                    break
            else:
                # ToolMessage 没有前一条消息，无法验证 run_id，终止扫描
                break
            i -= 1
        elif isinstance(m, AIMessage) and m.additional_kwargs.get("run_id") == run_id:
            # 继续向前扫描，可能还有更多的 ToolMessage
            i -= 1
        else:
            # 遇到不属于本次 run 的非 ToolMessage，终止扫描
            break

    return list(reversed(results))

def init_state():
    return {
        "messages": [],
        "run_id": str(uuid.uuid4()),
        "plan": [],
        "current_step": 0
    }

async def planner(state: State, writer: StreamWriter):
    """生成执行计划，写入 plan 和初始化状态"""
    messages = state.get("messages", [])
    run_id = state.get("run_id", str(uuid.uuid4()))
    logger.info("messages:" + str(messages))
    # 获取用户输入
    # user_input = messages[-1]['content'] if messages else ""
    # logger.info("用户输入:" + user_input)
    conversation = "\n".join(
        f"{m.type}: {m.content}" for m in messages[:-1] 
        if m.type in ['human', 'ai'] and hasattr(m, 'content') and m.content.strip()  # 只包含human和ai类型且内容非空
    )
    logger.info("历史对话:\n" + conversation)
    user_input = ""
    files = []

    if messages and isinstance(messages[-1], HumanMessage):
        # 提取文本内容
        user_input = extract_text(messages[-1].content)
    
    # 处理state中的文件（从独立的files字段提取）
    state_files = state.get("files", [])
    files.extend(state_files)
    
    logger.info(f"用户输入: {user_input}")
    if files:
        for f in files:
            logger.info(f"收到文件: {f.get('name')} (路径: {f.get('path')})")
    # 使用从YAML加载的提示词，传入合并后的工具描述
    system_prompt = planner_prompt_data['system'].format(user_input=user_input, conversation=conversation, files=files)
    # 必有的writer
    writer({"event": "plan_start", "text": "🎯 正在生成计划...\n"})
    # 确保LLM已绑定工具
    global llm_with_tools
    if llm_with_tools is None:
        await initialize_llm_with_tools()

    # 使用绑定了工具的LLM调用生成计划
    response = await llm_with_tools.ainvoke(
            [
                SystemMessage(content=system_prompt)
            ]
        )
    
    try:
        # 解析响应为JSON
        response_content = response.content if hasattr(response, 'content') else str(response)
        # logger.info(response_content)
        parsed_response = json.loads(response_content)
        logger.info(parsed_response)
        
        # 验证响应格式是否正确
        if not isinstance(parsed_response, dict) or "plan" not in parsed_response:
            error_message = "计划格式错误，缺少必要字段"
            return {
            "plan": [],
            "current_step": 0,
            "messages": messages + [AIMessage(content=error_message)],
            "run_id": run_id  # 保留或生成run_id
        }
        # logger.info(f"Planner: {parsed_response['plan']}")

        # 绘制工作流程图并保存到workflow目录
        draw_workflow(parsed_response["plan"])
        # 处理执行计划，确保只包含有效的tool字段
        processed_plan = []
        for step in parsed_response["plan"]:
            tool_name = step.get("tool", "")
            if tool_name:  # 只保留有有效tool名称的步骤
                processed_step = {"tool": tool_name}
                processed_plan.append(processed_step)
        logger.info(f"Planner: 处理后的执行计划: {processed_plan}")
        if writer:
            writer({"event": "plan_final", "plan": processed_plan})   # 只推成品
            writer({"event": "plan_end", "text": "\n"})
        # 返回更新后的状态
        # 计划更新到State里
        # 资源路径以键值对形式加入State的result里
        # messages.append(AIMessage(content=response_content))
        # logger.info(f"Planner: 最后的messages: {messages}")
        logger.info(f"Planner run_id: {run_id}")
        return {
            "plan": processed_plan,
            "current_step": 0,
            "messages": messages,
            "files": files,  # 将文件信息保存到state中
            "run_id": run_id  # 保留或生成run_id
        }
    except json.JSONDecodeError:
        # 如果解析失败，返回错误消息
        error_message = "计划生成失败，请重试"
        return {
            "plan": [],
            "current_step": 0,
            "messages": messages + [
                # {"role": "assistant", "content": error_message}
                AIMessage(content=error_message)
            ],
            "run_id": run_id  # 保留或生成run_id
        }
    except Exception as e:
        # 捕获其他异常
        error_message = f"计划生成过程中发生错误: {str(e)}"
        return {
            "plan": [],
            "current_step": 0,
            "messages": messages + [
                # {"role": "assistant", "content": error_message}
                AIMessage(content=error_message)
            ],
        }

# ====================================
# 2. Executor（决定当前执行哪个步骤）
# ====================================

async def executor(state: State):
    """
    Executor：
    - 判断计划是否结束
    - 若未结束：使用 bind_tools 的 LLM 为当前工具生成 tool_call
    - 交给 ToolNode 执行
    """

    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    messages = state.get("messages", [])

    # ---------- 空计划 ----------
    if not plan:
        return {
            "next": "summarizer",
            "plan": plan,
            "messages": messages,
        }
        
    success = True
    result_json = None

    # -----------------------------
    # 1. 解析上一个工具执行结果
    # -----------------------------
    if messages and isinstance(messages[-1], ToolMessage):
        last_tool_msg = messages[-1]
        try:
            raw_text = last_tool_msg.content[0]["text"]
            result_json = json.loads(raw_text)
            success = result_json.get("status") == "success"
        except Exception as e:
            logger.error(f"Executor: 解析工具结果失败: {e}")
            success = False

    # -----------------------------
    # 2. 失败处理 + 重试
    # -----------------------------
    retry_count = state.get("retry_count", 0)
    max_retry = 1

    if not success:
        logger.error(
            f"Executor: 工具执行失败: "
            f"{result_json.get('message') if isinstance(result_json, dict) else ''}"
        )

        if retry_count < max_retry:
            logger.info(
                f"Executor: 工具执行失败，正在重试 "
                f"({retry_count + 1}/{max_retry})"
            )

            return {
                "messages": messages[:-1],   # 回滚 tool_call + tool_result
                "current_step": current_step,  # ⚠️ 不前进
                "plan": plan,
                "retry_count": retry_count + 1,
            }
    
    # 计划执行完，进入 checker
    if current_step >= len(plan):
        return {
            "current_step": current_step,
            "next": "check",
            "messages": messages,
        }

    # ---------- 当前步骤 ----------
    step = plan[current_step]
    tool_name = step["tool"]

    logger.info(
        f"Executor: Step {current_step + 1}/{len(plan)} → {tool_name}"
    )
    # 在此传递了文件信息
    files = state.get("files", [])
    file_context = ""
    if files:
        file_context = (
            "当前可用文件如下：\n" +
            "\n".join(
                f"- {f['name']} ({f['path']}, {f.get('type')})"
                for f in files
            )
        )
    # ---------- 让 LLM 生成 tool_call ----------
    llm_messages = messages + [
        SystemMessage(content=file_context),
        HumanMessage(
            content=f"""
            请调用工具 `{tool_name}`。
            要求：
            - 自动从上下文中提取所需参数
            - 不要输出任何解释文本
            """
        )
    ]

    if llm_with_tools is None:
        await initialize_llm_with_tools()

    response = await llm_with_tools.ainvoke(
        llm_messages,
        tool_choice={
            "type": "function",
            "function": {"name": tool_name},
        },
    )

    if not response.tool_calls:
        raise RuntimeError(
            f"Executor: LLM 未生成 tool_call,{tool_name}"
        )
    logger.info(f"Executor run_id: {state['run_id']}")
    # 给用于生成tool_call的AIMessage加上 run_id
    response.additional_kwargs["run_id"] = state["run_id"]
    logger.info(state["run_id"])
    # response是AIMessage，包含tool_calls
    return {
        "messages": messages + [response],
        "current_step": current_step + 1,  # 递增步骤，避免无限循环
        "plan": plan,
        "run_id": state["run_id"],
        "next": "tools",   # ← 交给 ToolNode
        "retry_count": 0
    }

# ====================================
# 4. Checker（检查是否满足用户需求）
# ====================================
async def checker(state: State):
    """检查最终执行结果，使用LLM评估是否需要重规划"""
    # 获取最新的用户查询（从messages中提取最后一条用户消息）
    messages = state["messages"]
    # logger.info(f"Checker: 消息状态 - {messages}")
    user_query = None
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            user_query = m.content
            break

    # 2. 提取工具执行结果（仅本次对话）
    recent_tool_messages = get_recent_run_tool_messages(messages, state["run_id"])
    tool_results = extract_tool_results(recent_tool_messages)
    if not tool_results:
        final_result = ""
    else:
        final_result = json.dumps(tool_results, ensure_ascii=False, indent=2)
    logger.info(f"Checker: 提取到的 - {final_result}")

    conversation = "\n".join(
        f"{m.type}: {m.content}" for m in messages[:-1] 
        if m.type in ['human', 'ai'] and hasattr(m, 'content') and m.content.strip()  # 只包含human和ai类型且内容非空
    )

    logger.info("\nChecker: 评估执行结果是否满足需求")
    logger.info(f"Checker: 用户问题: {user_query}")
    logger.info(f"Checker: 执行结果: {final_result}")

    # 使用LLM评估执行结果是否满足用户需求
    # 使用从YAML加载的提示词
    system_prompt = checker_prompt_data['system']
    
    # 构建完整的对话历史作为上下文
    evaluation_prompt = [
        SystemMessage(content=system_prompt.format(
            user_query=user_query,
            final_result=final_result,
            conversation=conversation
        ))
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
                "plan": state.get("plan", []),  # 保留执行计划
                "messages": state.get("messages", [])  # 保留对话历史
            }
        else:
            # 检查通过，将进入summarizer节点生成最终回复
            if not evaluation_result["satisfies_needs"]:
                # 如果不满足需求但也不重新规划，预先设置一个基本回复
                state["reply"] = final_result
            return {
                "replan": False,
                "next": "end",
                "plan": state.get("plan", []),  # 保留执行计划
                "messages": state.get("messages", [])  # 保留对话历史
            }
    except Exception as e:
        logger.info(f"Checker: 评估过程出错: {str(e)}")
        # 出错时默认不重新规划
        return {
            "replan": False,
            "next": "end",
            "plan": state.get("plan", []),  # 保留执行计划
            "messages": state.get("messages", [])  # 保留对话历史
        }

# ====================================
# 5. Summarizer（生成最终回复）
# ====================================
async def summarizer(state: State, writer:StreamWriter):
    """使用LLM生成最终回复 - 使用StreamWriter进行流式输出"""
    # 获取最新的用户查询（从messages中提取最后一条用户消息）
    user_query = ""
    for msg in reversed(state["messages"]):
        # 使用属性访问而不是字典访问，因为msg是Message对象
        if hasattr(msg, 'role') and msg.role == "user":
            user_query = msg.content if hasattr(msg, 'content') else ""
            break

    messages = state["messages"]
    # 提取工具执行结果（仅本次对话）
    recent_tool_messages = get_recent_run_tool_messages(messages, state["run_id"])
    tool_results = extract_tool_results(recent_tool_messages)
    if not tool_results:
        final_result = ""
    else:
        final_result = json.dumps(tool_results, ensure_ascii=False, indent=2)

    logger.info("\nSummarizer: 生成最终回复（使用StreamWriter流式输出）")

    # 初始化累积回复
    accumulated_reply = ""
    final_messages = state["messages"].copy()  
    # logger.info(f"Summarizer:回复前messages - {final_messages}")
    conversation = "\n".join(
        f"{m.type}: {m.content}" for m in final_messages[:-1] 
        if m.type in ['human', 'ai'] and hasattr(m, 'content') and m.content.strip()  # 只包含human和ai类型且内容非空
    )
    logger.info(f"Summarizer: 对话历史 - {conversation}")
    # 使用从YAML加载的提示词
    system_prompt = summarizer_prompt_data['system']
    
    # 构建完整的对话历史作为上下文
    summary_prompt = [
        SystemMessage(content=system_prompt.format(
            user_query=user_query,
            final_result=final_result,
            conversation=conversation
        ))
    ]

    try:
        if hasattr(llm, "astream"):
            # 如果LLM支持异步流式调用
            async for chunk in llm.astream(summary_prompt):
                if hasattr(chunk, 'content') and chunk.content:
                    # 更新累积回复
                    accumulated_reply += chunk.content
                    
                    # 使用StreamWriter发送流式数据
                    writer({
                        "event_type": "custom_stream",
                        "messages": final_messages + [{"role": "assistant", "content": accumulated_reply}],
                        "reply": chunk.content,
                        "is_partial": True
                    })
    except Exception as e:
        logger.error(f"Summarizer: 生成回复时出错: {str(e)}")
        # 出错时使用执行结果作为回复
        accumulated_reply = final_result
        
        writer({
                "event_type": "custom_stream",
                "messages": final_messages + [{"role": "assistant", "content": accumulated_reply}],
                "reply": accumulated_reply,
                "is_partial": False
            })
    final_messages.append(AIMessage(content=accumulated_reply))
    logger.info(f"Summarizer: 最终messages - {final_messages}")
    # 返回最终状态，而不是使用yield
    return {
        "messages": final_messages,
        "reply": accumulated_reply,
        "plan": state.get("plan", [])
    }

# ================================
# 8. 构建 LangGraph
# ================================

graph = StateGraph(State)

# 添加节点
graph.add_node("planner", planner, aflow=True)  # Planner现在是异步节点
graph.add_node("executor", executor)  # 执行计划管理步骤
# graph.add_node("router", router, aflow=True)     # 执行工具处理输入输出（异步节点）
# 前面获取的mcp工具
tools = asyncio.run(get_tools_for_toolnode())
tool_node = ToolNode(tools)

graph.add_node("tools", tool_node)
graph.add_node("checker", checker, aflow=True)   # 检查结果是否满足要求（异步节点）
graph.add_node("summarizer", summarizer, aflow=True)  # 生成最终回复（异步节点）

# 设置入口点
graph.set_entry_point("planner")
# 连接节点
graph.add_edge("planner", "executor")  # Planner生成计划后交给Executor

# 添加条件边，根据executor返回的状态决定下一步
graph.add_conditional_edges(
    "executor",
    lambda x: x.get("next", "end"),
    {
        "tools": "tools",            # 🔑 新增：交给 ToolNode 执行
        "check": "checker",
        "summarizer": "summarizer",
        "end": END
    }
)
graph.add_edge("tools", "executor")
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
# graph_png = graph.get_graph().draw_mermaid_png()
# 编译图，支持流式输出
app_bindtools = graph.compile(name="bridge_bindtools")

logger.info("BridgeBindTools: 图编译完成，支持流式输出")

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