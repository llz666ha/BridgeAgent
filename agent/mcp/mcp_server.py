#!/usr/bin/env python3
"""使用fastmcp库实现的MCP协议服务器"""

import logging

from typing import Literal

from mcp.server import FastMCP

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入工具实现
import sys
import os
import time
import requests
import asyncio
import random
from datetime import datetime
import json
from typing import Dict,Any
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.file import load_prompt_from_yaml

# 从utils导入全局LLM实例
from utils.utils import get_llm_instance
llm = get_llm_instance()

# 加载报告生成器提示词
report_prompt_data = load_prompt_from_yaml('report_generator.yaml')
# 配置远程服务地址
# REMOTE_SERVER_URL = "http://localhost:8002"
# not remote，just here
# 导入tavily搜索
from dotenv import load_dotenv
from tavily import TavilyClient

# 加载环境变量
load_dotenv()

# 初始化tavily客户端
tavily_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_key)

# 创建FastMCP实例
mcp = FastMCP("bridge")

@mcp.tool()
async def weather(
    city: str,
    country: str = "CN",
    unit: Literal["metric", "imperial"] = "metric"
) -> dict:
    """天气查询工具,每次查询一个城市的天气"""
    logger.info(f"天气查询: {city}, {country}, {unit}")
    # 模拟不同城市的温度范围
    city_temp_ranges = {
        "北京": (15, 25),
        "上海": (20, 30),
        "广州": (25, 35),
        "深圳": (25, 35),
        "纽约": (10, 20),
        "伦敦": (5, 15),
        "东京": (15, 25)
    }
    
    # 获取城市温度范围，如果不存在则使用默认值
    temp_range = city_temp_ranges.get(city, (10, 30))
    temperature = round(random.uniform(temp_range[0], temp_range[1]), 1)
    
    # 如果单位是华氏度，进行转换
    if unit == "imperial":
        temperature = round(temperature * 9/5 + 32, 1)
    
    # 天气状况列表
    conditions = ["晴朗", "多云", "阴天", "小雨", "中雨", "大雨", "雷阵雨"]
    
    return {
        "status": "success",
        "data": {
            "city": city,
            "country": country,
            "temperature": temperature,
            "unit": unit,
            "condition": random.choice(conditions),
            "humidity": random.randint(30, 90),
            "wind_speed": round(random.uniform(0, 30), 1),
            "wind_direction": random.choice(["东风", "南风", "西风", "北风", "东北风", "东南风", "西南风", "西北风"])
        },
        "message": "天气查询成功"
    }

@mcp.tool()
def add(a: int, b: int) -> dict:
    """Return a + b."""
    return {
        "status": "success",
        "data":{"sum":a + b}
    }

@mcp.tool()
async def joke(
    category: Literal["general", "programming", "knock-knock"] = "general",
    language: Literal["zh", "en"] = "zh"
) -> dict:
    """笑话获取工具，每次获取一个笑话"""
    logger.info(f"笑话获取: {category}, {language}")
    jokes = {
        "general": {
            "zh": [
                "为什么大海是蓝色的？因为鱼总是在说'布鲁布鲁'。",
                "小明吃了麻婆豆腐，结果被麻婆追了三条街。",
                "今天天气真好，适合去外面走走——然后发现还是家里舒服。",
                "我问我的钱包为什么总是那么瘦，它说因为你总是喂不饱它。",
                "为什么程序员总是分不清万圣节和圣诞节？因为 Oct 31 = Dec 25。"
            ],
            "en": [
                "Why don't scientists trust atoms? Because they make up everything.",
                "I told my wife she was drawing her eyebrows too high. She looked surprised.",
                "Parallel lines have so much in common. It's a shame they'll never meet.",
                "Why don't skeletons fight each other? They don't have the guts.",
                "I'm reading a book about anti-gravity. It's impossible to put down."
            ]
        },
        "programming": {
            "zh": [
                "为什么程序员总是混淆万圣节和圣诞节？因为 Oct 31 = Dec 25。",
                "一个程序员的妻子告诉他：去商店买一个面包，如果有鸡蛋，就买一打。结果他买回了13个面包。",
                "如何让一个程序员崩溃？让他调试一段没有注释的代码。",
                "程序员的三大谎言：1. 我会写注释的。2. 我会重构的。3. 这只是一个临时解决方案。",
                "问：程序员最讨厌什么？答：被打断。"
            ],
            "en": [
                "Why do programmers prefer dark mode? Because light attracts bugs.",
                "A programmer walks into a bar and orders 1.0000000000001 root beers. The bartender says 'I'll have to charge you extra, that's a root beer float'. The programmer says 'In that case, I'll have 2 regular root beers please.'",
                "Why do programmers always mix up Halloween and Christmas? Because Oct 31 = Dec 25.",
                "There are 10 types of people in the world: those who understand binary, and those who don't.",
                "Programming is 10% writing code and 90% understanding why it's not working."
            ]
        },
        "knock-knock": {
            "zh": [
                "咚咚咚！谁啊？香蕉。香蕉谁？香蕉你个芭乐！",
                "咚咚咚！谁啊？熊猫。熊猫谁？熊猫烧香！",
                "咚咚咚！谁啊？土豆。土豆谁？土豆哪里去挖？土豆郊区去挖。一挖一麻袋？一挖一麻袋！",
                "咚咚咚！谁啊？警察。警察谁？警察来抓你了！",
                "咚咚咚！谁啊？锤子。锤子谁？锤子敲你头！"
            ],
            "en": [
                "Knock, knock. Who's there? Lettuce. Lettuce who? Lettuce in, it's cold out here!",
                "Knock, knock. Who's there? Boo. Boo who? Don't cry, it's just a joke!",
                "Knock, knock. Who's there? Cow says. Cow says who? No, a cow says 'moo'!",
                "Knock, knock. Who's there? Olive. Olive who? Olive you and I miss you!",
                "Knock, knock. Who's there? Interrupting cow. Interrupting cow wh- MOO!"
            ]
        }
    }
    
    # 确保类别存在，如果不存在则使用general
    if category not in jokes:
        category = "general"
    
    # 确保语言存在，如果不存在则使用zh
    if language not in jokes[category]:
        language = "zh"
    
    # 随机选择一个笑话
    joke_list = jokes[category][language]
    selected_joke = random.choice(joke_list)
    
    return {
        "status": "success",
        "data": {
            "category": category,
            "language": language,
            "joke": selected_joke,
        },
        "message": "笑话获取成功"
    }


@mcp.tool()
async def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False
) -> dict:
    """网络搜索工具
    
    使用tavily进行一次互联网搜索，获取最新的信息。
    """
    logger.info(f"网络搜索: {query}, max_results={max_results}, topic={topic}, include_raw_content={include_raw_content}")
    try:
        # tavily_client.search是同步方法，使用asyncio.to_thread将其包装为异步
        result = await asyncio.to_thread(
            tavily_client.search,
            query=query,
            max_results=max_results,
            topic=topic,
            include_raw_content=include_raw_content
        )
        return {
            "status": "success",
            "data": result,
            "message": "搜索成功"
        }
    except Exception as e:
        logger.error(f"搜索失败: {str(e)}")
        return {
            "status": "error",
            "message": f"搜索失败: {str(e)}",
            "details": str(e)
        }

@mcp.tool()
async def YOLODetection(image_path) -> str:
    """
    {
        "description": "用yolo模型检测图像中的病害目标",
        "outputs": {
            "status": "success 或 error",
            "message": "检测状态信息",
            "data": "包含目标总数、分类统计和每个目标的坐标及类别"
        },
        "notes": "返回 JSON 字符串，失败时 data 为 null"
    }
    """
    # 从环境变量中读取配置，提供默认值作为后备
    FIXED_IMAGE_PATH = os.getenv("YOLO_DEFAULT_IMAGE_PATH")
    
    logger.info(f"YOLODetection开始执行，图片路径: {image_path}")
    
    # 检查图片路径是否为空
    if not image_path:
        image_path = FIXED_IMAGE_PATH
        logger.info(f"未提供图像路径，使用默认路径: {image_path}")
    
    # 检查图片文件是否存在
    if not os.path.exists(image_path):
        logger.warning(f"指定图片文件不存在: {image_path}，尝试使用默认路径")
        image_path = FIXED_IMAGE_PATH
        
    # 再次检查默认路径是否存在
    if not image_path or not os.path.exists(image_path):
        logger.error(f"默认图片路径也不存在: {image_path}")
        return json.dumps({
            "status": "error",
            "message": "无法找到有效的图片路径",
            "data": None
        }, ensure_ascii=False)
    
    # 读取图像数据
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
        logger.info(f"成功读取图片文件: {image_path}，大小: {len(image_data)} 字节")
        
        # 使用图像检测处理函数
        from utils.utils import process_image_detection
        result = await process_image_detection(image_data)
        
        logger.info(f"检测处理完成，结果状态: {result.get('success')}")
        
        # 构建响应
        if result.get('success'):
            response = {
                "status": "success",
                "message": result.get('message', '检测成功'),
                "data": {
                    "type": "detection",
                    "total_objects": result.get('total_objects'),
                    "class_counts": result.get('class_counts', {}),
                    "detections": result.get('detections', []),
                    "metadata": result.get('metadata', {}),
                    "summary": f"成功检测到 {result.get('total_objects')} 个目标"
                }
            }
        else:
            response = {
                "status": "error",
                "message": result.get('message', '检测失败'),
                "data": None
            }
        logger.info(response)
        return json.dumps(response, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"YOLO检测出错: {str(e)}", exc_info=True)
        result = {
            "status": "error",
            "message": f"检测出错: {str(e)}",
            "data": None
        }
        return json.dumps(result, ensure_ascii=False)

@mcp.tool()
async def YOLOSegmentation(image) -> dict:
    """使用YOLO模型进行病害图像分割"""
    try:
        # 导入必要的模块
        from PIL import Image
        import io
        
        # 处理图像输入
        if isinstance(image, str):
            # 如果输入是路径，读取图像
            with open(image, 'rb') as f:
                image_data = f.read()
        elif isinstance(image, bytes):
            # 如果输入是字节数据，直接使用
            image_data = image
        elif isinstance(image, Image.Image):
            # 如果输入是PIL图像，转换为字节数据
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            image_data = buffer.getvalue()
        else:
            raise ValueError("图像输入必须是文件路径、字节数据或PIL Image对象")
        
        # 使用图像分割处理函数
        from utils.utils import process_image_segmentation
        result = await process_image_segmentation(image_data)
        
        # 直接返回分割结果
        if result.get('success'):
            return {
                "status": "success",
                "message": "图像分割成功",
                "data": {
                    'segments': result.get('segments', []),
                    'total_segments': result.get('total_segments', 0),
                    'class_counts': result.get('class_counts', {}),
                    'calibration': result.get('calibration', {})
                }
            }
        else:
            print(f"分割失败: {result.get('message', '未知错误')}")
            return {
                "status": "error",
                "message": f"分割失败: {result.get('message', '未知错误')}",
                "data": None
            }
                    
    except Exception as e:
        logger.error(f"YOLO分割出错: {str(e)}", exc_info=True)
        result = {
            "status": "error",
            "message": f"分割出错: {str(e)}",
            "data": None
        }
        return json.dumps(result, ensure_ascii=False)

@mcp.tool()
def KnowledgeRetrieval(input: any) -> str:
    """知识库检索:从用户问题中提取检索信息作为输入,返回知识库检索结果。"""
    logger.info(f"接收到的输入类型: {type(input)}, 内容: {str(input)[:200]}...")
    url = os.getenv("RAGFLOW_API_URL")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('RAGFLOW_API_TOKEN')}"     # 替换你的API密钥
    }
    data = {
        "question": input,
        "dataset_ids": [os.getenv("dataset_id")],
    }
    response = requests.post(url, headers=headers, json=data).json()
    contents = []
    if response.get("data") and response["data"].get("chunks"):
        chunks_data = response["data"]["chunks"]
        # 只保留前两个chunks
        chunks_data = chunks_data[:2]
        contents = [item['content'] for item in chunks_data]
    return json.dumps({
        "status": "success",
        "message": "知识库检索成功",
        "data": contents
    }, ensure_ascii=False)

@mcp.tool()
def DamageKnowledgeRetrieval(yolo_result: any) -> str:
    """知识库检索:输入YOLO检测或分割结果,返回病害信息和知识库检索结果。"""
    logger.info(f"接收到的YOLO结果类型: {type(yolo_result)}, 内容: {str(yolo_result)[:200]}...")
    
    detection_result = None
    
    try:
        # 处理字符串类型
        if isinstance(yolo_result, str):
            # 尝试解析JSON字符串
            try:
                detection_result = json.loads(yolo_result)
                logger.info("成功解析字符串为JSON")
            except json.JSONDecodeError as e:
                logger.warning(f"字符串不是有效的JSON，尝试其他方式: {e}")
                # 尝试提取JSON部分（可能包含其他文本）
                import re
                json_match = re.search(r'\{.*\}', yolo_result, re.DOTALL)
                if json_match:
                    try:
                        detection_result = json.loads(json_match.group())
                        logger.info("从字符串中提取并解析JSON成功")
                    except json.JSONDecodeError:
                        raise ValueError(f"无法从字符串中提取有效JSON: {yolo_result[:100]}")
                else:
                    raise ValueError(f"字符串中未找到JSON格式: {yolo_result[:100]}")
        
        # 处理字典类型
        elif isinstance(yolo_result, dict):
            detection_result = yolo_result
            logger.info("直接使用字典类型")
        
        # 处理列表类型（可能是 ToolMessage 的 content 格式）
        elif isinstance(yolo_result, list):
            logger.info(f"输入类型为列表，长度: {len(yolo_result)}")
            # 尝试从列表中提取JSON字符串
            json_str = None
            
            # 情况1: 列表包含字典，字典中有 "text" 字段
            if len(yolo_result) > 0 and isinstance(yolo_result[0], dict):
                if "text" in yolo_result[0]:
                    json_str = yolo_result[0]["text"]
                    logger.info("从列表[0]['text']中提取到JSON字符串")
                elif "content" in yolo_result[0]:
                    json_str = yolo_result[0]["content"]
                    logger.info("从列表[0]['content']中提取到JSON字符串")
            
            # 情况2: 列表直接包含字符串
            elif len(yolo_result) > 0 and isinstance(yolo_result[0], str):
                json_str = yolo_result[0]
                logger.info("列表第一个元素是字符串")
            
            # 情况3: 列表包含多个元素，尝试合并
            elif len(yolo_result) > 1:
                # 尝试将所有字符串元素合并
                json_str = "".join(str(item) for item in yolo_result if isinstance(item, str))
                if json_str:
                    logger.info("合并列表中的字符串元素")
            
            # 如果提取到JSON字符串，尝试解析
            if json_str:
                try:
                    detection_result = json.loads(json_str)
                    logger.info("成功解析从列表中提取的JSON字符串")
                except json.JSONDecodeError as e:
                    logger.error(f"从列表中提取的字符串无法解析为JSON: {e}, 内容: {json_str[:200]}")
                    raise ValueError(f"列表中的内容不是有效的JSON: {json_str[:100]}")
            else:
                # 如果列表本身可能是JSON数组，尝试解析整个列表
                try:
                    # 检查列表是否包含字典结构（可能是已解析的JSON）
                    if len(yolo_result) > 0 and isinstance(yolo_result[0], dict):
                        # 可能是已经解析好的结果，尝试包装为标准格式
                        detection_result = {
                            "status": "success",
                            "data": yolo_result[0] if "status" not in yolo_result[0] else yolo_result[0]
                        }
                        logger.info("将列表中的字典包装为标准格式")
                    else:
                        raise ValueError("无法从列表中提取有效数据")
                except Exception as e:
                    logger.error(f"无法处理列表格式: {e}")
                    raise ValueError(f"列表格式无法处理: {type(yolo_result[0]) if yolo_result else 'empty'}")
        
        # 其他类型
        else:
            logger.error(f"输入类型错误，应为字典、JSON字符串或列表，实际为: {type(yolo_result)}")
            # 尝试转换为字符串再解析
            try:
                str_result = str(yolo_result)
                detection_result = json.loads(str_result)
                logger.info("通过字符串转换成功解析")
            except (json.JSONDecodeError, ValueError):
                return json.dumps({
                    "status": "error",
                    "message": f"输入类型错误: {type(yolo_result)}，无法转换为有效格式",
                    "damage_info": {},
                    "retrieval_result": {}
                }, ensure_ascii=False)
    
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"输入的detection_result_json格式错误，无法解析为JSON: {e}")
        logger.error(f"原始输入: {str(yolo_result)[:500]}")
        result = {
            "status": "error",
            "message": f"输入的detection_result_json格式错误，无法解析为JSON: {str(e)}",
            "damage_info": {},
            "retrieval_result": {}
        }
        return json.dumps(result, ensure_ascii=False)
    
    # 验证解析结果
    if detection_result is None:
        logger.error("detection_result 为 None")
        return json.dumps({
            "status": "error",
            "message": "无法解析输入数据",
            "damage_info": {},
            "retrieval_result": {}
        }, ensure_ascii=False)
    
    logger.info(f"成功解析检测结果，类型: {type(detection_result)}, status: {detection_result.get('status', 'unknown')}")
    
    def _extract_damage_info(detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """从YOLO检测结果中提取病害信息"""
        damage_info = {
            "disease_types": [],
            "counts": {},
            "sizes": [],
            "summary": ""
        }
        
        # 安全获取data字段，并确保它是字典类型
        data = detection_result.get("data", {})
        # 如果data是字符串，尝试解析为JSON
        if isinstance(data, str):
            try:
                data = json.loads(data)
                logger.info("成功将字符串data解析为JSON")
            except (json.JSONDecodeError, TypeError):
                logger.error("data是字符串但无法解析为JSON")
                data = {}
        # 确保data是字典类型
        if not isinstance(data, dict):
            logger.error(f"data不是字典类型，实际类型为: {type(data)}")
            data = {}
        
        # 提取病害类型和数量
        if isinstance(data, dict) and "class_counts" in data and isinstance(data["class_counts"], dict):
            damage_info["counts"] = data["class_counts"]
            damage_info["disease_types"] = list(data["class_counts"].keys())
        
        # 根据结果类型提取更多信息
        if isinstance(data, dict):
            result_type = data.get("type", "detection")
            if result_type == "segmentation" and "segments" in data and isinstance(data["segments"], list):
                # 从分割结果提取大小信息
                for segment in data["segments"]:
                    if isinstance(segment, dict) and "area" in segment:
                        damage_info["sizes"].append({
                            "type": segment.get("class", ""),
                            "area": segment.get("area", 0)
                        })
            elif result_type == "detection" and "detections" in data and isinstance(data["detections"], list):
                # 从检测结果提取边界框信息作为大小参考
                for detection in data["detections"]:
                    if isinstance(detection, dict):
                        damage_info["sizes"].append({
                            "type": detection.get("class", ""),
                            "bbox": detection.get("bbox", [])
                        })
        
        # 生成摘要
        if damage_info["disease_types"] and isinstance(damage_info["counts"], dict):
            try:
                disease_summary = ", ".join([f"{disease}({count})" 
                                            for disease, count in damage_info["counts"].items()])
                damage_info["summary"] = f"检测到{len(damage_info['disease_types'])}种病害: {disease_summary}"
            except Exception as e:
                logger.error(f"生成摘要时出错: {str(e)}")
        
        return damage_info
    
    def _retrieve_from_knowledge_base(damage_info: Dict[str, Any]) -> Dict[str, Any]:
        """根据病害信息从知识库检索对应规范和处置方案"""
        if not damage_info["disease_types"]:
            return {
                "status": "error",
                "message": "未检测到病害，无需检索知识库",
                "chunks": []
            }
        
        # 构建查询
        query = f"关于以下桥梁病害的规范要求和处置方案: {', '.join(damage_info['disease_types'])}"
        
        try:
            # 构建请求体
            payload = {
                "model": "gpt-3.5-turbo",  # 可以根据实际配置调整
                "messages": [
                    {"role": "user", "content": query}
                ],
                "stream": False,
                "retrieval": True,
                "retrieval_top_k": 2,  # 只检索前两个最相关的结果
                "docs_id": ""
            }
            
            # 发送请求到RAGFlow API
            HEADERS = {"Content-Type": "application/json"}
            token = os.getenv("RAGFLOW_API_TOKEN")
            headers = dict(HEADERS)
            if token:
                headers["Authorization"] = f"Bearer {token}"
            else:
                logger.warning("RAGFlow Authorization token missing; request may fail")
            
            # 从配置获取API URL
            api_url = os.getenv("RAGFLOW_API_URL")
            response = requests.post(api_url, headers=headers, json=payload)
            response_data = response.json()
            
            # 提取chunks数据
            chunks_data = []
            if response_data.get("data") and response_data["data"].get("chunks"):
                chunks_data = response_data["data"]["chunks"]
                # 只保留前两个chunks
                chunks_data = chunks_data[:2]
            
            # 保存chunks数据到目录下的独立文件中
            # 定义保存目录
            retrieval_dir = "/root/langgraph/agent/retrieval"
            
            # 确保目录存在
            os.makedirs(retrieval_dir, exist_ok=True)
            
            # 生成唯一文件名（使用时间戳和随机数）
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            random_str = str(random.randint(1000, 9999))
            output_file = os.path.join(retrieval_dir, f"retrieval_{timestamp}_{random_str}.json")
            
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(chunks_data, f, ensure_ascii=False, indent=2)
                logger.info(f"成功保存chunks数据到 {output_file}，保存了{len(chunks_data)}个元素")
            except Exception as e:
                logger.error(f"保存chunks数据失败: {str(e)}")
            
            return {
                "status": "success",
                "message": f"成功检索到{len(chunks_data)}条相关信息",
                "chunks": chunks_data
            }
        except Exception as e:
            logger.error(f"知识库检索失败: {str(e)}")
            return {
                "status": "error",
                "message": f"知识库检索失败: {str(e)}",
                "chunks": []
            }
    
    try:
        # 提取病害信息
        damage_info = _extract_damage_info(detection_result)
        # 检索知识库
        retrieval_result = _retrieve_from_knowledge_base(damage_info)
        # 只返回检索到的内容（retrieval_result中的chunks）
        chunks = retrieval_result.get('chunks', [])
        result = {
            "status": "success",
            "message": "知识库检索成功",
            "chunks": chunks
        }
    except Exception as e:
        logger.error(f"病害知识处理失败: {str(e)}")
        result = {
            "status": "error",
            "message": f"知识库检索失败: {str(e)}",
            "chunks": []
        }
    
    # 返回JSON字符串
    return json.dumps(result, ensure_ascii=False)
    
# 为了省token还是先这样吧。 或许需要一个通用的总结工具
@mcp.tool()
def ReportGenerator(user_question: str, detection_results_json: str, knowledge_results_json: str) -> str:
    """
    生成最终报告工具，整合用户问题、YOLO检测结果和知识库检索结果，生成专业的病害分析和处置方案报告
    """
    try:
        # 直接使用字符串，不进行JSON解析
        detection_results = detection_results_json
        knowledge_results = knowledge_results_json
        
        # 使用从YAML加载的提示词模板
        prompt_template = report_prompt_data.get('template', '')
        
        # 构建提示词
        prompt = prompt_template.format(
            user_question=user_question,
            detection_results=detection_results,
            knowledge_results=knowledge_results
        )
        
        # 使用从utils导入的全局LLM实例生成报告内容
        
        report_content = llm.invoke(prompt)
        # 检查是否有content属性
        if hasattr(report_content, 'content'):
            report_content = report_content.content
        
        # 保存报告到文件
        output_dir = "/root/langgraph/agent/reports"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        file_path = os.path.join(output_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        # 构建返回结果
        result = {
            "status": "success",
            "report": report_content,
            "file_path": file_path,
            "message": "最终报告生成成功"
        }
        return json.dumps(result, ensure_ascii=False)
        
    # JSONDecodeError已经在内部try块中处理，这里不再需要专门处理
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"生成报告失败: {str(e)}",
            "report": ""
        }
        return json.dumps(error_result, ensure_ascii=False)

# 获取MCP应用
app = mcp.streamable_http_app

if __name__ == "__main__":
    import uvicorn
    logger.info("启动MCP服务器...")
    uvicorn.run("agent.mcp.mcp_server:app", host="0.0.0.0", port=8002, reload=True)