from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import re
import os
import json
from typing import Dict, Any, List
import dashscope
from dashscope import Generation

app = FastAPI()

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量存储背景故事
current_background = """
    - 星际飞船房间编号：
      - 健康室：Room_01
"""

# 正则表达式，用于提取大模型返回各部分内容
SECTION_REGEX = re.compile(r'\[(\w+)\](.*?)\[/\1\]', re.DOTALL)
# 角色数据库文件路径
CHARACTERS_DB_FILE = "characters_db.json"

# 配置 dashscope API Key（请确保环境变量中已设置或直接填入有效的 API Key）
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "sk-your-real-api-key")

# 加载角色数据库；如文件不存在则使用默认值
def load_characters() -> dict:
    try:
        with open(CHARACTERS_DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"角色数据库文件 {CHARACTERS_DB_FILE} 未找到，使用默认初始化值")
        return {
            "ms": {
                "role": "维修工程师",
                "description": "具有丰富的机械维修经验，擅长快速诊断问题，性格沉稳但偶尔急躁。维修速度较快，通常能在5-10分钟内解决常见故障，但面对复杂问题时需要更多时间。"
            },
            "ms2": {
                "role": "安全员",
                "description": "训练有素的应急专家，擅长高效组织疏散，沟通能力强。能够以每分钟疏散10-15人的速度执行任务，但在高压环境下容易过度谨慎。"
            },
            "ms3": {
                "role": "厨师",
                "description": "拥有创意烹饪能力，擅长开发新菜品，但烹饪速度较慢。常规餐点需要15-20分钟准备，但在紧急情况下能优先处理基础餐食。性格内向，但对食材选择有强迫症。"
            }
        }
    except Exception as e:
        logger.error(f"加载角色数据库失败: {str(e)}", exc_info=True)
        return {}

# 初始化角色数据库为全局变量
characters_db = load_characters()

# 定义请求和响应数据模型
class TextRequest(BaseModel):
    text: str

class GameBackgroundRequest(BaseModel):
    background_story: str  # 用于更新背景故事

class AiResponse(BaseModel):
    name: str
    target_room: str
    response: str

@app.post("/set_background")
def set_background(request: GameBackgroundRequest):
    """
    接收并更新背景故事。
    """
    global current_background
    try:
        if not request.background_story.strip():
            raise HTTPException(status_code=400, detail="背景故事不能为空")
        current_background = request.background_story
        logger.info(f"已更新背景故事，长度：{len(current_background)}字符")
        return {
            "status": "success",
            "message": "背景故事已更新",
            "received_length": len(current_background)
        }
    except Exception as e:
        logger.error(f"背景故事更新失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process", response_model=Dict[str, Any])
def process_text(request: TextRequest):
    try:
        user_input = request.text.strip()
        prompt = generate_prompt_template(user_input, current_background, characters_db)
        
        # 调用大模型生成响应（请确保 dashscope 接口正确返回文本格式）
        response = Generation.call(
            model=Generation.Models.qwen_max,
            prompt=prompt,
            stream=False
        )
        generated_text = response.output.text.strip()
        logger.info("大模型返回文本成功")
        logger.debug(f"生成的文本：{generated_text}")
        
        # 提取大模型返回的结构化数据
        analysis = extract_analysis(generated_text)
        actions = extract_actions(generated_text)
        reflection = analysis.get("reflection", "")
        
        # 若存在反思内容，更新角色描述并调用 RLHF 更新接口
        if reflection:
            update_characters(reflection)
            reinforcement_learning_update(analysis)
        
        return {
            "analysis": analysis,
            "actions": actions
        }
    except json.JSONDecodeError as e:
        error_msg = f"JSON 解析失败: {str(e)}。"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as ex:
        logger.error(f"处理文本失败: {str(ex)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(ex))

def generate_prompt_template(user_input: str, background: str, characters: dict) -> str:
    """
    根据用户输入、背景故事和角色信息生成大模型提示，
    提示中要求输出详细的思维链（Chain-of-Thought）及其它结构化内容。
    """
    formatted_chars = "\n      - ".join([
        f"{k}（{v['role']}）：{v['description']}" for k, v in characters.items()
    ])
    
    return f"""
    你是一个智能调度系统，请严格按照以下规则处理请求(输出格式一定要严格执行)：
    
    背景如下：
    {background}
    
    角色数据库：
    - {formatted_chars}
    
    任务描述：{user_input}
    
    输出要求：
    1. 在[ANALYSIS]部分输出详细的思维链（Chain-of-Thought），描述各角色分步思考问题的过程，
       包括行为决策偏好和改进建议。
    2. 在[TIME_ESTIMATE]部分量化角色描述所需时间（最后必须写明总计xx分钟）。
    3. 在[REFLECTION]部分输出自然语言的改进建议，用于后续 RLHF 反思训练。
    4. 在[CONVERSATION]部分输出各角色间的对话，须符合角色设定。
    5. 在[CONFLICT]部分描述角色间因决策偏好或任务理解产生的冲突。
    6. 在[CONFLICT_RESOLUTION]部分说明如何通过合理决策解决冲突（风格偏旁白）。
    7. 在[PROGRESS_EVENTS]部分按进度百分比列出关键事件。
    8. 在[ACTIONS]部分输出严格符合格式的 JSON 数组，每个对象包含 name、target_room、response 三个字段，
       使用英文双引号包裹键名和值，例如：[{{"name": "ms", "target_room": "Room_02", "response": "正在检查空调系统..."}}]
    
    重要规则：
    1. 每个部分必须用【[SECTION_NAME]】和【[/SECTION_NAME]】包裹。
    2. 不得添加任何额外文本或格式（如 Markdown）。
    3. 如果某个部分无内容，仍需保留空标记对。
    4. 禁止使用中文标点符号。
    
    输出格式示例：
    [ANALYSIS]
    维修工程师ms分步分析问题：第一步检查设备，第二步定位故障原因……
    [/ANALYSIS]
    
    [CONVERSATION]
    ……
    [/CONVERSATION]
    
    [CONFLICT]
    ……
    [/CONFLICT]
    
    [CONFLICT_RESOLUTION]
    ……
    [/CONFLICT_RESOLUTION]
    
    [REFLECTION]
    建议增加安全员的高压环境适应性训练……
    [/REFLECTION]
    
    [PROGRESS_EVENTS]
    10%：初步方案确定
    30%：遇到技术障碍
    50%：冲突解决
    80%：完成关键测试
    100%：任务完成
    [/PROGRESS_EVENTS]
    
    [ACTIONS]
    [{{"name": "ms", "target_room": "Room_02", "response": "正在检查空调系统..."}}]
    [/ACTIONS]
    """

def extract_analysis(text: str) -> Dict[str, Any]:
    """
    从大模型返回的文本中解析各个区域的内容：
    包括 ANALYSIS、CONVERSATION、CONFLICT、CONFLICT_RESOLUTION、REFLECTION、TIME_ESTIMATE、PROGRESS_EVENTS。
    """
    sections = {
        "ANALYSIS": "analysis",
        "CONVERSATION": "conversation",
        "CONFLICT": "conflict",
        "CONFLICT_RESOLUTION": "conflict_resolution",
        "REFLECTION": "reflection",
        "TIME_ESTIMATE": "time_estimate",
        "PROGRESS_EVENTS": "progress_events"
    }
    
    result = {}
    for key, name in sections.items():
        pattern = rf'\[{key}\](.*?)\[/{key}\]'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            # 清理行首可能多余的标记
            content = re.sub(r'^\s*\[\+?\]\s*', '', content, flags=re.MULTILINE)
            result[name] = content
        else:
            # 若找不到则返回空字符串
            result[name] = ""
    
    # 从 TIME_ESTIMATE 中解析总时间（单位：分钟）
    if result.get("time_estimate"):
        total_match = re.search(r'总计\s*(\d+)\s*分钟', result["time_estimate"])
        if total_match:
            result["total_minutes"] = int(total_match.group(1))
    
    # 解析 PROGRESS_EVENTS 部分为字典列表
    if result.get("progress_events"):
        events = []
        for line in result["progress_events"].split('\n'):
            if '%' in line:
                percent, _, desc = line.partition('：')
                try:
                    events.append({
                        'percent': int(percent.replace('%', '').strip()),
                        'description': desc.strip()
                    })
                except ValueError:
                    continue
        result["progress_events"] = events
    else:
        result["progress_events"] = []
    
    return result

def extract_actions(text: str) -> List[Dict[str, Any]]:
    """
    从大模型返回文本中提取 [ACTIONS] 部分，并转换为 JSON 数组。
    包含中文标点替换，确保 JSON 格式正确。
    """
    start = text.find("[ACTIONS]") + len("[ACTIONS]")
    end = text.find("[/ACTIONS]")
    actions_str = text[start:end].strip()
    logger.debug(f"原始 actions_str 内容:\n{actions_str}")
    
    # 替换中文标点为英文标点，确保 JSON 格式正确
    actions_str = (
        actions_str
        .replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("：", ":")
        .replace("，", ",")
        .replace("；", ";")
        .replace("'", '"')
    )
    try:
        return json.loads(actions_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析失败: {e}，尝试补全外层中括号")
        try:
            if not actions_str.startswith("["):
                actions_str = f"[{actions_str}]"
            if not actions_str.endswith("]"):
                actions_str = f"{actions_str}]"
            return json.loads(actions_str)
        except Exception as e2:
            logger.error(f"修复后仍失败: {e2}")
            raise

@app.post("/reload_characters", response_model=str)
def reload_characters():
    """
    接口：手动重新加载角色数据库。
    """
    global characters_db
    try:
        characters_db = load_characters()
        return "角色数据库已重新加载！"
    except Exception as e:
        logger.error(f"重新加载角色数据库失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def save_characters(db):
    """
    将当前角色数据库保存到文件中。
    """
    try:
        with open(CHARACTERS_DB_FILE, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
        logger.info(f"角色数据库已保存到 {CHARACTERS_DB_FILE}")
    except Exception as e:
        logger.error(f"保存角色数据库失败: {str(e)}", exc_info=True)

def reinforcement_learning_update(analysis: Dict[str, Any]):
    """
    调用 RLHF 接口，利用大模型生成的详细分析（包含思维链及对话）进行行为偏好更新和反思训练，
    并将 RLHF 返回的建议更新到角色数据库中。
    
    注意：此处假设存在一个真实的 RLHF 服务接口（例如 REST API），你需要根据实际情况修改 RLHF 调用部分。
    """
    try:
        # 构造 RLHF 请求数据（此处直接使用分析内容作为训练参数，实际可扩展为更多参数）
        rlhf_data = {
            "analysis": analysis,
            "characters": characters_db
        }
        # 模拟 RLHF 调用，这里直接使用大模型接口替代，
        # 如果有真实服务，请替换为例如 requests.post("http://rlhf-service/update", json=rlhf_data)
        rl_prompt = f"""
        根据以下分析内容，针对各角色的行为决策偏好进行 RLHF 训练，请输出每个角色的更新建议：
        分析内容：{json.dumps(analysis, ensure_ascii=False)}
        
        输出格式：
        - ms：更新建议
        - ms2：更新建议
        - ms3：更新建议
        """
        response = Generation.call(
            model=Generation.Models.qwen_max,
            prompt=rl_prompt,
            stream=False
        )
        rlhf_result = response.output.text.strip()
        logger.info("RLHF 更新调用成功")
        logger.debug(f"RLHF 返回结果：{rlhf_result}")

        # 解析 RLHF 返回结果并更新角色数据库
        for line in rlhf_result.split('\n'):
            if line.startswith('- '):
                try:
                    # 去除前缀“- ”
                    line = line[2:]
                    # 根据冒号分割角色名称和新描述（注意使用中文冒号）
                    char_name_part, new_desc = line.split('：', 1)
                    char_name = char_name_part.strip().replace('*', '')
                    new_desc = new_desc.strip()
                    if char_name in characters_db:
                        characters_db[char_name]['description'] = new_desc
                        logger.info(f"RLHF 更新角色 {char_name} 的描述：{new_desc}")
                    else:
                        logger.warning(f"RLHF 返回的无效角色名称: {char_name}")
                except ValueError:
                    logger.warning(f"无法解析 RLHF 更新行：{line}")
        # 保存更新后的角色数据库到文件中
        save_characters(characters_db)
    except Exception as e:
        logger.error(f"RLHF 更新失败: {str(e)}", exc_info=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
