from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import re
import os
import dashscope
from dashscope import Generation
from typing import Dict, Any, List
import json

app = FastAPI()
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量存储背景故事
current_background = """
    - 星际飞船房间编号：
      - 健康室：Room_01

"""

#预处理正则化表达加快解析返回内容的速度
SECTION_REGEX = re.compile(r'\[(\w+)\](.*?)\[/\1\]', re.DOTALL)
# 角色数据库文件路径
CHARACTERS_DB_FILE = "characters_db.json"
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "sk-af527af40b3c4d82bbebe333a99601cc")

# 加载角色数据库（初始化时加载）
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

# 初始化角色数据库（全局变量）
characters_db = load_characters()

class TextRequest(BaseModel):
    text: str

class GameBackgroundRequest(BaseModel):
    background_story: str  # 新增专用背景故事接收模型

class AiResponse(BaseModel):
    name: str
    target_room: str
    response: str

@app.post("/set_background")
def set_background(request: GameBackgroundRequest):
    """接收并存储背景故事的新接口"""
    global current_background
    try:
        # 基础验证
        if not request.background_story.strip():
            raise HTTPException(status_code=400, detail="背景故事不能为空")
        
        # 更新全局背景
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
        """处理主要逻辑的原有接口"""
        try:
            user_input = request.text.strip()
            
            # 生成动态提示（现在使用全局背景）
            prompt = generate_prompt_template(user_input, current_background, characters_db)
            
            # 调用LLM处理任务（包含量化和反思）
            response = Generation.call(
                model=Generation.Models.qwen_max,
                prompt=prompt,
                stream=False
            )
            generated_text = response.output.text.strip()
            print(generated_text)
            # 提取结构化数据
            analysis = extract_analysis(generated_text)
            actions = extract_actions(generated_text)
            reflection = analysis.get("reflection", "")
            
            formatted_analysis = {
                # 原始文本字段
                "analysis": analysis.get("analysis", ""),
                "conversation": analysis.get("conversation", ""),
                "conflict": analysis.get("conflict", ""),
                "conflict_resolution": analysis.get("conflict_resolution", ""),
                "reflection": analysis.get("reflection", ""),
                "time_estimate": analysis.get("time_estimate", ""),
                "progress_events": analysis.get("progress_events", ""),
                
                # 结构化数据转为字符串
                "total_minutes": str(analysis.get("total_minutes", 0)),
                "progress_json": json.dumps(analysis.get("progress_events", []))
                }

            # 根据反思更新角色描述
            if reflection:
                update_characters(reflection)
            print(f"analysis: {analysis}")
            print(f"actions: {actions}")
            return {
                "analysis": analysis,
                "actions": actions
            }
        
        except json.JSONDecodeError as e:
         # 直接返回错误信息到客户端
            error_msg = f"JSON 解析失败: {str(e)}。原始 actions_str 内容:\n{actions_str}"
            logging.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)


    # 以下原有工具函数保持不变
def generate_prompt_template(user_input, background, characters):
        formatted_chars = "\n      - ".join([
            f"{k}（{v['role']}）：{v['description']}" 
            for k, v in characters.items()
        ])
        
        return f"""
        你是一个智能调度系统，请严格按照以下规则处理请求(输出格式一定要严格执行)：

        背景如下：
        {background}

        角色数据库：
        - {formatted_chars}

        任务描述：{user_input}

        输出要求：
        1. 在[ANALYSIS]部分用自然语言描述各角色对于任务都是如何思考的(思考解决方案以及可能出现的问题)
        2. 在[TIME_ESTIMATE]部分根据角色描述量化时间(最后一定要加上总计xx分钟)
        3. 在[REFLECTION]部分用自然语言建议角色能力改进
        4.在[CONVERSATION]根据每个人的人物设定和你给出的分析内容输出人物间的对话（要符合人物设定的主观判断）
        5.在[CONFLICT]根据角色对问题的分析以及对话输出一些对解决问题的冲突（比如方案冲突优先级冲突）
        6.在[CONFLICT_RESOLUTION]说明冲突是如何解决的风格需要偏向旁白的感觉
        7.在[PROGRESS_EVENTS]按进度百分比列出关键事件（例如：10%：初步方案确定(什么样的初步方案)；30%：遇到技术障碍等(什么技术障碍)）
        总之这些关键事件要比较详细能够模拟出真实感
        8. [ACTIONS]部分输出严格符合格式的JSON数组：
        - 每个对象必须包含 name、target_room、response 三个字段
        - 使用英文双引号（"..."）包裹键名和字符串值
        - 例如：[{{"name": "ms", "target_room": "Room_02", "response": "正在检查空调系统..."}}]
        # 重要规则(确保遵守以上规则的同时执行这些规则)：
        1. 每个部分必须用【[SECTION_NAME]】和【[/SECTION_NAME]】包裹
        2. 不得添加任何额外文本或格式（如Markdown）
        3. 如果某个部分无内容，仍需保留空的标记对
        4. 禁止使用中文标点符号（如“”、——）

        输出格式示例：
        [ANALYSIS]
        维修工程师ms能快速诊断问题，但面对复杂故障时需要更多时间...
        [/ANALYSIS]
        
        [CONVERSATION]
        name：conversation。
        name2:conversation.....
        [/CONVERSATION]
        
        [CONFLICT]
        发生了什么冲突
        [/CONFLICT]
        
        [CONFLICT_RESOLUTION]
        如何解决冲突
        [/CONFLICT_RESOLUTION]

        [REFLECTION]
        建议增加安全员的高压环境适应性训练，提升其决策效率...
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

def extract_analysis(text: str) -> Dict[str, str]:
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
        # 使用非贪婪匹配，允许跨行内容
        pattern = rf'\[{key}\](.*?)\[/{key}\]'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            content = match.group(1).strip()
            # 清理可能残留的标记符号
            content = re.sub(r'^\s*\[\+?\]\s*', '', content, flags=re.MULTILINE)
            result[name] = content
        else:
            # 尝试模糊匹配（处理可能的格式错误）
            fuzzy_pattern = rf'{key}[\s\S]*?(.*?)(?=\n\[|$)'
            fuzzy_match = re.search(fuzzy_pattern, text, re.IGNORECASE)
            result[name] = fuzzy_match.group(1).strip() if fuzzy_match else f"未找到{key}内容"

    # 特殊处理时间估算的量化信息
    if 'time_estimate' in result:
        time_text = result['time_estimate']
        # 提取总时间
        total_match = re.search(r'总计\s*(\d+)\s*分钟', time_text)
        if total_match:
            result['total_minutes'] = int(total_match.group(1))
    
    # 处理进度事件的格式化
    if 'progress_events' in result:
        events = []
        for line in result['progress_events'].split('\n'):
            if '%' in line:
                percent, _, desc = line.partition('：')
                events.append({
                    'percent': int(percent.replace('%', '').strip()),
                    'description': desc.strip()
                })
        result['progress_events'] = events
    
    return result

def extract_actions(text: str) -> List[Dict[str, Any]]:
    start = text.find("[ACTIONS]") + len("[ACTIONS]")
    end = text.find("[/ACTIONS]")
    actions_str = text[start:end].strip()
    
    # 调试输出：打印原始内容
    print(f"原始 actions_str 内容:\n{actions_str}")
    
    # 替换中文标点为英文标点
    actions_str = (
        actions_str
        .replace("“", '"')  # 替换中文双引号（左）
        .replace("”", '"')  # 替换中文双引号（右）
        .replace("‘", "'")  # 替换中文单引号（左）
        .replace("’", "'")  # 替换中文单引号（右）
        .replace("：", ":")  # 替换中文冒号
        .replace("，", ",")  # 替换中文逗号
        .replace("；", ";")  # 替换中文分号
        .replace("'", '"')  # 替换单引号为双引号（防止 LLM 使用单引号）
                    )
    
    # 尝试解析并捕获错误
    try:
        return json.loads(actions_str)
    except json.JSONDecodeError as e:
        print(f"JSON 解析失败: {e}")
        print(f"错误位置: 行 {e.lineno}, 列 {e.pos}")
        # 尝试修复后再次解析（可选）
        try:
            # 添加外层方括号（如果缺失）
            if not actions_str.startswith("["):
                actions_str = f"[{actions_str}]"
            if not actions_str.endswith("]"):
                actions_str = f"{actions_str}]"
            return json.loads(actions_str)
        except Exception as e2:
            print(f"修复后仍失败: {e2}")
            raise

@app.post("/reload_characters", response_model=str)
def reload_characters():
    """手动触发角色数据库重新加载"""
    global characters_db
    try:
        characters_db = load_characters()
        return "角色数据库已重新加载！"
    except Exception as e:
        logger.error(f"重新加载角色数据库失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def save_characters(db):
    try:
        with open(CHARACTERS_DB_FILE, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
        logger.info(f"角色数据库已保存到 {CHARACTERS_DB_FILE}")
    except Exception as e:
        logger.error(f"保存角色数据库失败: {str(e)}", exc_info=True)

def update_characters(reflection_text):
    global characters_db
    try:
        llm_prompt = f"""
        根据以下反思内容，生成每个角色的优化后描述：
        {reflection_text}
        
        输出格式：
        - ms：优化后的完整描述
        - ms2：优化后的完整描述
        - ms3：优化后的完整描述
        """
        
        response = Generation.call(
            model=Generation.Models.qwen_max,
            prompt=llm_prompt,
            stream=False
        )
        updated_text = response.output.text.strip()
        logger.debug(f"LLM返回文本:\n{updated_text}")
        
        for line in updated_text.split('\n'):
            if line.startswith('- '):
                try:
                    line = line[2:]  # 去除开头的 '- '
                    char_name_part, new_desc = line.split('：', 1)
                    char_name = char_name_part.strip().replace('*', '')
                    new_desc = new_desc.strip()
                    
                    if char_name in characters_db:
                        characters_db[char_name]['description'] = new_desc
                        logger.info(f"更新角色 {char_name} 的描述：{new_desc}")
                    else:
                        logger.warning(f"无效的角色名称: {char_name}")
                except ValueError:
                    logger.warning(f"无效的行格式: {line}")
        
        save_characters(characters_db)
        
    except Exception as e:
        logger.error(f"更新角色失败: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)