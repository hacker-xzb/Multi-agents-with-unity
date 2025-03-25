from fastapi import FastAPI, HTTPException
from typing import List, Dict
from pydantic import BaseModel
import logging
import re
import random  # 模拟动态响应

app = FastAPI()

logging.basicConfig(level=logging.DEBUG)

class TextRequest(BaseModel):
    text: str

class AiResponse(BaseModel):
    name: str          # AI名称（如"ms"）
    target_room: str   # 目标房间（如"Room_03"）
    response: str      # 响应信息

@app.post("/process")
async def process_text(request: TextRequest):
    try:
        text = request.text
        words = text.split()
        
        responses = []
        
        for word in words:
            # 新正则表达式：匹配格式为 "名称_目标房间"（如：ms_Room_03）
            match = re.match(r"([a-zA-Z0-9]+)_(.*)", word)
            if match:
                name, target_room = match.groups()
                response = generate_response(name, target_room)
                responses.append({
                    "name": name.strip(),
                    "target_room": target_room.strip(),
                    "response": response
                })
            else:
                # 忽略不符合格式的指令
                logging.warning(f"无效指令格式: {word}")
        
        return {"responses": responses}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_response(name: str, target_room: str) -> str:
    """模拟 LLM 生成动态 response"""
    templates = [
        f"AI '{name}' 已开始前往 {target_room}，预计耗时 2 分钟。",
        f"收到！{name} 正在导航到 {target_room}，请稍候。",
        f"指令确认：{name} 将移动到 {target_room}，当前路径无障碍。",
        f"AI '{name}' 已收到目标 {target_room}，准备执行任务。",
        f"警报！{name} 检测到 {target_room} 有异常，立即前往处理。"
    ]
    return random.choice(templates)  # 随机选择模板（实际应调用 LLM）