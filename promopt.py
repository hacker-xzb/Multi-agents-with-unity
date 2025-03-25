def get_qwen_prompt(user_input: str) -> str:
    """生成通义千问的指令模板"""
    return f"""
    指令：根据用户输入解析多AI指令，并生成符合以下JSON格式的响应：
    用户输入："{user_input}"
    格式要求：
    1. 输出必须是一个包含"responses"键的JSON对象。
    2. "responses"是一个数组，每个元素包含：
       - "name": AI名称（如"ms"）
       - "target_room": 目标房间名称（如"Room_03"）
       - "response": 自然语言确认信息（如"AI 'ms' 正在前往 Room_03"）
    3. 如果输入格式错误，返回空数组。
    示例输出：
    {{
        "responses": [
            {{
                "name": "ms",
                "target_room": "Room_03",
                "response": "AI 'ms' 已开始前往 Room_03"
            }}
        ]
    }}
    """