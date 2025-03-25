from fastapi import FastAPI, HTTPException
import openai  # 需要先配置你的 OpenAI API Key

app = FastAPI()

# 配置 OpenAI API Key（替换为你的密钥）
openai.api_key = "YOUR_OPENAI_API_KEY"

@app.post("/process")
async def process_text(text: str):
    try:
        # 1. 分割字符串提取关键词（示例：提取类似"Room_XXX"的关键词）
        keywords = [word for word in text.split() if word.startswith("Room_")]

        # 2. 调用 LLM API 处理文本
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"根据用户输入 '{text}'，生成行动指令。关键词：{keywords}",
            max_tokens=100
        )
        ai_response = response.choices[0].text.strip()

        # 3. 返回结果（包含关键词和 LLM 的响应）
        return {
            "keywords": keywords,
            "response": ai_response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)