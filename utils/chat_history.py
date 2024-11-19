from openai import OpenAI
 
client = OpenAI(
    #api_key = "token-abc123", # 本地模型api-key
    api_key = 'sk-e59bb34cf3f14d2296526aa27896cf04',#阿里云Dashscope api-key
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1", #阿里云Dashscope地址
    #base_url = "http://192.168.68.61:8000/v1", #本地模型地址
)
 
history = [
    {"role": "system", "content": "你是由阿里巴巴提供的AI大模型问答服务，你擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。"}
]
 
def chat(query, history):
    history.append({
        "role": "user", 
        "content": query
    })
    completion = client.chat.completions.create(
        #model="qwen25-14b", #本地模型名称
        model="qwen-turbo",
        messages=history,
        temperature=0.2,
    )
    result = completion.choices[0].message.content
    history.append({
        "role": "assistant",
        "content": result
    })
    return result