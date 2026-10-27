# 手写 ReAct 循环

import json
import os
import urllib.request
from pathlib import Path

# ================= 加载 .env =================
def load_env(path: Path = Path(__file__).with_name(".env")) -> None:
    """把 .env 里的配置读进环境变量（已存在的环境变量优先，不覆盖）"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

load_env()

# ================= 定义工具 =================
def get_weather(city: str) -> str:
    """模拟天气查询工具"""
    if "北京" in city:
        return "北京今天晴，气温 25 度。"
    elif "上海" in city:
        return "上海今天大雨，气温 20 度。"
    return f"查不到{city}的天气信息。"

# 模块 1：工具注册表（Router/Registry）
# 新增工具只需在这里登记：工具名 -> 描述（给模型看）+ 执行函数
TOOLS = {
    "get_weather": {
        "description": '查询天气。参数为城市的名字 (例如: "北京")。',
        "fn": get_weather
    }
}

# ================= 编写 System Prompt =================
# 模块 4：Prompt 约束（契约）
def build_system_prompt() -> str:
    tool_list = "\n".join(
        f"- {name}: {info['description']}" for name, info in TOOLS.items()
    )
    return f"""
            你是一个智能助手，可以通过调用工具来回答用户问题。

            你可以使用的工具如下：
            {tool_list}

            请严格按照以下格式回答：
            思考: 你的思考过程
            行动: 工具名称
            参数: 工具参数 (必须是合法的 JSON 字符串，如果没有参数则为 {{}})
            """

SYSTEM_PROMPT = build_system_prompt()

# LLM调用
def call_llm(prompt_history):
    # 阿里云百炼（DashScope）的 OpenAI 兼容接口，API_KEY 从 .env 读取
    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise SystemExit("未找到 API_KEY：请在 .env 中配置 API_KEY=sk-...")
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "qwen-plus",  # 百炼通义千问；可换成 qwen-turbo / qwen-max 等
        "messages": prompt_history,
        "temperature": 0.1  # 温度调低，让输出更稳定
    }
    
    req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers)

    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))
        return result["choices"][0]["message"]["content"]


# ================= 运行 Agent =================
def run_agent(user_query: str):
    print(f"\n👤 用户: {user_query}")
    
    # 初始化上下文（记忆系统）
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
    
    # 设定最大循环次数，防止死循环（熔断机制）
    max_steps = 5 

    for step in range(max_steps):
        print(f"\n 这是第 {step+1} 次循环：")

        # 1. 让模型思考（LLM 推理）
        response_text = call_llm(messages) 
        print(f"🤖 模型输入:\n{messages}")
        print(f"🤖 模型输出:\n{response_text}")
        
        # 把模型的回答加入上下文
        messages.append({"role": "assistant", "content": response_text})

        # 2. 解析模型输出（简单的文本解析，真实生产环境会用 JSON Mode 或 Function Calling）
        if "行动:" in response_text:
            try:
                # 思考: 用户询问北京今天的天气情况以及是否适合出门，需要先获取北京的天气信息，再根据天气情况判断是否适合出门。
                # 行动: get_weather
                # 参数: {"city": "北京"}
                action = response_text.split("行动:")[1].split("参数:")[0].strip() # 拆分行动和参数之间的文本
                params = json.loads(response_text.split("参数:")[1].strip()) # 拆分参数部分并解析为 JSON
                print(f"\n🛠️ 准备调用工具: {action}, 参数: {params}")
                
                # 3. 执行工具
                if action in TOOLS:
                    # 这里可以加入后端的异常捕获、重试机制
                    observation = TOOLS[action]["fn"](**params)
                else:
                    observation = f"错误：找不到工具 {action}"

                print(f"👀 观察结果: {observation}")

                # 4. 将观察结果喂回给模型（Feedback）
                messages.append({
                    "role": "user", 
                    "content": f"观察: {observation}\n请根据观察结果继续思考，如果得出最终答案，请直接输出：最终答案: [你的回答]"
                })
                
            except Exception as e:
                # 错误兜底：如果模型输出格式错误，打回去重试
                error_msg = f"解析失败：{str(e)}。请严格按照'思考/行动/参数'的格式输出。"
                messages.append({"role": "user", "content": error_msg})
        elif "最终答案:" in response_text:
            # 模型决定结束循环
            final_answer = response_text.split("最终答案:")[1].strip()
            print(f"\n✅ Agent 最终回答: {final_answer}")
            return final_answer
        else:
            # 既没有行动也没有最终答案，提示模型
            messages.append({"role": "user", "content": "请输出'行动'或'最终答案'。"})

    print("\n⚠️ 达到最大循环次数，强制终止（熔断触发）。")
    return None

# ================= 测试运行 =================
if __name__ == "__main__":
    user_query = input("请输入你的问题（直接回车退出）: ").strip()
    if user_query:
        run_agent(user_query)