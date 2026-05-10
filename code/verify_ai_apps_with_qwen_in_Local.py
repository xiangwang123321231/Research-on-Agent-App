# -*- coding: utf-8 -*-
import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件中的环境变量
load_dotenv()

# 获取 API Key (本地模型往往不需要真实的 Key)
api_key = os.getenv("LOCAL_API_KEY", "lm-studio")

# 初始化 OpenAI 客户端，调用本地服务 
# Ollama 默认: http://localhost:11434/v1
client = OpenAI(
    api_key=api_key,
    base_url="http://localhost:11434/v1", # 以 Ollama 为例
)
# 配置路径
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_file = os.path.join(base_dir, "data", "google_play_apps_ai_extracted.jsonl")
output_file = os.path.join(base_dir, "data", "google_play_apps_qwen_verified.jsonl")

# 设定的模型名称 (本地服务部署的具体模型名称)
MODEL_NAME = "qwen3:14b" # 请换成你本地运行的模型名，例如 "qwen" 

def check_is_ai_app(description, recent_changes):
    """调用 Qwen 模型检查应用是否属于 AI 应用"""
    
    system_prompt = (
        "你是一个专业的应用分析助手。请通过用户提供的移动应用介绍 (Description) "
        "和更新日志 (Recent Changes)，判断这款应用是否包含或者自称包含AI相关功能。"
        "注意：现阶段我们的筛选标准非常宽松，只要应用在描述或更新日志中自称是AI应用，"
        "或者稍微提及了其包含AI功能（如AI助手, AI生成, 机器学习等，即使可能是蹭热度或极为边缘的功能），"
        "都请判断为是AI应用（is_ai: true）。"
        "【重要排除条件】：如果该应用是一款游戏，且提到的“AI”仅仅是指游戏内的常规AI（如：AI对手、电脑玩家、NPC行为、敌军AI等），请不要将其判定为AI应用（即 is_ai: false）。"
        "只有完全与AI无关，或仅包含基础游戏AI的应用才填 false。"
        "\n请只返回一个最简的 JSON 字符串，并输出简要的理由，格式如下：\n"
        '{"is_ai": true, "reason": "..."} 或 {"is_ai": false, "reason": "..."}'
    )
    
    user_prompt = "应用介绍:\n{}\n\n最近更新:\n{}".format(
        (description or '')[:3000],
        (recent_changes or '')[:1000]
    )
    
    max_retries = 20
    for attempt in range(max_retries):
        # 建立一个较为安全的重试时间递增机制
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            result_str = response.choices[0].message.content
            
            # 兼容本地模型经常带用 Markdown 格式输出 JSON 的情况
            result_str = result_str.strip()
            if result_str.startswith("```json"):
                result_str = result_str[7:]
            elif result_str.startswith("```"):
                result_str = result_str[3:]
            if result_str.endswith("```"):
                result_str = result_str[:-3]
            result_str = result_str.strip()
            
            return json.loads(result_str)
            
        except Exception as e:
            print("\n模型调用失败 (尝试 {}/{}): {}".format(attempt + 1, max_retries, e))
            if attempt == max_retries - 1:
                return {"is_ai": None, "reason": "Error: {}".format(str(e))}
            time.sleep(min(2 * (attempt + 1), 10)) # 错误后延迟递增，最高到10秒

def verify_apps():
    if not os.path.exists(input_file):
        print("错误: 找不到输入文件 {}。".format(input_file))
        return

    print("开始使用 {} 验证 AI 应用...\n输入文件: {}".format(MODEL_NAME, input_file))
    
    processed_count = 0
    ai_app_count = 0
    
    # 支持断点续传（如果输出文件存在，先读取已处理的数量）
    processed_docids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f_out:
            for line in f_out:
                try:
                    data = json.loads(line)
                    processed_docids.add(data.get("docid"))
                    # 兼容之前可能嵌套的格式和新的扁平格式
                    is_ai = data.get("is_ai") if "is_ai" in data else data.get("llm_verification", {}).get("is_ai")
                    if is_ai == "yes" or is_ai is True:
                        ai_app_count += 1
                except:
                    pass
        print("发现已处理记录 {} 条 (其中已确认为AI应用 {} 个)，将跳过这些记录...".format(len(processed_docids), ai_app_count))

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'a', encoding='utf-8') as f_out:
         
        for line in f_in:
            try:
                record = json.loads(line)
                docid = record.get("docid", "")
                
                if docid in processed_docids:
                    continue
                
                desc = record.get("descriptionHtml", "")
                changes = record.get("recentChangesHtml", "")
                
                # 调用大模型验证
                llm_result = check_is_ai_app(desc, changes)
                
                # 如果连续重试失败 (例如本地服务崩溃)，直接停止执行
                if llm_result.get("is_ai") is None:
                    print("\n[中止] 模型服务出现连续错误，请检查本地模型是否正常运行。错误原因: {}".format(llm_result.get('reason')))
                    break
                
                # 构建精简输出格式，加入 reason 字段
                is_ai_str = "yes" if llm_result.get("is_ai") else "no"
                output_record = {
                    "docid": docid,
                    "is_ai": is_ai_str,
                    "reason": llm_result.get("reason", "")
                }
                
                # 写入精简结果
                f_out.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                f_out.flush() # 实时保存，防止中断
                
                processed_count += 1
                if is_ai_str == "yes":
                    ai_app_count += 1
                print("[验证完成] 包名: {} -> {} | 理由: {}... | 当前确认为AI的总数: {}".format(
                    docid, is_ai_str, output_record['reason'][:30], ai_app_count))
                
            except json.JSONDecodeError:
                pass

    print("\n验证结束！本次处理了 {} 条记录。".format(processed_count))
    print("当前总计通过验证的 AI 应用数量: {} 个。".format(ai_app_count))
    print("结果保存至: {}".format(output_file))


if __name__ == "__main__":
    verify_apps()
