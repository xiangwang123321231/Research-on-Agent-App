import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件中的环境变量
load_dotenv()

# 获取 API Key
api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
if not api_key:
    raise ValueError("请在 .env 文件中设置 DASHSCOPE_API_KEY 或 QWEN_API_KEY")

# 初始化 OpenAI 客户端，使用阿里云 DashScope 的兼容模式接口
client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 配置路径
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_file = os.path.join(base_dir, "data", "google_play_apps_ai_extracted.jsonl")
output_file = os.path.join(base_dir, "data", "google_play_apps_qwen_verified.jsonl")

# 设定的模型名称
MODEL_NAME = "qwen3.6-plus"

def check_is_ai_app(description, recent_changes):
    """调用 Qwen 模型检查应用是否属于 AI 应用"""
    
    system_prompt = (
        "你是一个专业的应用分析助手。请通过用户提供的移动应用介绍 (Description) "
        "和更新日志 (Recent Changes)，判断这款应用是否包含或者自称包含AI相关功能。"
        "注意：现阶段我们的筛选标准非常宽松，只要应用在描述或更新日志中自称是AI应用，"
        "或者稍微提及了其包含AI功能（如AI助手, AI生成, 机器学习等，即使可能是蹭热度或极为边缘的功能），"
        "都请判断为是AI应用（is_ai: true）。只有完全与AI无关的应用才填 false。"
        "\n请只返回一个合法的 JSON 字符串，格式如下：\n"
        '{"is_ai": true/false, "reason": "在此简要说明判断理由"}'
    )
    
    user_prompt = f"应用介绍:\n{(description or '')[:3000]}\n\n最近更新:\n{(recent_changes or '')[:1000]}"
    
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
        return json.loads(result_str)
        
    except Exception as e:
        print(f"\n模型调用失败: {e}")
        return {"is_ai": None, "reason": f"Error: {str(e)}"}

def verify_apps():
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}。")
        return

    print(f"开始使用 {MODEL_NAME} 验证 AI 应用...\n输入文件: {input_file}")
    
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
                    if data.get("llm_verification", {}).get("is_ai"):
                        ai_app_count += 1
                except:
                    pass
        print(f"发现已处理记录 {len(processed_docids)} 条 (其中已确认为AI应用 {ai_app_count} 个)，将跳过这些记录...")

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
                
                record["llm_verification"] = llm_result
                
                # 写入结果
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush() # 实时保存，防止中断
                
                processed_count += 1
                is_ai_zh = "是" if llm_result.get("is_ai") else "否"
                if llm_result.get("is_ai"):
                    ai_app_count += 1
                print(f"[验证完成] 包名: {docid} -> 是否AI: {is_ai_zh} | 当前确认为AI的总数: {ai_app_count} | 理由: {llm_result.get('reason', '')[:30]}...")
                
                # 为了防止触发 API 限流，可以稍微等待一下
                time.sleep(0.5)
                
            except json.JSONDecodeError:
                pass

    print(f"\n验证结束！本次处理了 {processed_count} 条记录。")
    print(f"当前总计通过验证的 AI 应用数量: {ai_app_count} 个。")
    print(f"结果保存至: {output_file}")


if __name__ == "__main__":
    verify_apps()
