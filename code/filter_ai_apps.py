import os
import json
import re
import html

# 配置路径
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_file = os.path.join(base_dir, "data", "google_play_apps_latest_post_2022.jsonl")
output_file = os.path.join(base_dir, "data", "google_play_apps_ai_filtered.jsonl")

# 1. 预编译正则表达式以提高千万级数据的处理性能
# HTML 标签剥离：匹配所有 <...>，之后将会被替换为空格
PATTERN_HTML_TAG = re.compile(r'<[^>]+>')

# 严格的 AI 缩写匹配：使用否定环视 (Negative Lookaround) 替代 \b
# 这样既支持独立单词 "AI" 和 "A.I."，又能完美防止字母黏连带来的子串误判，杜绝 "AIR"(空气), "AIM"(目标), "AID"(援助) 等词的误杀
PATTERN_STRICT_AI = re.compile(r'(?<![a-zA-Z])(AI|A\.I\.)(?![a-zA-Z])')

# 宽容的 AI 相关词汇匹配 (包含生成式AI、机器学习、传统全称等)：忽略大小写
# 涵盖了主流模型家族、平台库、下游任务、移动端App常见AI术语(Avatar, Enhancer, Voice Cloning)等
PATTERN_BROAD_AI = re.compile(
    r'\b(chatgpt|gpt(?:-?[234](?:\.5)?(?:-?(?:turbo|o(?:-?mini)?))?)?|'
    r'lla?ma(?:-?[23](?:\.1)?(?:-?(?:chat|instruct))?)?|mistral(?:-?[23]?(?:\.x)?(?:-?(?:chat|instruct))?)?|'
    r'qwen(?:-?[12]?(?:\.5)?(?:-?(?:chat|instruct))?)?|deepseek(?:-?(?:chat|coder|v2|v3|r1)?)?|phi(?:-?[234]?)?|gemma|'
    r'llms?|large language model|generative ai|genai|aigc|midjourney|dall-?e|stable diffusion|'
    r'sora|runwayml?|whisper|'
    r'openai|anthropic|claude|gemini|bard|copilot|hugging ?face|langchain|perplexity|'
    r'machine learning|deep learning|neural network|nlp|natural language processing|rag|'
    r'ai[- ]*(?:agent|chatbot|assistant|copilot|generator|art|voice|avatar|photo|filter|enhancer|character|companion)|'
    r'artificial intelligence|text[- ]to[- ]text|text[- ]to[- ]image|'
    r'text[- ]to[- ]video|text[- ]to[- ]speech|speech[- ]to[- ]text|image[- ]to[- ]text|'
    r'voice[- ]clon(?:e|ing)|deepfake|'
    r'multimodal)\b',
    re.IGNORECASE
)

# 传统/游戏 AI 排雷词表：如果匹配到这些，大概率是旧时代的游戏 NPC (打上标签，但不丢弃)
PATTERN_TRADITIONAL_AI = re.compile(
    r'\b(ai opponent|ai enemy|ai bot|enemy ai|game ai|cpu opponent|computer opponent|versus ai|vs ai)\b', 
    re.IGNORECASE
)


def clean_text(text):
    """
    文本清洗管道：
    1. html.unescape() 将 &#39; 等 HTML 实体转化为正常符号 (比如单引号)
    2. 将所有的 <br>, <b> 等 HTML 标签替换为空格 (防连字)
    3. 合并多余的连续空格
    """
    if not text:
        return ""
    
    # 1. 反转义 HTML 实体字符
    text = html.unescape(text)
    # 2. 剥离 HTML 标签，替换为空格
    text = PATTERN_HTML_TAG.sub(' ', text)
    # 3. 规范化空格、换行符（把连续的空白字符替换为一个空格）
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def filter_ai_apps():
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}。请先运行时间及版本筛选脚本。")
        return

    print(f"开始 AI 应用深度挖掘扫描...\n输入文件: {input_file}")
    
    processed_count = 0
    ai_app_count = 0
    error_count = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
         
        for line in f_in:
            processed_count += 1
            
            try:
                record = json.loads(line)
                
                # 1. 安全提取目标文本字段 (主要依赖完整的 html 描述)
                desc_html = record.get('descriptionHtml') or ""
                
                # 2. 安全获取深层嵌套的 recentChangesHtml
                details = record.get('details') or {}
                app_details = details.get('appDetails') or {}
                recent_changes = app_details.get('recentChangesHtml') or ""
                
                raw_full_text = " ".join([desc_html, recent_changes])
                
                # 3. 终极深度清洗
                clean_full_text = clean_text(raw_full_text)
                
                if not clean_full_text:
                    continue
                
                # 4. 执行多维正则匹配
                has_strict_ai = bool(PATTERN_STRICT_AI.search(clean_full_text))
                has_broad_ai = bool(PATTERN_BROAD_AI.search(clean_full_text))
                
                # 只要命中任意一个 AI 规则，我们就把它收录进来
                if has_strict_ai or has_broad_ai:
                    # 5. 负面标签打标检测 (判断是否是游戏中的传统 "AI 对手")
                    is_traditional = bool(PATTERN_TRADITIONAL_AI.search(clean_full_text))
                    
                    # 6. 为 JSON 附加一层我们的 AI 分析结果字典，方便后续数据集直接使用
                    record["ai_analysis_tags"] = {
                        "is_ai": True,
                        "strict_ai_matched": has_strict_ai,
                        "broad_ai_matched": has_broad_ai,
                        "is_traditional_game_ai": is_traditional
                    }
                    
                    # 重新序列化并写入
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    ai_app_count += 1
                    
            except json.JSONDecodeError:
                error_count += 1
            except Exception as e:
                error_count += 1
                
            # 控制台进度刷新
            if processed_count % 50000 == 0:
                print(f"\r已扫描: {processed_count} 条记录 | 当前已挖掘出 AI 应用: {ai_app_count} 个... ", end="")

    print(f"\n\n过滤挖掘完毕！")
    print(f"总计扫描记录: {processed_count}")
    print(f"100% 覆盖找到的 AI 应用数: {ai_app_count}")
    print(f"提取与解析错误数: {error_count}")
    print(f"高质量 AI 数据集已生成至: {output_file}")


if __name__ == "__main__":
    filter_ai_apps()