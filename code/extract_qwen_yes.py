import os
import json

def extract_verified_ai_apps():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "data", "google_play_apps_qwen_verified.jsonl")
    output_file = os.path.join(base_dir, "data", "google_play_apps_qwen_yes.jsonl")

    if not os.path.exists(input_file):
        print(f"Error: 找不到输入文件 {input_file}")
        return

    print(f"开始从 {input_file} 提取 Qwen 验证通过 (is_ai = yes) 的应用...")

    processed_count = 0
    yes_count = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
                
            processed_count += 1
            try:
                record = json.loads(line)
                # 检查 is_ai 字段是否为 yes (忽略大小写)
                if str(record.get("is_ai", "")).strip().lower() == "yes":
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    yes_count += 1
            except json.JSONDecodeError:
                print(f"警告: 第 {processed_count} 行无法解析为 JSON: {line[:50]}...")
            except Exception as e:
                print(f"处理第 {processed_count} 行时发生错误: {e}")

            if processed_count % 1000 == 0:
                print(f"\r已处理: {processed_count} 行 | 提取符合条件的 AI 应用: {yes_count} 个", end="")

    print(f"\n\n提取完成！")
    print(f"总计检查记录: {processed_count}")
    print(f"验证结果为 'yes' 的应用数: {yes_count}")
    print(f"结果已保存至: {output_file}")


if __name__ == "__main__":
    extract_verified_ai_apps()
