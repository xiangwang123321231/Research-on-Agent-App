import os
import json

"""
精炼结构，提取出docid、descriptionHtml、recentChangesHtml三个字段
目的：为后续大模型筛选提供更干净的输入文本，减少不必要的字段干扰。
注意：descriptionHtml 和 recentChangesHtml 可能包含 HTML 标签，后续需要进行清洗处理。
"""
# 配置路径
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_file = os.path.join(base_dir, "data", "google_play_apps_ai_filtered.jsonl")
output_file = os.path.join(base_dir, "data", "google_play_apps_ai_extracted.jsonl")

def extract_fields():
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}。请先运行 `filter_ai_apps.py` 脚本。")
        return

    print(f"开始提取目标字段...\n输入文件: {input_file}")
    
    processed_count = 0
    extracted_count = 0
    error_count = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
         
        for line in f_in:
            processed_count += 1
            try:
                record = json.loads(line)
                
                # 提取基础字段
                # 注意：有些数据的包名可能是在 'docid' 或其他字段中，这里以提取 docid 为准
                docid = record.get('docid', '') 
                
                # 提取介绍字段
                description_html = record.get('descriptionHtml', '')
                
                # 提取嵌套的版本更新字段
                details = record.get('details', {})
                app_details = details.get('appDetails', {})
                recent_changes = app_details.get('recentChangesHtml', '')
                
                # 构建精简结构
                extracted_record = {
                    "docid": docid,
                    "descriptionHtml": description_html,
                    "recentChangesHtml": recent_changes
                }
                
                f_out.write(json.dumps(extracted_record, ensure_ascii=False) + "\n")
                extracted_count += 1
                    
            except json.JSONDecodeError:
                error_count += 1
            except Exception as e:
                error_count += 1
                
            if processed_count % 10000 == 0:
                print(f"\r已处理: {processed_count} 条记录 | 已提取: {extracted_count} 条... ", end="")

    print(f"\n\n提取完成！")
    print(f"扫描总记录数: {processed_count}")
    print(f"成功提取记录数: {extracted_count}")
    print(f"解析错误数: {error_count}")
    print(f"精简后的文件已保存至:\n{output_file}")

if __name__ == "__main__":
    extract_fields()
