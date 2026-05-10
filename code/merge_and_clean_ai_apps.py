import os
import json

def merge_and_clean_ai_apps():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. 输入文件
    qwen_yes_file = os.path.join(base_dir, "data", "google_play_apps_qwen_yes.jsonl")
    raw_filtered_file = os.path.join(base_dir, "data", "google_play_apps_ai_filtered.jsonl")
    
    # 2. 最终输出的数据集文件
    output_file = os.path.join(base_dir, "data", "google_play_apps_final_dataset.jsonl")

    if not os.path.exists(qwen_yes_file):
        print(f"Error: 找不到 Qwen 验证通过名单文件 {qwen_yes_file}")
        return
    if not os.path.exists(raw_filtered_file):
        print(f"Error: 找不到包含完整信息的原始过滤文件 {raw_filtered_file}")
        return

    # 第一步：把所有 is_ai == "yes" 的 docid 加载到内存集合(Set)中，实现 O(1) 极速查找
    print(f"步骤 1: 正在加载 Qwen 验证为 'yes' 的白名单库...")
    verified_docids = set()
    with open(qwen_yes_file, 'r', encoding='utf-8') as f_yes:
        for line in f_yes:
            line = line.strip()
            if not line: continue
            try:
                record = json.loads(line)
                docid = record.get("docid")
                if docid:
                    verified_docids.add(docid)
            except Exception as e:
                pass
    print(f"白名单加载完成，总共找到 {len(verified_docids)} 个确认的 AI 应用。")

    # 第二步：遍历包含完整信息的原始文件，进行匹配（Join）和字段清洗
    print(f"\n步骤 2: 开始从原始文件提取并清洗数据...")
    processed_count = 0
    extracted_count = 0

    fields_to_remove = ["image", "ai_analysis_tags"]

    with open(raw_filtered_file, 'r', encoding='utf-8') as f_raw, \
         open(output_file, 'w', encoding='utf-8') as f_out:
         
        for line in f_raw:
            processed_count += 1
            line = line.strip()
            if not line: continue
            
            try:
                record = json.loads(line)
                docid = record.get("docid")
                
                # 如果这个应用的 docid 存在于 Qwen 盖章通过的白名单里
                if docid in verified_docids:
                    # 移除不需要的字段
                    for field in fields_to_remove:
                        record.pop(field, None) # None 保证如果字段本来就不存在也不会报错
                    
                    # 将清洗洗合并后的纯净数据写入最终数据集
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    extracted_count += 1
                    
                    # 可以在这里从集合中删除已经找到的 docid，加速后续查找并防止原始数据中的重复数据导致重复写入
                    # verified_docids.remove(docid) 
                    
            except json.JSONDecodeError:
                continue
                
            if processed_count % 10000 == 0:
                print(f"\r已扫描原始记录: {processed_count} 条 | 已成功提取最终 AI 应用: {extracted_count} 个", end="")

    print(f"\n\n任务完成！")
    print(f"成功提取并清洗了 {extracted_count} 个 AI 应用信息。")
    print(f"不需要的字段 {fields_to_remove} 已经被剔除。")
    print(f"最终的纯净黄金数据集保存在: \n{output_file}")


if __name__ == "__main__":
    merge_and_clean_ai_apps()
