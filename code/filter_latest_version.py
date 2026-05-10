import os
import json
"""
根据版本筛选，保留最新版本的应用数据，剔除历史版本冗余记录。
"""
# 配置路径
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_file = os.path.join(base_dir, "data", "google_play_apps_post_2022.jsonl")
output_file = os.path.join(base_dir, "data", "google_play_apps_latest_post_2022.jsonl")

def find_latest_versions():
    """
    因为要找到每个应用（docid/包名）的最新版本，但文件极大不能全部塞入内存。
    我们采用经典且高效的“双趟扫描（Two-Pass）”算法：
    第一趟：只读取必要字段，在内存中维护每个 docid 对应的最大 versionCode。
    第二趟：根据最大 versionCode 字典，边读边将命中最大版本的数据写入本地，并剔除字典缓存，处理重复值。
    """
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}。请先运行时间筛选脚本。")
        return

    print("=== 第一阶段：扫描所有文件，建立最新版本号映射 ===")
    max_versions = {}
    processed_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            processed_count += 1
            try:
                record = json.loads(line)
                docid = record.get("docid")  # 官方文档提到 docid/packageName 是全局唯一
                details = record.get("details") or {}
                app_details = details.get("appDetails") or {}
                versionCode_raw = app_details.get("versionCode", -1)
                az_metadata_date = record.get("az_metadata_date", "")
                
                try:
                    versionCode = int(versionCode_raw)
                except (ValueError, TypeError):
                    versionCode = -1

                if docid:
                    # 如果 app 没出现过
                    if docid not in max_versions:
                        max_versions[docid] = {"versionCode": versionCode, "date": az_metadata_date}
                    else:
                        current_max_vc = max_versions[docid]["versionCode"]
                        current_max_date = max_versions[docid]["date"]
                        
                        # 1. 如果版本号更大，直接替换
                        if versionCode > current_max_vc:
                            max_versions[docid] = {"versionCode": versionCode, "date": az_metadata_date}
                        # 2. 如果版本号相同，按官方要求比较 az_metadata_date，保留最新爬取的记录
                        elif versionCode == current_max_vc and az_metadata_date > current_max_date:
                            max_versions[docid] = {"versionCode": versionCode, "date": az_metadata_date}
            except json.JSONDecodeError:
                pass
            
            if processed_count % 100000 == 0:
                print(f"\r[阶段1] 已扫描: {processed_count} 条记录, 发现 {len(max_versions)} 个唯一应用...", end="")

    print(f"\n第一阶段完成，共提取 {len(max_versions)} 个独立应用的最新版本记录。")
    print("=== 第二阶段：重扫文件并边筛边写入本地文件 ===")

    processed_count_2 = 0
    written_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
         
        for line in f_in:
            processed_count_2 += 1
            try:
                record = json.loads(line)
                docid = record.get("docid")
                details = record.get("details") or {}
                app_details = details.get("appDetails") or {}
                versionCode_raw = app_details.get("versionCode", -1)
                az_metadata_date = record.get("az_metadata_date", "")
                
                try:
                    versionCode = int(versionCode_raw)
                except (ValueError, TypeError):
                    versionCode = -1
                
                # 双重校验：必须 docid、versionCode 且 az_metadata_date 完全匹配我们缓存的最优解
                if docid and docid in max_versions:
                    target = max_versions[docid]
                    if versionCode == target["versionCode"] and az_metadata_date == target["date"]:
                        f_out.write(line)
                        written_count += 1
                        
                        # 核心防重复：成功写入后删除，防止同一时间、同版本出现绝对冗余项（双保险）
                        del max_versions[docid]

            except json.JSONDecodeError:
                pass
            
            if processed_count_2 % 100000 == 0:
                print(f"\r[阶段2] 已重扫: {processed_count_2} 条记录, 已成功找到并写入: {written_count} 个应用...", end="")

    print(f"\n\n过滤完成！")
    print(f"最终保留了 {written_count} 个最新版本的应用数据。")
    print(f"结果已保存至: {output_file}")


if __name__ == "__main__":
    find_latest_versions()
