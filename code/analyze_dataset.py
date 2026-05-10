import os
import json
from collections import Counter

def analyze_dataset():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "data", "google_play_apps_final_dataset.jsonl")

    if not os.path.exists(input_file):
        print(f"找不到数据文件: {input_file}")
        return

    total_apps = 0
    downloads_counter = Counter()
    has_ads_count = 0
    developers_counter = Counter()

    # 权限统计
    permissions_counter = Counter()

    # 评分统计
    valid_ratings_count = 0
    total_rating_sum = 0.0

    # 缺失值/无效值统计
    missing_stats = {
        "docid": 0,
        "title": 0,
        "descriptionShort": 0,
        "descriptionHtml": 0,
        "developerName": 0,
        "developerEmail": 0,
        "numDownloads": 0,
        "installationSize": 0,
        "uploadDate": 0,
        "versionString": 0,
        "empty_permissions": 0
    }

    print("正在解析数据...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): 
                continue
            
            record = json.loads(line)
            total_apps += 1

            # 检查缺失值
            if not record.get("docid"): missing_stats["docid"] += 1
            if not record.get("title"): missing_stats["title"] += 1
            if not record.get("descriptionShort"): missing_stats["descriptionShort"] += 1
            if not record.get("descriptionHtml"): missing_stats["descriptionHtml"] += 1

            details = record.get("details", {}).get("appDetails", {})
            
            if not details.get("developerName"): missing_stats["developerName"] += 1
            if not details.get("developerEmail"): missing_stats["developerEmail"] += 1
            if not details.get("numDownloads"): missing_stats["numDownloads"] += 1
            if not details.get("installationSize"): missing_stats["installationSize"] += 1
            if not details.get("uploadDate"): missing_stats["uploadDate"] += 1
            if not details.get("versionString"): missing_stats["versionString"] += 1
            if not details.get("permission"): missing_stats["empty_permissions"] += 1

            # 统计广告
            if details.get("containsAds") == "Contains ads":
                has_ads_count += 1
                
            # 统计下载量
            downloads = details.get("numDownloads", "Unknown")
            downloads_counter[downloads] += 1
            
            # 统计开发者
            developer = details.get("developerName", "Unknown")
            developers_counter[developer] += 1
            
            # 统计权限
            permissions = details.get("permission", [])
            for p in permissions:
                permissions_counter[p] += 1
                
            # 统计评分
            rating_info = record.get("aggregateRating", {})
            star_rating = float(rating_info.get("starRating", 0.0))
            if star_rating > 0:
                valid_ratings_count += 1
                total_rating_sum += star_rating

    print(f"\n================ 数据集分析报告 ================")
    print(f"➤ 总应用数量: {total_apps}")
    
    if total_apps == 0:
        return
        
    print(f"➤ 包含广告的应用: {has_ads_count} 个 ({(has_ads_count/total_apps*100):.2f}%)")
    
    if valid_ratings_count > 0:
        print(f"➤ 有效评分应用数: {valid_ratings_count} 个")
        print(f"➤ 总体平均星级: {(total_rating_sum / valid_ratings_count):.2f} / 5.0")
    else:
        print("➤ 暂无评分数据")

    print("\n--- 下载量分布 (Top 10) ---")
    for d, c in downloads_counter.most_common(10):
        print(f"  {d}: {c} 个")

    print("\n--- 最常请求的权限 (Top 10) ---")
    for p, c in permissions_counter.most_common(10):
        # 让权限名显示得短一点，更容易阅读
        p_short = p.split('.')[-1]
        print(f"  {p_short} (完整: {p}): {c} 次")
        
    print("\n--- 提交最多应用的开发者 (Top 5) ---")
    for dev, c in developers_counter.most_common(5):
        print(f"  {dev}: {c} 个应用")        
    print("\n--- 数据缺失与无效值分析 ---")
    print("以下字段表示为空字符串('')、None或空列表[]的应用数量(及其比例):")
    for field, count in missing_stats.items():
        if count > 0:
            print(f"  [!] {field}: {count} 个缺失 ({(count/total_apps*100):.2f}%)")
        else:
            print(f"  [✓] {field}: 无缺失")
    print("================================================\n")

if __name__ == '__main__':
    analyze_dataset()
