import gzip
import os
import json
import re
from datetime import datetime

# 配置路径
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_file = os.path.join(base_dir, "androzoo-metadata", "gp-metadata-full.jsonl.gz")
output_dir = os.path.join(base_dir, "data")
output_file = os.path.join(output_dir, "google_play_apps_post_2022.jsonl")

os.makedirs(output_dir, exist_ok=True)

# 目标过滤日期点 (2022年11月1日)
TARGET_DATE = datetime(2022, 11, 1)

def parse_date(date_str):
    """
    尝试解析完全无视系统语言环境 (Locale) 的日期格式，
    预防在中文 Windows 系统下 strptime 无法解析英文月份 (如 "Mar") 的致命问题。
    """
    if not date_str:
        return None
        
    date_str = date_str.strip()
    
    # 1. 尝试解析标准 ISO 格式 (YYYY-MM-DD)
    if '-' in date_str and len(date_str) >= 10:
        try:
            return datetime.strptime(date_str[:10], "%Y-%m-%d")
        except ValueError:
            pass

    # 英文月份前缀哈希表
    MONTHS = {
        'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
        'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12
    }
    
    # 2. 尝试匹配 "Mar 26, 2020", "Mar. 26 2020"
    matches = re.findall(r'([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})', date_str)
    if matches:
        month_str, day_str, year_str = matches[0]
    else:
        # 3. 尝试匹配 "26 Mar 2020" 或 "26 Mar. 2020"
        matches = re.findall(r'(\d{1,2})\s+([A-Za-z]+)\.?\,?\s+(\d{4})', date_str)
        if matches:
            day_str, month_str, year_str = matches[0]
        else:
            return None

    # 将提取的月份转为纯数字
    month_prefix = month_str.lower()[:3]
    if month_prefix in MONTHS:
        try:
            return datetime(int(year_str), MONTHS[month_prefix], int(day_str))
        except ValueError:
            pass

    return None

def filter_apps():
    print(f"开始处理: {input_file}")
    print(f"过滤条件: uploadDate >= 2022年11月1日")
    
    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}。")
        return

    processed_count = 0
    matched_count = 0
    error_count = 0
    unparsed_date_count = 0
    missing_date_count = 0

    try:
        # 预编译正则以提升上百倍的匹配速度
        upload_date_pattern = re.compile(r'"uploadDate"\s*:\s*"([^"]+)"')
        
        # 使用 gzip 和 iter 流式处理，极低内存消耗
        with gzip.open(input_file, 'rt', encoding='utf-8') as f_in:
            with open(output_file, 'w', encoding='utf-8') as f_out:
                for line in f_in:
                    processed_count += 1
                    
                    try:
                        # 性能优化：直接正则提取日期，跳过耗时的 json.loads
                        match = upload_date_pattern.search(line)
                        if match:
                            upload_date_str = match.group(1)
                            parsed_date = parse_date(upload_date_str)
                            if parsed_date:
                                if parsed_date >= TARGET_DATE:
                                    f_out.write(line)
                                    matched_count += 1
                            else:
                                unparsed_date_count += 1
                        else:
                            missing_date_count += 1
                    except Exception as loop_e:
                        # 避免个别极为脏乱的数据直接导致脚本全盘崩溃退出
                        error_count += 1
                    
                    # 打印进度 (每十万条打印一次进度)
                    if processed_count % 100000 == 0:
                        print(f"\r已处理: {processed_count} | 满足条件: {matched_count} | 日期解析失败: {unparsed_date_count}", end="")

        print(f"\n\n过滤完成！")
        print(f"总计处理记录: {processed_count}")
        print(f"--> 符合条件的记录数: {matched_count}")
        print(f"--> 缺少日期的记录数: {missing_date_count}")
        print(f"--> 无法解析日期的记录数: {unparsed_date_count} (非常重要，如大于0需调查数据)")
        print(f"--> JSON解析错误的记录: {error_count}")
        print(f"结果已保存至: {output_file}")
        
    except Exception as e:
        print(f"\n发生错误: {e}")

if __name__ == "__main__":
    filter_apps()
