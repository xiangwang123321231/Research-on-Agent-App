import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from datetime import datetime

def parse_date(date_str):
    if not date_str:
        return None
    date_str = str(date_str).strip()
    if '-' in date_str and len(date_str) >= 10:
        try:
            return datetime.strptime(date_str[:10], "%Y-%m-%d")
        except ValueError:
            pass
    MONTHS = {
        'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
        'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12
    }
    matches = re.findall(r'([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})', date_str)
    if matches:
        month_str, day_str, year_str = matches[0]
    else:
        matches = re.findall(r'(\d{1,2})\s+([A-Za-z]+)\.?\,?\s+(\d{4})', date_str)
        if matches:
            day_str, month_str, year_str = matches[0]
        else:
            return None
    month_prefix = month_str.lower()[:3]
    if month_prefix in MONTHS:
        try:
            return datetime(int(year_str), MONTHS[month_prefix], int(day_str))
        except ValueError:
            pass
    return None

def analyze_time_trend():
    input_file = "data/google_play_apps_final_dataset.jsonl"
    output_csv = "data/time_evolution_trend_monthly.csv"
    output_csv_yearly = "data/time_evolution_trend_yearly.csv"
    output_plot = "data/time_evolution_trend.png"

    records = []
    
    print(f"Reading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            docid = data.get('docid')
            
            # 【严格遵循官方指南】：时间分析只能使用 details.appDetails.uploadDate
            try:
                upload_date = data.get('details', {}).get('appDetails', {}).get('uploadDate')
                if upload_date:
                    records.append({'docid': docid, 'uploadDate': upload_date})
            except Exception as e:
                continue

    df = pd.DataFrame(records)
    print(f"Total records extracted: {len(df)}")
    
    # 使用自定义的 parse_date 处理复杂格式
    df['uploadDate_dt'] = df['uploadDate'].apply(parse_date)
    
    # 丢弃无法解析日期的脏数据
    df_clean = df.dropna(subset=['uploadDate_dt']).copy()
    print(f"Valid records after parsing date: {len(df_clean)}")

    # 提取年份和月份
    df_clean['Year'] = df_clean['uploadDate_dt'].dt.year
    df_clean['YearMonth'] = df_clean['uploadDate_dt'].dt.to_period('M')

    # 按月统计
    trend_monthly = df_clean.groupby('YearMonth').size().reset_index(name='AppCount')
    trend_monthly['YearMonth'] = trend_monthly['YearMonth'].astype(str)
    
    # 按年统计
    trend_yearly = df_clean.groupby('Year').size().reset_index(name='AppCount')

    # 保存统计结果到 CSV
    trend_monthly.to_csv(output_csv, index=False)
    trend_yearly.to_csv(output_csv_yearly, index=False)
    print(f"Saved monthly trend to {output_csv}")
    print(f"Saved yearly trend to {output_csv_yearly}")

    # 绘制折线图并保存
    plt.figure(figsize=(14, 6))
    plt.plot(trend_monthly['YearMonth'], trend_monthly['AppCount'], marker='o', linestyle='-', color='b')
    plt.xticks(rotation=90, fontsize=8) 
    plt.title('Time Evolution of AI Apps on Google Play (Based on uploadDate)')
    plt.xlabel('Publish Month (uploadDate)')
    plt.ylabel('Number of AI Apps Uploaded')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"Saved trend plot to {output_plot}")
    
    # 打印简要概览
    print("\n========= Yearly AI Apps Trend =========")
    print(trend_yearly.to_string(index=False))
    print("=========================================\n")

if __name__ == "__main__":
    analyze_time_trend()