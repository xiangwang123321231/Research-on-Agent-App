import csv
import os
import time

# 输入和输出文件路径
INPUT_CSV = r"../data/latest_with-added-date.csv"
OUTPUT_CSV = r"../data/google_play_apps.csv"

def filter_google_play_apps(input_path, output_path, chunk_size=100000):
    """
    流式读取CSV文件，过滤出来源为Google Play的应用，并分批写入文件。
    根据AndroZoo文档提示，通过 `markets` 字段过滤，实际上表头显示 `markets` 在第12列（索引11）。
    """
    print(f"开始过滤 Google Play 应用...")
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_processed = 0
    matched_count = 0
    start_time = time.time()

    try:
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as infile, \
             open(output_path, 'w', encoding='utf-8', newline='') as outfile:
            
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            
            # 读取并写入表头
            try:
                header = next(reader)
                writer.writerow(header)
            except StopIteration:
                print("输入文件为空！")
                return

            batch_rows = []
            
            for row in reader:
                total_processed += 1
                
                # awk '$11 ~ /play\.google\.com/' 对应 Python 中索引 11（因为表头显示 markets 在第12列，索引为11）
                # 由于有些行可能因为格式不规范导致列数不足，需要检查长度
                if len(row) > 11 and 'play.google.com' in row[11]:
                    batch_rows.append(row)
                    matched_count += 1
                
                # 达到一定数量后批量写入并清空内存
                if len(batch_rows) >= chunk_size:
                    writer.writerows(batch_rows)
                    batch_rows.clear()
                    print(f"已处理 {total_processed} 行代码，当前共找到 {matched_count} 个 Google Play 应用...")
            
            # 写入剩余未写入的行
            if batch_rows:
                writer.writerows(batch_rows)
                
            elapsed_time = time.time() - start_time
            print("\n过滤完成！")
            print(f"总计处理行数：{total_processed}")
            print(f"Google Play 应用总数：{matched_count}")
            print(f"耗时：{elapsed_time:.2f} 秒")
            print(f"结果已保存至：{output_path}")

    except FileNotFoundError:
        print(f"未找到输入文件 {input_path}，请确认路径是否正确。")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    # 解析当前脚本所在的相对路径环境
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.abspath(os.path.join(script_dir, INPUT_CSV))
    output_file = os.path.abspath(os.path.join(script_dir, OUTPUT_CSV))
    
    filter_google_play_apps(input_file, output_file)
