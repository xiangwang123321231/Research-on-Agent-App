import csv
import os

INPUT_CSV = r"../data/latest_with-added-date.csv"
OUTPUT_CSV = r"../data/first_10_rows.csv"

def check_csv_columns(input_path, output_path):
    print(f"正在读取文件: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as infile, \
             open(output_path, 'w', encoding='utf-8', newline='') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            
            for i, row in enumerate(reader):
                # 只读取前10行 (包含表头的话是 i=0 到 i=9)
                if i >= 10:
                    break
                
                # 写入到输出文件
                writer.writerow(row)
                
                if i == 0:
                    print("\n--- 表头 (Header) ---")
                else:
                    print(f"\n--- 第 {i} 行数据 ---")
                
                # 打印第11列（索引为10）
                if len(row) > 10:
                    print(f"第 11 列内容 (index 10): {row[10]}")
                else:
                    print(f"警告：该行只有 {len(row)} 列，没有第 11 列")
                
                # 打印整行的前12列出来作为参考，避免太长
                preview_row = row[:12] if len(row) >= 12 else row
                print(f"局部数据预览 (前12列): {preview_row}")
                
        print(f"\n前十行数据已成功保存至: {output_path}")
                
    except FileNotFoundError:
        print(f"找不到文件: {input_path}")
    except Exception as e:
        print(f"读取时发生错误: {e}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.abspath(os.path.join(script_dir, INPUT_CSV))
    output_file = os.path.abspath(os.path.join(script_dir, OUTPUT_CSV))
    check_csv_columns(input_file, output_file)
