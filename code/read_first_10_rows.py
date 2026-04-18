import gzip
import os

# 配置路径
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_file = os.path.join(base_dir, "androzoo-metadata", "gp-metadata-full.jsonl.gz")
output_dir = os.path.join(base_dir, "data")
output_file = os.path.join(output_dir, "first_10_rows.jsonl") # 保持原有的 JSONL 格式

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

def extract_first_n_lines(n=10):
    print(f"开始读取 {input_file} 的前 {n} 行...")
    
    if not os.path.exists(input_file):
        print(f"错误：找不到文件 {input_file}。请确保下载已完成。")
        return

    try:
        # 使用 gzip 以文本模式 ('rt') 打开压缩文件
        with gzip.open(input_file, 'rt', encoding='utf-8') as f_in:
            with open(output_file, 'w', encoding='utf-8') as f_out:
                lines_read = 0
                for line in f_in:
                    f_out.write(line)
                    lines_read += 1
                    if lines_read >= n:
                        break
        
        print(f"成功提取前 {lines_read} 行！")
        print(f"文件已保存至: {output_file}")
        
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

if __name__ == "__main__":
    extract_first_n_lines(10)
