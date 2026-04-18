import requests
import os
import time

# 配置
API_KEY = os.getenv("API_KEY")  # AndroZoo API Key
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "androzoo-metadata")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gp-metadata-full.jsonl.gz")
URL = "https://androzoo.uni.lu/api/get_gp_metadata_file/full"

chunk_size = 1024 * 1024 * 2 # 2MB 每次
max_retries = 20 # 最大断点重试次数
retry_delay = 5  # 重试间隔等待秒数

def download_with_resume():
    downloaded = 0
    if os.path.exists(OUTPUT_FILE):
        downloaded = os.path.getsize(OUTPUT_FILE)
        
    for attempt in range(max_retries):
        try:
            headers = {}
            if downloaded > 0:
                headers["Range"] = f"bytes={downloaded}-"
                print(f"\n发现已有文件，尝试从 {downloaded / (1024*1024):.2f} MB 处恢复下载 (尝试 {attempt + 1}/{max_retries})...")
            else:
                print(f"\n开始下载 {OUTPUT_FILE} (尝试 {attempt + 1}/{max_retries})...")

            response = requests.get(URL, params={"apikey": API_KEY}, headers=headers, stream=True, timeout=30)
            
            # 如果服务器返回416，说明 Range 请求不合法，可能已经整个文件下完了
            if response.status_code == 416:
                print("\n服务器返回 416 (Range Not Satisfiable)，文件可能已下载完毕。")
                return

            # 如果我们传了Range但服务器返回200，说明它不支持断点续传，给我们从头发了
            if downloaded > 0 and response.status_code == 200:
                print("\n服务器不支持断点续传，将重新开始下载...")
                downloaded = 0
                mode = "wb"
            else:
                response.raise_for_status()
                mode = "ab" if downloaded > 0 else "wb"

            total_size_header = response.headers.get("content-length")
            content_range = response.headers.get("content-range")
            if content_range:
                # 例如返回 'bytes 100-200/1000'
                total_size = int(content_range.split('/')[-1])
            elif total_size_header:
                total_size = downloaded + int(total_size_header)
            else:
                total_size = 0
                
            last_print_time = time.time()
            
            # 开始分块下载并写入文件
            with open(OUTPUT_FILE, mode) as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        current_time = time.time()
                        if current_time - last_print_time > 0.5:
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                print(f"\r下载进度: {progress:.1f}% ({downloaded / (1024*1024):.2f}/{total_size / (1024*1024):.2f} MB)", end="")
                            else:
                                print(f"\r已下载: {downloaded / (1024*1024):.2f} MB", end="")
                            last_print_time = current_time
            
            print(f"\n下载完成或当前重试结束。最终大小: {downloaded / (1024*1024):.2f} MB")
            return # 没有发生异常说明流安全读取到了最后

        except requests.exceptions.ChunkedEncodingError as e:
            print(f"\n[网络中断] ChunkedEncodingError: {e}")
        except requests.exceptions.RequestException as e:
            print(f"\n[网络错误]: {e}")
            
        if attempt < max_retries - 1:
            print(f"等待 {retry_delay} 秒后准备重试...")
            time.sleep(retry_delay)
        else:
            print("\n达到最大重试次数，下载中止。你再次运行此脚本仍可断点续传。")
            raise

if __name__ == "__main__":
    download_with_resume()