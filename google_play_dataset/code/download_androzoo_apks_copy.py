import os
import random
import shutil
import sys
import threading
import time
import zipfile

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# 路径配置
# ============================================================
# 服务器部署结构:
#   yzhou/code/   ← 本脚本 + final_sha256s_v2.txt
#   yzhou/apks/   ← 下载的 APK 输出目录（脚本自动创建）
# 注意: 不要用 dirname(dirname(...)) 上跳两级 —— 那是"本地两级目录结构"的
# 假设，在服务器平铺目录下会找错位置。
code_dir = os.path.dirname(os.path.abspath(__file__))   # yzhou/code
yzhou_dir = os.path.dirname(code_dir)                    # yzhou

# 输入: extract_sha256s.py 输出的纯文本 SHA256 列表（与脚本同目录）
sha256_file = os.path.join(code_dir, "google_play_apps_llm_sha256s_new_only_848.txt")
# 输出: APK 下载目录（yzhou/apks）
download_dir = os.path.join(yzhou_dir, "apks")

# API Key: 优先从环境变量读取，否则使用默认值
API_KEY = os.environ.get("ANDROZOO_API_KEY", "d970447e47bbec33210fa8f17d5626c8e50999bd7f537db26ae791f14d98a07e")

# 下载限制: None = 全量下载，设为数字则随机抽取 N 个
DOWNLOAD_LIMIT = None

# 并发下载线程数:
#   1 = 串行（最稳，AndroZoo 限流时重试更从容）
#   4~8 = 加速。18,199 个全量下载，串行预计 20~50 小时，并发 4 可缩至 5~15 小时。
# 如果大量出现"被限流"日志，把并发调回 1~2。
CONCURRENCY = 4

# 失败重试次数
MAX_RETRIES = 3

# 单个文件最长下载时间（秒）: 防止慢网速/服务器静默断连（TCP 半开连接）时
# 线程永久挂死。慢网速下宁可超时失败重下，也不要 4 个线程全被僵尸连接占满。
MAX_FILE_SECONDS = 3600

# 磁盘剩余空间低于该值（字节）直接报错退出，避免写满磁盘
MIN_FREE_DISK_BYTES = 20 * 1024 * 1024 * 1024  # 20GB


def load_sha256s(file_path, limit=None):
    """从纯文本文件读取 SHA256 列表（每行一个），去重。
    设置 limit 则随机抽取指定数量。"""
    print(f"正在读取 SHA256 列表: {file_path}", flush=True)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"SHA256 文件不存在: {file_path}")

    sha256_set = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            sha = line.strip()
            if sha and len(sha) == 64:  # SHA256 固定 64 个十六进制字符
                sha256_set.add(sha.upper())

    sha256_list = sorted(sha256_set)
    print(f"  有效 SHA256 总数: {len(sha256_list)}", flush=True)

    if limit is not None and limit < len(sha256_list):
        sha256_list = random.sample(sha256_list, limit)
        print(f"  随机抽取: {limit} 个", flush=True)

    return sha256_list


# 全局 Session: 复用 HTTP 连接池，避免每次请求重新建连
session = requests.Session()
session.headers.update({"User-Agent": "androzoo-bulk-downloader/1.0 (research pipeline; +https://androzoo.uni.lu)"})
# 默认直连（忽略系统/环境变量代理）: 环境里残留的 https_proxy 指向失效代理时
# 会让所有请求 TLS 握手失败。需要走代理时: USE_PROXY=1 python3 -u script.py
session.trust_env = os.environ.get("USE_PROXY") == "1"

# API Key 失效时全局停止（避免 18K 个全部白跑）
abort_event = threading.Event()


def download_one(sha256):
    """下载单个 APK。线程安全（互不依赖），返回 (status, sha256, reason, nbytes)。
    status: ok / skip / fail
    reason: 失败原因（限流/404/网络/超时/校验/abort），用于诊断
    nbytes: 本次成功下载的字节数（失败为 0）
    """
    apk_path = os.path.join(download_dir, f"{sha256}.apk")
    temp_path = f"{apk_path}.part"

    # Key 失效后其余任务直接跳过（快速退出）
    if abort_event.is_set():
        return "skip", sha256, "abort", 0

    # 跳过已下载完成的文件（用 zipfile 完整校验，避免截断文件被误判为已完成）
    if os.path.exists(apk_path) and os.path.getsize(apk_path) > 0:
        if zipfile.is_zipfile(apk_path):
            return "skip", sha256, "already_done", 0
        else:
            os.remove(apk_path)

    # 清理可能残留的临时文件
    if os.path.exists(temp_path):
        os.remove(temp_path)

    url = f"https://androzoo.uni.lu/api/download?apikey={API_KEY}&sha256={sha256}"
    start = time.time()

    for attempt in range(1, MAX_RETRIES + 1):
        # abort 后立即退出（401/403 或 Ctrl+C 后，等待中的线程不再发起新请求）
        if abort_event.is_set():
            return "skip", sha256, "abort", 0
        # 重试前整体超时检查：重试累计超 2 小时也放弃
        if time.time() - start > 2 * MAX_FILE_SECONDS:
            return "fail", sha256, "timeout_total", 0
        try:
            resp = session.get(url, stream=True, timeout=120)

            if resp.status_code == 200:
                content_type = resp.headers.get("Content-Type", "")
                if "text/html" in content_type:
                    # AndroZoo 限流时返回 HTML 错误页；随机抖动避免并发重试风暴
                    resp.close()
                    if attempt == MAX_RETRIES:
                        return "fail", sha256, "rate_limited", 0
                    time.sleep(30 * attempt + random.uniform(0, 10))
                    continue

                nbytes = 0
                with open(temp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        # abort 后放弃在途下载（不再等慢网速写完整个文件）
                        if abort_event.is_set():
                            resp.close()
                            os.remove(temp_path)
                            return "skip", sha256, "abort", 0
                        # 单文件下载时限：防止僵尸连接占死线程
                        if time.time() - start > MAX_FILE_SECONDS:
                            resp.close()
                            os.remove(temp_path)
                            return "fail", sha256, "timeout", 0
                        if chunk:
                            f.write(chunk)
                            nbytes += len(chunk)

                # 校验：完整 zipfile 校验，避免截断/损坏文件混入
                if not zipfile.is_zipfile(temp_path):
                    os.remove(temp_path)
                    return "fail", sha256, "bad_zip", 0

                os.replace(temp_path, apk_path)
                return "ok", sha256, "", nbytes

            elif resp.status_code == 404:
                resp.close()
                return "fail", sha256, "not_found", 0

            elif resp.status_code in (401, 403):
                resp.close()
                abort_event.set()
                print(f"  API Key 无效 ({resp.status_code})，请检查 ANDROZOO_API_KEY，停止下载", flush=True)
                return "fail", sha256, "auth_fail", 0

            elif resp.status_code == 429:
                # 限流状态码：按限流节奏退避（比普通错误更长的等待）
                resp.close()
                if attempt == MAX_RETRIES:
                    return "fail", sha256, "rate_limited", 0
                time.sleep(30 * attempt + random.uniform(0, 10))
                continue

            else:
                resp.close()  # 非 200 响应不关连接会泄漏
                if attempt == MAX_RETRIES:
                    return "fail", sha256, f"http_{resp.status_code}", 0
                time.sleep(min(3 * attempt, 10))

        except requests.exceptions.RequestException as e:
            if isinstance(e, requests.exceptions.SSLError):
                # SSL 握手失败多为连接池里复用的旧连接已被服务器静默关闭
                # （长时间运行后常见）。清空连接池，让所有线程下次重新 TLS 握手。
                # session.close() 线程安全：只会丢弃空闲连接，不影响在途请求。
                session.close()
            if attempt == MAX_RETRIES:
                if isinstance(e, requests.exceptions.SSLError):
                    # 诊断: 打印具体 SSL 错误（证书验证失败/连接被重置/EOF 等）
                    print(f"    SSL 失败详情: {type(e).__name__}: {e}", flush=True)
                return "fail", sha256, f"net_{type(e).__name__}", 0
            # 网络异常退避: 10s/40s/90s 指数增长，给网络/服务器恢复时间（慢网速场景）
            time.sleep(min(10 * attempt * attempt, 120))

        except Exception as e:
            return "fail", sha256, f"exc_{type(e).__name__}", 0

    return "fail", sha256, "retries_exhausted", 0


def fmt_size(nbytes):
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024


def fmt_duration(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def download_apks(sha256_list):
    """批量下载 APK，支持断点续传（跳过已存在的文件）和失败重试。"""
    os.makedirs(download_dir, exist_ok=True)

    # 磁盘空间自检：590GB 总量场景，写满磁盘会让大批 .part 失败
    free_bytes = shutil.disk_usage(download_dir).free
    if free_bytes < MIN_FREE_DISK_BYTES:
        print(f"错误: 磁盘剩余空间不足 ({fmt_size(free_bytes)})，低于最低要求 {fmt_size(MIN_FREE_DISK_BYTES)}，退出", flush=True)
        sys.exit(1)
    print(f"磁盘剩余空间: {fmt_size(free_bytes)}", flush=True)

    total = len(sha256_list)
    stats = {"ok": 0, "skip": 0, "fail": 0}
    failed = []  # (sha256, reason)
    total_bytes = 0
    lock = threading.Lock()

    print(f"共 {total} 个 APK 待下载，保存至: {download_dir}", flush=True)
    print(f"并发线程数: {CONCURRENCY}", flush=True)
    print("=" * 60, flush=True)

    start_time = time.time()
    ex = ThreadPoolExecutor(max_workers=CONCURRENCY)
    try:
        futures = [ex.submit(download_one, sha) for sha in sha256_list]
        for done, fut in enumerate(as_completed(futures), 1):
            if abort_event.is_set():
                break  # Key 失效：放弃剩余任务
            status, sha, reason, nbytes = fut.result()
            with lock:
                stats[status] += 1
                total_bytes += nbytes
                if status == "fail":
                    failed.append((sha, reason))
            if status == "ok":
                print(f"[{done}/{total}] 成功: {sha}", flush=True)
            elif status == "skip":
                print(f"[{done}/{total}] 跳过（已存在）: {sha}", flush=True)
            else:
                print(f"[{done}/{total}] 失败({reason}): {sha}", flush=True)

            # 每 25 个打印一次速度与剩余时间估计
            if done % 25 == 0:
                elapsed = time.time() - start_time
                speed = total_bytes / elapsed if elapsed > 0 else 0
                # 剩余量减去已 skip 数: 后面的文件不会再 skip，重跑续传场景 ETA 更准
                remaining = total - done - stats["skip"]
                eta = (remaining * (total_bytes / stats["ok"])) / speed if speed > 0 and stats["ok"] > 0 else 0
                print(
                    f"  进度 {done}/{total} | 已下载 {fmt_size(total_bytes)} | "
                    f"均速 {fmt_size(speed)}/s | 预计剩余 {fmt_duration(eta)}",
                    flush=True,
                )
    except KeyboardInterrupt:
        # Ctrl+C / nohup kill: 立即放弃未开始的任务并退出，.part 残留下次重跑自动清理
        abort_event.set()
        ex.shutdown(wait=False, cancel_futures=True)
        print("\n收到中断，已停止。已完成的文件保留，可直接重跑续传。", flush=True)
        os._exit(1)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

    # 失败列表写入文件（带原因），方便重跑时单独补下或诊断
    if failed:
        failed_file = os.path.join(download_dir, "failed_sha256s.txt")
        with open(failed_file, "w", encoding="utf-8") as f:
            f.write("\n".join(f"{sha} {reason}" for sha, reason in failed))
        print(f"失败 {len(failed)} 个，列表已保存: {failed_file}", flush=True)

    # 统计
    print(flush=True)
    print("=" * 60, flush=True)
    print("下载任务结束", flush=True)
    print("=" * 60, flush=True)
    print(f"  总计:       {total}", flush=True)
    print(f"  成功:       {stats['ok']}", flush=True)
    print(f"  跳过:       {stats['skip']}", flush=True)
    print(f"  失败:       {stats['fail']}", flush=True)
    print(f"  已下载数据: {fmt_size(total_bytes)}", flush=True)
    print(f"  耗时:       {fmt_duration(time.time() - start_time)}", flush=True)
    print(f"  保存目录:   {download_dir}", flush=True)


def main():
    if API_KEY == "YOUR_API_KEY_HERE":
        print("错误: 请设置 ANDROZOO_API_KEY 环境变量，或在脚本中填入 API Key")
        sys.exit(1)

    sha256_list = load_sha256s(sha256_file, limit=DOWNLOAD_LIMIT)

    if not sha256_list:
        print("没有可下载的 SHA256，退出。")
        return

    download_apks(sha256_list)


if __name__ == "__main__":
    main()
