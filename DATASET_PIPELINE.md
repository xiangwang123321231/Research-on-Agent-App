# Android AI 应用数据集处理与过滤白皮书

本文档记录了从 AndroZoo 原始 Google Play 元数据集提取并构建高质量「Android 端人工智能应用数据集」的完整流水线过程，以及最终数据集的具体数据格式特征。

总计处理记录数: 18452006

### 注意：对Googleplay 元数据处理的代码放在code文件夹下，data_set_code是对androzoo中的最新的latest.csv.gz文件的处理（用不上了）

## 一、 数据集筛选流程 (Pipeline)

我们的挖掘与清洗工作分为三个核心阶段，旨在从千万级庞大原始数据中提纯出有效的高质量 AI 应用。

### 第一阶段：时间过滤 (Time Filtering)
- **处理脚本**: `code/filter_post_2022_apps.py`
- **输入数据**: `androzoo-metadata/gp-metadata-full.jsonl.gz` (AndroZoo 原始超大压缩包)
- **输出数据**: `data/google_play_apps_post_2022.jsonl` [结果：总处理 18,452,006 条原始记录，最终保留 5,410,750 条符合时间的记录，缺少日期的记录数: 296132]
- **过滤逻辑**: 
  - 核心依据：生成式 AI 爆发的标志性节点。
  - 提取 `details.appDetails.uploadDate` 字段，利用定制正则表达式兼容十余种复杂日期格式（如 `Mar 26, 2020`, `26 Aug. 2023` 等）。
  - **只保留 `uploadDate >= 2022年11月1日` 的应用记录**。

### 第二阶段：最新版本瘦身与去重 (Version Deduping)
- **处理脚本**: `code/filter_latest_version.py`
- **输入数据**: `data/google_play_apps_post_2022.jsonl`
- **输出数据**: `data/google_play_apps_latest_post_2022.jsonl` [结果：输入 5,410,750 条记录，去重后保留 1,210,816 个独立最新版本应用]
- **过滤逻辑**:
  - 原始数据中同一个应用（`docid`）可能存在十几个历史版本的记录。
  - 采用极其优秀的 **Two-Pass (双趟扫描) 算法**：
    1. 第一趟扫描建立 `docid : max(versionCode)` 的全局映射表。
    2. 第二趟扫描筛选出匹配最大版本的记录，并且落盘后立即剔除字典键，彻底避免了同一版本号存在多条冗余记录的缺陷。
  - **只保留每个包名（App）在清洗时间段内的唯一最新版本**。

### 第三阶段：AI 特征深度挖掘与排雷 (AI Feature Extraction)
- **处理脚本**: `code/filter_ai_apps.py`
- **输入数据**: `data/google_play_apps_latest_post_2022.jsonl`
- **输出数据**: `data/google_play_apps_ai_filtered.jsonl` [结果：扫描 1,210,816 个独立应用，最终挖掘出 40,571 个高质量 AI 应用]
- **过滤逻辑**:
  - 联合检索核心文本：综合 `descriptionHtml` 与 `recentChangesHtml`。
  - **深度数据清洗**：反转义 HTML 实体、擦除 `<br>`/`<b>` 全量 HTML 标签、格式化冗余空格（防止脏数据带来类似 `<br>AI` 的连字误判）。
  - **智能召回判定机制**：
    - **严格校验 (Strict Matching)**: 引入“否定环视 (Negative Lookaround)” 正则 `(?<![a-zA-Z])(AI|A\.I\.)(?![a-zA-Z])` 提取独立的 AI 缩写，完美杜绝 `AIR`, `AIM`, `AID` 等子串误杀。
    - **宽容语义 (Broad Matching)**: 覆盖 GPT 全系版本后缀 (包括 3.5, 4o, turbo)、开源大模型 (Llama, DeepSeek, Qwen)、垂类应用词汇 (Avatar, Voice cloning, AI enhancer) 及生态组件库等底层代名词识别。
  - **“传统游戏AI”剔除 (Negative Filtering)**: 检测到诸如 `cpu opponent`, `ai enemy`, `vs ai`, `play against ai` 等单机游戏人机对战特征，直接静默抛弃，保障生成式智能应用数据的纯正性。

---

## 二、 最终数据记录格式 (Data Schema)

在最终产出的 `google_play_apps_ai_filtered.jsonl` 数据集中，每一行均为一个独立且合法的 JSON 对象。主要包含以下极具价值的高维数据特征：

### 1. 应用基础标识 (App Basics)
- `docid` / `packageName`: 应用的唯一包名（例如 `com.openai.chatgpt`）。
- `title`: 应用商店展示名称。
- `descriptionShort`: 一句话简短描述。
- `descriptionHtml`: 应用在商店的完整长篇 HTML 描述介绍。
- `creator`: 开发者或发行公司名称。

### 2. 版本与技术详情 (Version & Updates) *位于 `details.appDetails`*
- `uploadDate`: 提取的最新版本上传/更新日期。
- `versionCode`: 内部数字版本号。
- `versionString`: 对外展示的分发版本号（如 "1.0.5"）。
- `recentChangesHtml`: 最新版本的更新日志说明 (What's new)。
- `permission`: 应用申请的 Android 权限列表集合（**隐私合规与静态分析的核心字段**）。
- `installationSize`: App 安装包大小。

### 3. 数据指标与受众 (Metrics & Ratings)
- `aggregateRating.starRating`: 应用当前的星级评分（满分 5.0）。
- `aggregateRating.ratingsCount` / `commentCount`: 总评分人数和用户评论数。
- `aggregateRating.oneStarRatings` ~ `fiveStarRatings`: 1星到5星详细的分布人数。
- `details.appDetails.numDownloads`: 商店公开的下载量量级范围（如 "1,000,000+ downloads"）。

### 4. 商业与生态信息 (Business & Dev Info)
- `offer`: 价格信息列表（包含价位、是否免费优惠以及币种）。
- `details.appDetails.containsAds`: 标记是否内嵌广告。
- `details.appDetails.developerEmail` / `developerWebsite` / `developerAddress`: 开发者公开的联络方式、官网主页与实体地址。
- `privacyPolicyUrl`: 产品隐私保护协议链接。

### 5. 附加挖掘凭证 (AI Analysis Tags) *数据处理自动特供*
在底层 JSON 的字典外部，由我们的第三阶段挖掘脚本额外附加了一层特殊的 AI 定位诊断结果，便于后续特征投喂：
```json
"ai_analysis_tags": {
    "is_ai": true,
    "strict_ai_matched": true,   // true/false 是否命中极其严格大写独立 AI/ A.I. 匹配
    "broad_ai_matched": false    // true/false 是否命中由于开源模型名称/长短语/细分智能赛道匹配而触发的捞取
}
```

