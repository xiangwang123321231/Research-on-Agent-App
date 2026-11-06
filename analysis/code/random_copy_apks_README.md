# 随机复制 APK

将 `random_copy_apks.py` 上传到服务器，在脚本所在目录运行：

```bash
python random_copy_apks.py
```

默认设置：

- 来源：`/home/ubuntu/workspace/jwang/metadata/apks`。
- 输出：`/home/ubuntu/workspace/jwang/metadata/apks_random_200`。
- 抽取 200 个 APK 文件，不放回随机抽样，种子为 `20261016`。
- 只复制，不移动或删除原文件；复制后核对源文件与副本的 SHA-256。
- 默认仅扫描来源目录的直属文件，扩展名大小写不敏感，不纳入符号链接。
- APK 不足 200 个或输出目录已存在时停止，不自动覆盖、删文件或更换种子。

如需包含子目录：

```bash
python random_copy_apks.py --recursive
```

自定义来源、输出和数量：

```bash
python random_copy_apks.py --source /path/to/apks --output /path/to/new_sample --count 200 --seed 20261016
```

输出目录必须位于来源目录之外。来源目录和文件清单不变、种子与运行环境一致时，选择结果可复现。不要根据结果好坏更换种子。

## 抽样记录与失败处理

输出中的 `sampling_manifest.json` 保存候选文件清单、随机种子、入选文件、复制状态、大小及校验后的 SHA-256。复制失败或中断时保留原抽样方案与已复制文件，不自动删掉或替换失败样本；清单状态为 `incomplete` 时不得视为完整抽样结果。重新运行不会覆盖已有目录，应先检查原因及原清单，而不是不断重抽。

## 用于独立验证时的边界

该脚本按文件随机抽样，不自动读取或排除 248 个训练/开发样本，也不按内容预先去重。若多个文件包含相同二进制，清单会记录不同内容哈希的数量，且不偷偷重新抽取。正式验证前应据 SHA-256 核对开发集重合与重复，并检查项目关联；不能直接把 200 个文件称为 200 个独立新应用。

本脚本不运行信号提取、训练或修改现有模型，不改动论文、标签或实验结果。

## 本地检查

在项目根目录运行以下命令，只使用临时测试文件，不读取研究 APK：

```bash
python -m unittest discover -s code/tests -p test_random_copy_apks.py -v
```
