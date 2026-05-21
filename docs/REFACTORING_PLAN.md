# patchTST_transformer 轻量重构方案

## 1. 重构方向

这次重构不走复杂包结构，不把代码拆成很多模块。项目继续保留“一个主 Python 文件”的形态，但要把混乱点收拢：

1. 所有输入路径和输出路径都支持绝对路径。
2. 数据窗口序列长度、时间间隔、预测目标列都可以通过配置随意更改。
3. 一个任务只生成一个任务目录。
4. 一个任务目录里包含该任务的模型、日志、配置快照、训练曲线、评估指标和预测结果。
5. 不再把日志、checkpoint、results、tmp config 分散到多个顶层目录。
6. 任务目录名由使用的 YAML 文件名决定，例如 `skippd_train.yaml` 对应 `outputs/skippd_train_YYYYMMDD_HHMMSS/`。
7. 保留单文件代码，但在文件内部按清晰 section 组织。

核心思路是：**不做重型工程化拆包，只做路径、配置、输出结构和单文件内部结构的整理。**

## 2. 当前问题

当前目录最大的问题不是“代码必须拆很多文件”，而是实验产物和配置路径太散：

- `train_itransformer.py` 同时做训练和评估，可以保留，但内部需要重新分区。
- `checkpoint/`、`checkpoints/`、`results/`、`log/`、`batch_logs/`、`tmp_eval_configs/` 分散保存不同产物。
- 一个任务会产生多个日志文件、多个结果目录，后续很难追踪。
- `train.yaml`、`eval.yaml` 中有大量服务器绝对路径，迁移时容易出错。
- 当前配置里的 `sequence_length`、`time_interval_minutes`、`target_horizon_minutes`、`target_column` 已经存在，但使用规则需要更明确。
- 批量评估脚本会生成很多临时 YAML，导致 `tmp_eval_configs/` 膨胀。

## 3. 保留单文件，但重排内部结构

建议继续使用一个主文件，例如：

```text
train_itransformer.py
```

但文件内部按下面顺序组织：

```text
1. imports 和全局兼容逻辑
2. 配置读取与配置校验
3. 路径解析与任务目录创建
4. 日志系统
5. 设备选择
6. 数据列解析
7. 数据加载与时间戳处理
8. 归一化与反归一化
9. 滑动窗口构造
10. 模型定义
11. 指标计算
12. 训练流程
13. 评估流程
14. 批量评估流程
15. main 入口
```

这样仍然是一个文件，但阅读时不会像现在一样混在一起。

## 4. 新的目录约定

推荐根目录只保留这些长期内容：

```text
patchTST_transformer/
├── train_itransformer.py
├── train.yaml
├── eval.yaml
├── batch_eval.yaml
├── run_train.sh
├── run_eval.sh
├── batch_eval.sh
├── data/
├── outputs/
└── docs/
```

其中 `outputs/` 是新的统一输出根目录。所有任务输出都放这里。

示例：

```text
outputs/
├── skippd_train_20260521_153000/
│   ├── task.log
│   ├── config.yaml
│   ├── model_best.pth
│   ├── norm_stats.npz
│   ├── train_loss.txt
│   ├── val_loss.txt
│   └── train_val_loss.png
├── folmos_eval_20260521_160000/
│   ├── task.log
│   ├── config.yaml
│   ├── test_metrics.txt
│   ├── test_predictions.csv
│   └── test_results.png
└── skippd_batch_eval_20260521_170000/
    ├── task.log
    ├── config.yaml
    ├── item_001_random_all_0p2/
    │   ├── test_metrics.txt
    │   ├── test_predictions.csv
    │   └── test_results.png
    └── item_002_random_all_0p4/
        ├── test_metrics.txt
        ├── test_predictions.csv
        └── test_results.png
```

## 5. 路径设计

### 5.1 所有输入输出都支持绝对路径

配置中所有路径字段都允许绝对路径。

如果传绝对路径，代码直接使用。

如果传相对路径，代码基于项目根目录解析。

建议统一路径字段：

```yaml
paths:
  project_root: C:/Users/18173/Desktop/python/patchTST_transformer
  output_root: C:/Users/18173/Desktop/python/patchTST_transformer/outputs

  train_csv: C:/absolute/path/train.csv
  val_csv: C:/absolute/path/val.csv
  test_csv: C:/absolute/path/test.csv

  model_path: C:/absolute/path/model_best.pth
  norm_stats_path: C:/absolute/path/norm_stats.npz
```

说明：

- `project_root` 可选。没有时默认取 `train_itransformer.py` 所在目录。
- `output_root` 可选。没有时默认是 `${project_root}/outputs`。
- 不再需要 `task_name`。任务名前缀由 `CONFIG_FILE` 指向的 YAML 文件名决定。
- `train_csv`、`val_csv`、`test_csv` 优先级最高。
- 兼容旧写法 `data_dir + train_file/val_file/test_file`，但新方案推荐直接写完整路径。
- `model_path` 和 `norm_stats_path` 在评估时可以指向任意已有任务目录里的模型和归一化文件。

### 5.2 每次运行创建一个任务目录

任务目录命名规则：

```text
{yaml文件名去掉扩展名}_{YYYYMMDD_HHMMSS}
```

例如：

```text
CONFIG_FILE=C:/configs/skippd_train.yaml
=> outputs/skippd_train_20260521_153000

CONFIG_FILE=C:/configs/folmos_eval.yaml
=> outputs/folmos_eval_20260521_160000
```

如果同一个 YAML 连续运行多次，时间戳会区分不同任务目录。

不建议再使用 `paths.output_dir` 指定完整任务目录，因为这会绕过“按 YAML 文件名命名任务”的规则。需要更换输出位置时，只改 `paths.output_root`。

## 6. 配置设计

### 6.1 训练配置示例

```yaml
mode: train
seed: 42

paths:
  output_root: C:/Users/18173/Desktop/python/patchTST_transformer/outputs
  train_csv: C:/data/Folmos/5min/train_data.csv
  val_csv: C:/data/Folmos/5min/val_data.csv

runtime:
  device: auto
  npu_device_index: 0
  cuda_device_index: 0

data:
  timestamp_col: timestamp
  timestamp_format: "%Y-%m-%d %H:%M:%S"
  feature_columns_csv: B(ghi_kt|5min),B(ghi_kt|10min),irr_ghi,irr_dni,irr_dhi
  target_column: target_ghi_15min
  sequence_length: 24
  time_interval_minutes: 5
  target_horizon_minutes: 15

model:
  hidden_size: 128
  depth: 3
  dim_head: 32
  heads: 4
  ff_mult: 4
  dropout: 0.1
  num_mem_tokens: 4
  use_reversible_instance_norm: true
  reversible_instance_norm_affine: true
  flash_attn: true

train:
  batch_size: 256
  shuffle: true
  num_workers: 0
  lr: 0.001
  epochs: 150
  early_stop_patience: 30
  save_epochs: [1, 2, 3, 4, 5, 6]
```

### 6.2 评估配置示例

```yaml
mode: eval
seed: 42

paths:
  output_root: C:/Users/18173/Desktop/python/patchTST_transformer/outputs
  test_csv: C:/data/Folmos/5min/test_data.csv
  val_csv: C:/data/Folmos/5min/val_data.csv
  model_path: C:/Users/18173/Desktop/python/patchTST_transformer/outputs/skippd_train_20260521_153000/model_best.pth
  norm_stats_path: C:/Users/18173/Desktop/python/patchTST_transformer/outputs/skippd_train_20260521_153000/norm_stats.npz

runtime:
  device: auto
  npu_device_index: 0
  cuda_device_index: 0

data:
  timestamp_col: timestamp
  timestamp_format: "%Y-%m-%d %H:%M:%S"
  feature_columns_csv: B(ghi_kt|5min),B(ghi_kt|10min),irr_ghi,irr_dni,irr_dhi
  target_column: target_ghi_15min
  sequence_length: 24
  time_interval_minutes: 5
  target_horizon_minutes: 15

model:
  hidden_size: 128
  depth: 3
  dim_head: 32
  heads: 4
  ff_mult: 4
  dropout: 0.1
  num_mem_tokens: 4
  use_reversible_instance_norm: true
  reversible_instance_norm_affine: true
  flash_attn: true

eval:
  batch_size: 256
  num_workers: 0
  plot_points: 300
  result_csv_file: test_predictions.csv
  ws_alpha: 0.1
```

## 7. 数据窗口、时间间隔、预测目标

这三个维度全部走配置：

```yaml
data:
  sequence_length: 24
  time_interval_minutes: 5
  target_horizon_minutes: 15
  target_column: target_ghi_15min
```

含义：

- `sequence_length`: 输入窗口长度，也就是每个样本使用多少个历史时间步。
- `time_interval_minutes`: 数据本身的时间间隔，用于记录实验含义和校验时间连续性。
- `target_horizon_minutes`: 预测未来多少分钟，用于导出 `target_timestamp`。
- `target_column`: 真正训练/评估使用的目标列。

重要规则：

- 模型训练实际使用 `target_column`。
- `target_horizon_minutes` 不自动推断目标列，只用于输出结果里的目标时间戳。
- 如果要从 15 分钟预测改成 30 分钟预测，需要同时改：

```yaml
target_column: target_ghi_30min
target_horizon_minutes: 30
```

窗口样本生成规则保持为：

```text
X[i] = 第 i 到 i + sequence_length - 1 行的 feature_columns
y[i] = 第 i + sequence_length - 1 行的 target_column
```

如果 CSV 里的 `target_column` 已经提前按未来 horizon 生成好，那么这个逻辑是正确的。

## 8. 可选的时间连续性校验

建议增加一个配置项：

```yaml
data:
  validate_time_interval: true
```

如果开启，就检查相邻时间戳是否等于 `time_interval_minutes`。

发现断点时有三种策略：

```yaml
data:
  time_gap_policy: warn
```

可选：

- `ignore`: 不检查。
- `warn`: 只写入日志，不中断。
- `error`: 发现不连续直接报错。

第一轮建议默认 `warn`，避免历史数据因为少量缺口跑不起来。

## 9. 一个任务一个日志

新的日志规则：

```text
每次运行只写一个日志文件:
  {task_dir}/task.log
```

不再生成：

- `log/train_xxx.log`
- `log/eval_xxx.log`
- `batch_logs/batch_eval_xxx.log`
- 多个 `.pid` 文件

Python 内部使用 `logging` 同时输出到控制台和 `{task_dir}/task.log`。

Shell 脚本如果还保留，只负责设置 `CONFIG_FILE` 并启动 Python，不再额外重定向生成第二份日志。

示例：

```text
outputs/skippd_train_20260521_153000/task.log
```

## 10. 一个任务目录保存所有产物

### 10.1 训练任务目录

训练输出统一为：

```text
outputs/{yaml_stem}_{time}/
├── task.log
├── config.yaml
├── model_best.pth
├── norm_stats.npz
├── train_loss.txt
├── val_loss.txt
├── train_val_loss.png
├── checkpoint_1.pth
├── checkpoint_2.pth
└── checkpoint_6.pth
```

说明：

- `config.yaml` 是运行时配置快照。
- `model_best.pth` 是最佳验证集模型。
- `checkpoint_{epoch}.pth` 只在配置了 `save_epochs` 时保存。
- 不再写入 `checkpoints/experiment_name/`。

### 10.2 评估任务目录

评估输出统一为：

```text
outputs/{yaml_stem}_{time}/
├── task.log
├── config.yaml
├── test_metrics.txt
├── test_predictions.csv
└── test_results.png
```

说明：

- `model_path` 和 `norm_stats_path` 可以指向任意训练任务目录，例如 `outputs/skippd_train_20260521_153000/`。
- 当前评估任务只保存评估产物，不复制大模型文件。
- 如果需要可复现实验，也可以增加：

```yaml
eval:
  copy_model_to_task_dir: false
```

默认不复制，避免浪费空间。

### 10.3 批量评估任务目录

批量评估不再生成一堆 `tmp_eval_configs/*.yaml`。

统一结构：

```text
outputs/{yaml_stem}_{time}/
├── task.log
├── config.yaml
├── summary.csv
├── item_001_dataset_a/
│   ├── test_metrics.txt
│   ├── test_predictions.csv
│   └── test_results.png
├── item_002_dataset_b/
│   ├── test_metrics.txt
│   ├── test_predictions.csv
│   └── test_results.png
└── item_003_dataset_c/
    ├── test_metrics.txt
    ├── test_predictions.csv
    └── test_results.png
```

`task.log` 记录所有子任务过程。

`summary.csv` 汇总每个子任务的指标。

## 11. 批量评估配置建议

新增 `batch_eval.yaml`：

```yaml
mode: batch_eval
seed: 42

paths:
  output_root: C:/Users/18173/Desktop/python/patchTST_transformer/outputs
  batch_data_dir: C:/data/Missing-data/fixed/skippd_test_data
  model_path: C:/Users/18173/Desktop/python/patchTST_transformer/outputs/skippd_train_20260521_153000/model_best.pth
  norm_stats_path: C:/Users/18173/Desktop/python/patchTST_transformer/outputs/skippd_train_20260521_153000/norm_stats.npz
  val_csv: C:/data/Folmos/5min/val_data.csv

batch_eval:
  pattern: "*.csv"
  item_name_from_filename: true

data:
  timestamp_col: timestamp
  timestamp_format: "%Y-%m-%d %H:%M:%S"
  feature_columns_csv: B(ghi_kt|5min),B(ghi_kt|10min),irr_ghi,irr_dni,irr_dhi
  target_column: target_ghi_15min
  sequence_length: 24
  time_interval_minutes: 5
  target_horizon_minutes: 15

eval:
  batch_size: 256
  num_workers: 0
  plot_points: 300
  ws_alpha: 0.1
```

批量逻辑：

1. 扫描 `paths.batch_data_dir` 下匹配 `batch_eval.pattern` 的 CSV。
2. 为每个 CSV 创建一个子目录。
3. 所有子任务共用同一个模型和归一化统计。
4. 所有日志写到批量任务根目录的 `task.log`。
5. 每个 CSV 的指标写入自己的子目录，同时汇总到 `summary.csv`。

## 12. Shell 脚本简化

Shell 脚本只做启动，不再管日志文件名。

`run_train.sh`：

```bash
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/train.yaml}"

python -u "${SCRIPT_DIR}/train_itransformer.py"
```

`run_eval.sh`：

```bash
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/eval.yaml}"

python -u "${SCRIPT_DIR}/train_itransformer.py"
```

如果需要后台运行，建议通过外部命令控制，例如：

```bash
nohup bash run_train.sh &
```

但 Python 自己仍然只写 `{task_dir}/task.log`。

## 13. 单文件内建议保留和调整的函数

### 13.1 保留但整理

这些函数可以继续保留在同一个文件里：

- `set_seed`
- `select_device`
- `parse_feature_columns`
- `load_single_csv`
- `fit_normalization_stats`
- `apply_normalization`
- `unnormalize_data`
- `create_sliding_window`
- `prepare_split_dataset`
- `eval_metrics`
- `compute_conformal_winkler_metrics`
- `iTransformer`
- `build_model`
- `train_flow`
- `eval_flow`

### 13.2 新增或重写

建议新增：

- `get_project_root(config, script_dir)`
- `get_yaml_stem(config_path)`
- `resolve_path(path, project_root)`
- `create_task_dir(config, mode, script_dir)`
- `setup_task_logger(task_dir)`
- `save_config_snapshot(config, task_dir)`
- `validate_data_config(data_cfg)`
- `validate_time_interval(df, timestamp_col, interval_minutes, policy)`
- `batch_eval_flow(config, script_dir)`
- `write_metrics(metrics, path)`
- `write_batch_summary(rows, summary_csv)`

### 13.3 删除或弱化

建议弱化旧逻辑：

- `get_experiment_paths` 改为 `create_task_paths`。
- 不再使用 `checkpoint_dir + experiment_name` 作为默认输出。
- 不再使用 `task_name`。
- 不再使用 `output_dir` 表达 results 专用目录，也不建议用它表达整个任务目录。
- 使用 `output_root` 表达统一输出根目录，默认是项目根目录下的 `outputs/`。
- 不再使用 `log/` 和 `batch_logs/`。
- 不再生成 `tmp_eval_configs/`。

## 14. 路径优先级

训练数据路径：

```text
paths.train_csv
  > paths.data_dir + paths.train_file
```

验证数据路径：

```text
paths.val_csv
  > paths.data_dir + paths.val_file
```

测试数据路径：

```text
paths.test_csv
  > paths.data_dir + paths.test_file
```

任务输出目录：

```text
paths.output_root / {yaml_stem}_{time}
  > project_root / outputs / {yaml_stem}_{time}
```

其中 `yaml_stem` 是当前 `CONFIG_FILE` 的文件名去掉扩展名：

```text
skippd_train.yaml -> skippd_train
folmos_eval.yaml -> folmos_eval
skippd_batch_eval.yaml -> skippd_batch_eval
```

评估模型路径：

```text
paths.model_path 必须显式提供
```

评估归一化路径：

```text
paths.norm_stats_path 必须显式提供
```

训练模型输出路径：

```text
task_dir / model_best.pth
```

训练归一化输出路径：

```text
task_dir / norm_stats.npz
```

## 15. 迁移步骤

### 阶段 1：任务目录和日志统一

目标：

- 每次运行只创建一个任务目录。
- 所有输出都进入该任务目录。
- 只生成一个 `task.log`。

改动：

- 新增 `create_task_dir`。
- 新增 `setup_task_logger`。
- 修改 `train_flow` 和 `eval_flow` 的输出路径。
- 修改 `run_train.sh`、`run_eval.sh`，去掉额外日志重定向。

验收：

- 跑一次 `skippd_train.yaml`，只出现 `outputs/skippd_train_*/task.log`。
- 跑一次 `folmos_eval.yaml`，只出现 `outputs/folmos_eval_*/task.log`。
- 根目录 `log/` 不再新增文件。

### 阶段 2：绝对路径和配置校验

目标：

- 所有输入输出路径都可直接写绝对路径。
- 相对路径行为明确。

改动：

- 重写 `make_abs_path` 为 `resolve_path`。
- 支持 `paths.project_root`、`paths.output_root`。
- 任务目录名来自 YAML 文件名，不来自配置字段。
- 启动时打印并记录所有解析后的关键路径。

验收：

- `train_csv`、`val_csv`、`test_csv` 使用绝对路径可以正常运行。
- `model_path`、`norm_stats_path` 使用绝对路径可以正常评估。

### 阶段 3：窗口、间隔、目标列完全配置化

目标：

- 改 `sequence_length` 后模型输入自动变化。
- 改 `target_column` 后训练目标自动变化。
- 改 `target_horizon_minutes` 后导出的目标时间戳自动变化。

改动：

- 新增 `validate_data_config`。
- 明确 `target_column` 必须存在于 CSV。
- `export_test_predictions_csv` 使用 `target_horizon_minutes` 生成 `target_timestamp`。
- 可选增加时间间隔校验。

验收：

- `target_ghi_15min` 和 `target_ghi_30min` 都能通过改配置运行。
- `sequence_length=12` 和 `sequence_length=24` 都能运行。

### 阶段 4：批量评估不再生成临时 YAML

目标：

- 批量评估只有一个大任务目录。
- 每个测试 CSV 一个子目录。
- 总日志只有一个。

改动：

- 新增 `batch_eval_flow`。
- 支持 `mode: batch_eval`。
- 由 Python 直接循环 CSV，不再由 Shell 生成临时 YAML。
- 写入 `summary.csv`。

验收：

- 不再新增 `tmp_eval_configs/`。
- `outputs/*_batch_eval_*/summary.csv` 包含所有子任务指标。

### 阶段 5：清理历史目录

目标：

- 新代码不再使用旧目录。
- 历史结果可以保留但不再增长。

建议：

- 旧 `results/`、`checkpoint/`、`checkpoints/`、`log/`、`batch_logs/` 可先保留。
- 新实验全部进入 `outputs/`。
- 后续确认不需要后，再手动归档或删除。

## 16. 第一轮最小改动范围

第一轮建议只改这些：

```text
train_itransformer.py
train.yaml
eval.yaml
run_train.sh
run_eval.sh
docs/REFACTORING_PLAN.md
```

第一轮不处理：

```text
results/
checkpoints/
checkpoint/
log/
batch_logs/
tmp_eval_configs/
kernel_meta/
```

第一轮目标是让新任务输出进入 `outputs/`，而不是清历史垃圾。

## 17. 建议的最终运行方式

训练：

```bash
CONFIG_FILE=/absolute/path/train.yaml python -u /absolute/path/train_itransformer.py
```

评估：

```bash
CONFIG_FILE=/absolute/path/eval.yaml python -u /absolute/path/train_itransformer.py
```

批量评估：

```bash
CONFIG_FILE=/absolute/path/batch_eval.yaml python -u /absolute/path/train_itransformer.py
```

Windows PowerShell：

```powershell
$env:CONFIG_FILE="C:\absolute\path\train.yaml"
python -u "C:\absolute\path\train_itransformer.py"
```

## 18. 重构完成标准

完成后应满足：

- 主代码仍然是一个文件。
- 所有输入路径支持绝对路径。
- 所有输出进入 `outputs/{yaml_stem}_{time}/` 单个任务目录。
- 一个任务只有一个 `task.log`。
- 训练任务目录内包含模型、归一化统计、训练日志、loss 和图。
- 评估任务目录内包含日志、配置快照、指标、预测 CSV 和图片。
- 批量评估不再生成临时 YAML。
- `sequence_length`、`target_column`、`target_horizon_minutes`、`time_interval_minutes` 都可以通过配置修改。
- 新运行不再往 `log/`、`batch_logs/`、`tmp_eval_configs/`、`checkpoints/`、`results/` 写文件。
