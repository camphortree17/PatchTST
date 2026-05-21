import os
import time
import sys
import copy
import glob
import warnings
from collections import namedtuple

warnings.filterwarnings(
    'ignore',
    message=r'.*owner does not match the current owner.*',
    category=UserWarning,
)

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn, Tensor
from torch.nn import ModuleList
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    import torch_npu  # noqa: F401
    HAS_TORCH_NPU = hasattr(torch, 'npu')
except Exception:
    HAS_TORCH_NPU = False

try:
    from numpy.lib.stride_tricks import sliding_window_view
    HAS_SLIDING_WINDOW_VIEW = True
except Exception:
    HAS_SLIDING_WINDOW_VIEW = False


# =========================
# 基础工具函数
# =========================

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if HAS_TORCH_NPU and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)


def make_abs_path(path_str: str, script_dir: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    return os.path.abspath(os.path.join(script_dir, path_str))

def resolve_path_from_cfg(paths_cfg, script_dir, direct_key=None, dir_key=None, file_key=None):
    """
    优先读取 direct_key 对应的完整路径；
    若不存在，则回退到 dir_key + file_key 的组合方式。
    """
    if direct_key and paths_cfg.get(direct_key):
        return resolve_path(paths_cfg[direct_key], script_dir)

    if dir_key and file_key and paths_cfg.get(dir_key) and paths_cfg.get(file_key):
        base_dir = resolve_path(paths_cfg[dir_key], script_dir)
        return os.path.join(base_dir, paths_cfg[file_key])

    return None


def load_config(script_dir: str):
    config_file = os.environ.get('CONFIG_FILE', 'train.yaml')
    config_path = make_abs_path(config_file, script_dir)
    print(f'读取配置文件: {config_path}', flush=True)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config, config_path


class TeeStream:
    def __init__(self, console_stream, file_stream):
        self.console_stream = console_stream
        self.file_stream = file_stream

    def write(self, text):
        self.console_stream.write(text)
        self.file_stream.write(text)

    def flush(self):
        self.console_stream.flush()
        self.file_stream.flush()


def resolve_path(path_str, base_dir):
    if path_str is None:
        return None
    path_str = os.path.expanduser(str(path_str))
    if os.path.isabs(path_str):
        return os.path.abspath(path_str)
    return os.path.abspath(os.path.join(base_dir, path_str))


def get_project_root(config, script_dir):
    paths_cfg = config.get('paths', {})
    project_root = paths_cfg.get('project_root')
    if project_root:
        return resolve_path(project_root, script_dir)
    return script_dir


def create_task_dir(config, config_path, script_dir):
    project_root = get_project_root(config, script_dir)
    paths_cfg = config.setdefault('paths', {})
    output_root = resolve_path(paths_cfg.get('output_root', 'outputs'), project_root)
    yaml_stem = os.path.splitext(os.path.basename(config_path))[0]
    time_str = time.strftime('%Y%m%d_%H%M%S')
    task_dir = os.path.join(output_root, f'{yaml_stem}_{time_str}')
    if os.path.exists(task_dir):
        suffix = 2
        while os.path.exists(f'{task_dir}_{suffix}'):
            suffix += 1
        task_dir = f'{task_dir}_{suffix}'
    os.makedirs(task_dir, exist_ok=False)

    config['_runtime'] = {
        'project_root': project_root,
        'output_root': output_root,
        'task_dir': task_dir,
        'config_path': config_path,
        'yaml_stem': yaml_stem,
        'time_str': time_str,
    }
    return task_dir


def setup_task_logger(task_dir):
    log_path = os.path.join(task_dir, 'task.log')
    log_file = open(log_path, 'a', encoding='utf-8', buffering=1)
    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)
    return log_path


def save_config_snapshot(config, task_dir):
    snapshot = copy.deepcopy(config)
    snapshot.pop('_runtime', None)
    snapshot_path = os.path.join(task_dir, 'config.yaml')
    with open(snapshot_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(snapshot, f, allow_unicode=True, sort_keys=False)
    return snapshot_path


def get_path_base(config, script_dir):
    return config.get('_runtime', {}).get('project_root', script_dir)


def parse_feature_columns(data_cfg):
    """
    兼容两种配置方式：
    1) 旧版：feature_columns: [irr_ghi]
    2) 新版：feature_columns_csv: "col1,col2,col3"

    优先级：feature_columns_csv > feature_columns
    """
    feature_columns_csv = data_cfg.get('feature_columns_csv', None)
    if feature_columns_csv is not None:
        if isinstance(feature_columns_csv, str):
            parsed = [c.strip() for c in feature_columns_csv.split(',') if c.strip()]
        elif isinstance(feature_columns_csv, (list, tuple)):
            parsed = [str(c).strip() for c in feature_columns_csv if str(c).strip()]
        else:
            raise TypeError('data.feature_columns_csv 仅支持字符串或字符串列表。')

        if len(parsed) == 0:
            raise ValueError('data.feature_columns_csv 已配置，但解析后为空。')
        return parsed

    feature_columns = data_cfg.get('feature_columns', None)
    if feature_columns is None:
        raise ValueError('配置中缺少 data.feature_columns 或 data.feature_columns_csv。')

    if isinstance(feature_columns, str):
        feature_columns = [feature_columns]
    elif not isinstance(feature_columns, (list, tuple)):
        raise TypeError('data.feature_columns 必须是字符串列表或单个字符串。')

    parsed = [str(c).strip() for c in feature_columns if str(c).strip()]
    if len(parsed) == 0:
        raise ValueError('data.feature_columns 解析后为空。')
    return parsed


def get_norm_columns(feature_columns, target_column):
    return list(feature_columns) + [target_column]


def align_norm_stats_to_columns(norm_stats_path, expected_columns):
    if not os.path.exists(norm_stats_path):
        raise FileNotFoundError(f'找不到归一化统计量文件: {norm_stats_path}')

    stats = np.load(norm_stats_path, allow_pickle=True)
    min_vals = stats['min_vals'].astype(np.float32)
    max_vals = stats['max_vals'].astype(np.float32)
    stored_columns = stats['columns'].tolist() if 'columns' in stats.files else None

    expected_columns = list(expected_columns)

    if stored_columns is None:
        if len(min_vals) != len(expected_columns):
            raise ValueError(
                f'归一化统计量维度与当前配置不一致: stats={len(min_vals)}, expected={len(expected_columns)}。'
            )
        return min_vals, max_vals

    stored_columns = [str(c) for c in stored_columns]
    if len(stored_columns) != len(min_vals) or len(stored_columns) != len(max_vals):
        raise ValueError('归一化统计量文件损坏：columns 与 min/max 长度不一致。')

    index_map = {col: idx for idx, col in enumerate(stored_columns)}
    missing_cols = [col for col in expected_columns if col not in index_map]
    if missing_cols:
        raise ValueError(
            '当前配置所需列在归一化统计量文件中不存在: ' + ', '.join(missing_cols)
        )

    ordered_indices = [index_map[col] for col in expected_columns]
    return min_vals[ordered_indices], max_vals[ordered_indices]


def fit_normalization_stats(data):
    data_np = np.asarray(data, dtype=np.float32)
    min_vals = np.min(data_np, axis=0).astype(np.float32)
    max_vals = np.max(data_np, axis=0).astype(np.float32)
    return min_vals, max_vals


def apply_normalization(data, min_vals, max_vals):
    data_np = np.asarray(data, dtype=np.float32)
    denom = max_vals - min_vals
    denom = np.where(denom == 0, 1.0, denom).astype(np.float32)
    normalized = (data_np - min_vals) / denom
    return normalized.astype(np.float32)


def unnormalize_data(normalized_data, min_vals, max_vals):
    normalized_data = np.asarray(normalized_data, dtype=np.float32)
    return normalized_data * (max_vals - min_vals) + min_vals


def load_single_csv(csv_path, timestamp_col, required_columns, timestamp_format=None):
    print(f'[开始读取] {csv_path}', flush=True)
    df = pd.read_csv(csv_path)
    print(f'[读取完成] {csv_path}, shape={df.shape}', flush=True)

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f'{csv_path} 中缺少必要列: {col}')

    print(f'[开始转换时间戳] {csv_path}', flush=True)
    if timestamp_format:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], format=timestamp_format)
    else:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    print(f'[时间戳转换完成] {csv_path}', flush=True)

    df = df.sort_values(timestamp_col).reset_index(drop=True)
    df = df[required_columns].dropna().reset_index(drop=True)
    print(f'[清洗完成] {csv_path}, shape={df.shape}', flush=True)
    return df


def validate_time_interval(df, timestamp_col, interval_minutes, policy='warn', split_name='data'):
    policy = str(policy or 'ignore').strip().lower()
    if policy == 'ignore' or interval_minutes is None:
        return

    expected = pd.Timedelta(minutes=int(interval_minutes))
    diffs = df[timestamp_col].diff().dropna()
    bad_count = int((diffs != expected).sum())
    if bad_count == 0:
        print(f'[time interval ok] {split_name}: {interval_minutes} minutes', flush=True)
        return

    message = (
        f'[time interval gap] {split_name}: expected {interval_minutes} minutes, '
        f'found {bad_count} non-continuous steps'
    )
    if policy == 'error':
        raise ValueError(message)
    if policy == 'warn':
        print(message, flush=True)


def create_sliding_window(data, feature_columns, target_column, sequence_length):
    """
    输入：过去 sequence_length 个点
    输出：窗口最后一个输入时刻对应的 target_ghi_15min
    """
    feature_array = data[feature_columns].to_numpy(dtype=np.float32)
    target_array = data[target_column].to_numpy(dtype=np.float32)

    n_samples = len(data) - sequence_length + 1
    feature_dim = len(feature_columns)

    if n_samples <= 0:
        return (
            np.empty((0, sequence_length, feature_dim), dtype=np.float32),
            np.empty((0, 1), dtype=np.float32),
        )

    if HAS_SLIDING_WINDOW_VIEW:
        X_view = sliding_window_view(feature_array, window_shape=sequence_length, axis=0)
        X = np.transpose(X_view, (0, 2, 1))
    else:
        X = np.stack([feature_array[i:i + sequence_length] for i in range(n_samples)], axis=0)

    y = target_array[sequence_length - 1:].reshape(-1, 1)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=True).astype(np.float32, copy=False)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0, copy=True).astype(np.float32, copy=False)

    return X, y


def prepare_split_dataset(df, feature_columns, target_column, sequence_length, min_vals, max_vals, split_name):
    print(f'[开始归一化] {split_name}', flush=True)
    norm_columns = get_norm_columns(feature_columns, target_column)
    scaled_array = apply_normalization(df[norm_columns].values, min_vals, max_vals)
    scaled_df = pd.DataFrame(scaled_array, columns=norm_columns)
    print(f'[归一化完成] {split_name}', flush=True)

    print(f'[开始构造滑窗] {split_name}', flush=True)
    X, y = create_sliding_window(
        data=scaled_df,
        feature_columns=feature_columns,
        target_column=target_column,
        sequence_length=sequence_length
    )
    print(f'[滑窗完成] {split_name}, X={X.shape}, y={y.shape}', flush=True)

    X_tensor = torch.from_numpy(np.ascontiguousarray(X)).float()
    y_tensor = torch.from_numpy(np.ascontiguousarray(y)).float()
    return X_tensor, y_tensor


def eval_metrics(true, pred):
    true = np.asarray(true).reshape(-1)
    pred = np.asarray(pred).reshape(-1)

    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true, pred)

    max_true = np.max(true)
    if max_true == 0:
        nrmse = float('inf')
        nmae = float('inf')
    else:
        nrmse = rmse / max_true
        nmae = mae / max_true
    return mse, rmse, mae, nrmse, nmae, max_true




def ensure_2d_array(array):
    """统一保持 [N, C] 形状，单目标也保留 [N, 1]，避免 squeeze 成标量。"""
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def calibrate_abs_residual_quantile(y_true, y_pred, alpha=0.1, axis=0):
    """
    用验证集绝对残差校准 conformal 区间半径。
    y_true/y_pred: [N, C]
    返回：每个 target 的 (1-alpha) 分位数残差半径。
    """
    y_true = ensure_2d_array(y_true).astype(np.float64)
    y_pred = ensure_2d_array(y_pred).astype(np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f'验证集预测与真实值形状不一致: y_true={y_true.shape}, y_pred={y_pred.shape}'
        )
    if y_true.shape[0] == 0:
        raise ValueError('验证集样本数为 0，无法校准 WS 区间。')

    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f'ws_alpha 必须位于 (0, 1)，当前为: {alpha}')

    abs_residual = np.abs(y_true - y_pred)
    q = np.quantile(abs_residual, 1.0 - alpha, axis=axis)
    return np.asarray(q, dtype=np.float64)


def build_prediction_interval(y_pred, q_low, q_high, lower_bound=0.0):
    """根据点预测和校准残差分位数构造预测区间。"""
    y_pred = ensure_2d_array(y_pred).astype(np.float64)
    q_low = np.asarray(q_low, dtype=np.float64)
    q_high = np.asarray(q_high, dtype=np.float64)

    lower = y_pred + q_low
    upper = y_pred + q_high

    if lower_bound is not None:
        lower = np.maximum(float(lower_bound), lower)

    if np.any(upper < lower):
        raise ValueError('预测区间存在 upper < lower，请检查 q_low/q_high。')

    return lower, upper


def compute_winkler_score(y_true, lower, upper, alpha=0.1):
    """
    计算 Winkler Score、覆盖率和平均区间宽度。
    WS 越小越好；coverage 越接近 1-alpha 越合理；sharpness 为平均区间宽度。
    """
    y_true = ensure_2d_array(y_true).astype(np.float64)
    lower = ensure_2d_array(lower).astype(np.float64)
    upper = ensure_2d_array(upper).astype(np.float64)

    if not (y_true.shape == lower.shape == upper.shape):
        raise ValueError(
            'WS 输入形状不一致: '
            f'y_true={y_true.shape}, lower={lower.shape}, upper={upper.shape}'
        )
    if y_true.shape[0] == 0:
        raise ValueError('测试集样本数为 0，无法计算 WS。')

    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f'ws_alpha 必须位于 (0, 1)，当前为: {alpha}')

    width = upper - lower
    below = y_true < lower
    above = y_true > upper

    score = width.copy()
    score[below] += (2.0 / alpha) * (lower[below] - y_true[below])
    score[above] += (2.0 / alpha) * (y_true[above] - upper[above])

    covered = (y_true >= lower) & (y_true <= upper)

    return {
        'overall_ws': float(np.mean(score)),
        'per_target_ws': np.mean(score, axis=0).astype(np.float64),
        'overall_coverage': float(np.mean(covered)),
        'per_target_coverage': np.mean(covered, axis=0).astype(np.float64),
        'overall_sharpness': float(np.mean(width)),
        'per_target_sharpness': np.mean(width, axis=0).astype(np.float64),
    }


def compute_conformal_winkler_metrics(val_true, val_pred, test_true, test_pred, alpha=0.1):
    """使用验证集残差校准区间，并在测试集上计算 WS。"""
    q_radius = calibrate_abs_residual_quantile(
        y_true=val_true,
        y_pred=val_pred,
        alpha=alpha,
        axis=0,
    )
    lower, upper = build_prediction_interval(
        y_pred=test_pred,
        q_low=-q_radius,
        q_high=q_radius,
        lower_bound=0.0,
    )
    ws_summary = compute_winkler_score(
        y_true=test_true,
        lower=lower,
        upper=upper,
        alpha=alpha,
    )
    ws_summary['q_radius'] = q_radius
    ws_summary['lower'] = lower
    ws_summary['upper'] = upper
    return ws_summary


def predict_loader_norm(model, data_loader, device):
    """返回归一化尺度下的预测和真实值，统一保持 [N, C]。"""
    model.eval()
    pred_list, true_list = [], []
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x).detach().cpu().numpy()
            true = batch_y.numpy()
            pred_list.append(pred)
            true_list.append(true)

    if len(pred_list) == 0:
        return (
            np.empty((0, 1), dtype=np.float32),
            np.empty((0, 1), dtype=np.float32),
        )

    pred_norm = ensure_2d_array(np.vstack(pred_list))
    true_norm = ensure_2d_array(np.vstack(true_list))
    return pred_norm, true_norm

def npu_available():
    return HAS_TORCH_NPU and hasattr(torch, 'npu') and torch.npu.is_available()


def select_device(runtime_cfg):
    device_pref = str(runtime_cfg.get('device', 'auto')).strip().lower()
    npu_idx = int(runtime_cfg.get('npu_device_index', 0))
    cuda_idx = int(runtime_cfg.get('cuda_device_index', 0))

    if device_pref in ('auto', 'npu') and npu_available():
        device = torch.device(f'npu:{npu_idx}')
        torch.npu.set_device(device)
        return device, 'npu'

    if device_pref in ('auto', 'cuda', 'gpu') and torch.cuda.is_available():
        device = torch.device(f'cuda:{cuda_idx}')
        torch.cuda.set_device(device)
        return device, 'cuda'

    if device_pref == 'npu':
        raise RuntimeError('配置要求使用 NPU，但当前环境未检测到可用 NPU。')
    if device_pref in ('cuda', 'gpu'):
        raise RuntimeError('配置要求使用 CUDA，但当前环境未检测到可用 CUDA。')

    return torch.device('cpu'), 'cpu'


# =========================
# 辅助函数和类
# =========================
def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


Statistics = namedtuple('Statistics', ['mean', 'variance', 'gamma', 'beta'])


# =========================
# 模型模块（保持你的模型主体不变）
# =========================
class Attend(nn.Module):
    def __init__(self, *, dropout=0., heads=None, scale=None, flash=False, causal=False):
        super().__init__()
        self.scale = scale
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.causal = causal
        self.flash = flash

    def forward(self, q, k, v):
        n, heads, kv_heads, device, dtype = q.shape[-2], q.shape[1], k.shape[1], q.device, q.dtype
        scale = default(self.scale, q.shape[-1] ** -0.5)

        if self.flash and hasattr(F, 'scaled_dot_product_attention'):
            return F.scaled_dot_product_attention(
                q, k, v,
                is_causal=self.causal,
                dropout_p=self.dropout if self.training else 0.
            )

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * scale

        if self.causal:
            i, j = sim.shape[-2:]
            mask_value = -torch.finfo(sim.dtype).max
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        return torch.einsum('b h i j, b h j d -> b h i d', attn, v)


class RevIN(nn.Module):
    def __init__(self, num_variates, affine=True, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.num_variates = num_variates
        self.gamma = nn.Parameter(torch.ones(num_variates, 1), requires_grad=affine)
        self.beta = nn.Parameter(torch.zeros(num_variates, 1), requires_grad=affine)

    def forward(self, x, return_statistics=False):
        assert x.shape[1] == self.num_variates
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        var_rsqrt = var.clamp(min=self.eps).rsqrt()
        instance_normalized = (x - mean) * var_rsqrt
        rescaled = instance_normalized * self.gamma + self.beta

        def reverse_fn(scaled_output):
            clamped_gamma = torch.sign(self.gamma) * self.gamma.abs().clamp(min=self.eps)
            unscaled_output = (scaled_output - self.beta) / clamped_gamma
            return unscaled_output * var.sqrt() + mean

        if not return_statistics:
            return rescaled, reverse_fn

        return rescaled, reverse_fn, Statistics(mean, var, self.gamma, self.beta)


class Attention(nn.Module):
    def __init__(self, dim, dim_head=32, heads=4, dropout=0., flash=True):
        super().__init__()
        dim_inner = dim_head * heads
        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias=False),
            Rearrange('b n (qkv h d) -> qkv b h n d', qkv=3, h=heads)
        )
        self.attend = Attend(flash=flash, dropout=dropout)
        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        q, k, v = self.to_qkv(x)
        out = self.attend(q, k, v)
        return self.to_out(out)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = rearrange(x, '... (r d) -> r ... d', r=2)
        return x * F.gelu(gate)


def FeedForward(dim, mult=4, dropout=0.):
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_inner, dim)
    )


class iTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        input_height: int,
        input_width: int,
        hidden_size: int,
        output_dim: int,
        depth: int = 3,
        dim_head: int = 32,
        heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        num_mem_tokens: int = 4,
        use_reversible_instance_norm: bool = True,
        reversible_instance_norm_affine: bool = True,
        flash_attn: bool = True
    ):
        super().__init__()
        self.num_variates = input_width
        self.lookback_len = input_height
        self.dim = hidden_size
        self.num_tokens_per_variate = 1

        self.input_proj = nn.Sequential(
            nn.Linear(input_height, hidden_size * self.num_tokens_per_variate),
            Rearrange('b v (n d) -> b (v n) d', n=self.num_tokens_per_variate),
            nn.LayerNorm(hidden_size)
        )

        self.reversible_instance_norm = RevIN(
            num_variates=input_width,
            affine=reversible_instance_norm_affine
        ) if use_reversible_instance_norm else None

        self.mem_tokens = nn.Parameter(torch.randn(num_mem_tokens, hidden_size)) if num_mem_tokens > 0 else None

        self.layers = ModuleList([
            ModuleList([
                Attention(hidden_size, dim_head=dim_head, heads=heads, dropout=dropout, flash=flash_attn),
                nn.LayerNorm(hidden_size),
                FeedForward(hidden_size, mult=ff_mult, dropout=dropout),
                nn.LayerNorm(hidden_size)
            ]) for _ in range(depth)
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size * input_width, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, 'b n v -> b v n')

        if exists(self.reversible_instance_norm):
            x, reverse_fn = self.reversible_instance_norm(x)

        x = self.input_proj(x)

        if exists(self.mem_tokens):
            m = repeat(self.mem_tokens, 'm d -> b m d', b=x.shape[0])
            x, mem_ps = pack([m, x], 'b * d')

        for attn, attn_post_norm, ff, ff_post_norm in self.layers:
            x = attn(x) + x
            x = attn_post_norm(x)
            x = ff(x) + x
            x = ff_post_norm(x)

        if exists(self.mem_tokens):
            _, x = unpack(x, mem_ps, 'b * d')

        if exists(self.reversible_instance_norm):
            x = rearrange(x, 'b (v n) d -> n b v d', n=self.num_tokens_per_variate)
            x = reverse_fn(x)
            x = rearrange(x, 'n b v d -> b (v n) d', n=self.num_tokens_per_variate)

        x = rearrange(x, 'b v d -> b (v d)')
        output = self.output_proj(x)
        return output


def build_model(config, n_features, backend):
    data_cfg = config['data']
    model_cfg = config['model']

    flash_attn = bool(model_cfg.get('flash_attn', True))
    if backend == 'npu' and flash_attn:
        print('检测到 NPU，已自动将 model.flash_attn 从 true 调整为 false 以提高兼容性。', flush=True)
        flash_attn = False

    model = iTransformer(
        in_channels=1,
        input_height=int(data_cfg['sequence_length']),
        input_width=n_features,
        hidden_size=int(model_cfg.get('hidden_size', 128)),
        output_dim=1,
        depth=int(model_cfg.get('depth', 3)),
        dim_head=int(model_cfg.get('dim_head', 32)),
        heads=int(model_cfg.get('heads', 4)),
        ff_mult=int(model_cfg.get('ff_mult', 4)),
        dropout=float(model_cfg.get('dropout', 0.1)),
        num_mem_tokens=int(model_cfg.get('num_mem_tokens', 4)),
        use_reversible_instance_norm=bool(model_cfg.get('use_reversible_instance_norm', True)),
        reversible_instance_norm_affine=bool(model_cfg.get('reversible_instance_norm_affine', True)),
        flash_attn=flash_attn
    )
    return model


def get_experiment_paths(config, script_dir):
    runtime_meta = config.get('_runtime', {})
    if not runtime_meta.get('task_dir'):
        raise RuntimeError('Task directory is not initialized. Call create_task_dir before train/eval flow.')

    exp_dir = runtime_meta['task_dir']
    mode = str(config.get('mode', 'train')).strip().lower()
    paths_cfg = config.get('paths', {})
    base_dir = runtime_meta.get('project_root', script_dir)

    if mode == 'train':
        model_path = os.path.join(exp_dir, 'model_best.pth')
        norm_stats_path = os.path.join(exp_dir, 'norm_stats.npz')
    else:
        model_path = resolve_path(paths_cfg.get('model_path'), base_dir)
        norm_stats_path = resolve_path(paths_cfg.get('norm_stats_path'), base_dir)
        if not model_path:
            raise ValueError('paths.model_path must be set for eval or batch_eval mode')
        if not norm_stats_path:
            raise ValueError('paths.norm_stats_path must be set for eval or batch_eval mode')

    return exp_dir, model_path, norm_stats_path


def load_train_val_data(config, script_dir):
    paths_cfg = config['paths']
    data_cfg = config['data']
    path_base = get_path_base(config, script_dir)

    train_csv = resolve_path_from_cfg(
        paths_cfg, path_base,
        direct_key='train_csv',
        dir_key='data_dir',
        file_key='train_file'
    )
    val_csv = resolve_path_from_cfg(
        paths_cfg, path_base,
        direct_key='val_csv',
        dir_key='data_dir',
        file_key='val_file'
    )

    print(f'train_csv: {train_csv}', flush=True)
    print(f'val_csv: {val_csv}', flush=True)

    for p in [train_csv, val_csv]:
        if not p or not os.path.exists(p):
            raise FileNotFoundError(f'文件不存在: {p}')

    timestamp_col = data_cfg['timestamp_col']
    timestamp_format = data_cfg.get('timestamp_format', None)
    feature_columns = parse_feature_columns(data_cfg)
    target_column = data_cfg['target_column']
    sequence_length = int(data_cfg['sequence_length'])
    required_columns = [timestamp_col] + feature_columns + [target_column]

    train_df = load_single_csv(train_csv, timestamp_col, required_columns, timestamp_format)
    val_df = load_single_csv(val_csv, timestamp_col, required_columns, timestamp_format)
    validate_time_interval(
        train_df, timestamp_col, data_cfg.get('time_interval_minutes'),
        data_cfg.get('time_gap_policy', 'warn'), 'train'
    )
    validate_time_interval(
        val_df, timestamp_col, data_cfg.get('time_interval_minutes'),
        data_cfg.get('time_gap_policy', 'warn'), 'val'
    )

    norm_columns = get_norm_columns(feature_columns, target_column)
    min_vals, max_vals = fit_normalization_stats(train_df[norm_columns].values)
    return train_df, val_df, feature_columns, target_column, sequence_length, min_vals, max_vals


def load_test_data(config, script_dir):
    paths_cfg = config['paths']
    data_cfg = config['data']
    path_base = get_path_base(config, script_dir)

    test_csv = resolve_path_from_cfg(
        paths_cfg, path_base,
        direct_key='test_csv',
        dir_key='data_dir',
        file_key='test_file'
    )

    print(f'test_csv: {test_csv}', flush=True)

    if not test_csv or not os.path.exists(test_csv):
        raise FileNotFoundError(f'文件不存在: {test_csv}')

    timestamp_col = data_cfg['timestamp_col']
    timestamp_format = data_cfg.get('timestamp_format', None)
    feature_columns = parse_feature_columns(data_cfg)
    target_column = data_cfg['target_column']
    sequence_length = int(data_cfg['sequence_length'])
    required_columns = [timestamp_col] + feature_columns + [target_column]

    test_df = load_single_csv(test_csv, timestamp_col, required_columns, timestamp_format)
    validate_time_interval(
        test_df, timestamp_col, data_cfg.get('time_interval_minutes'),
        data_cfg.get('time_gap_policy', 'warn'), 'test'
    )
    return test_df, feature_columns, target_column, sequence_length


def load_val_data_for_eval(config, script_dir):
    """评估阶段读取验证集，用于 WS 的残差分位数校准；不重新拟合归一化统计量。"""
    paths_cfg = config['paths']
    data_cfg = config['data']
    path_base = get_path_base(config, script_dir)

    val_csv = resolve_path_from_cfg(
        paths_cfg, path_base,
        direct_key='val_csv',
        dir_key='data_dir',
        file_key='val_file'
    )

    print(f'val_csv: {val_csv}', flush=True)

    if not val_csv or not os.path.exists(val_csv):
        raise FileNotFoundError(
            f'文件不存在: {val_csv}。WS 指标需要验证集进行残差校准，请确认 paths.val_csv 或 paths.data_dir + paths.val_file 可用。'
        )

    timestamp_col = data_cfg['timestamp_col']
    timestamp_format = data_cfg.get('timestamp_format', None)
    feature_columns = parse_feature_columns(data_cfg)
    target_column = data_cfg['target_column']
    sequence_length = int(data_cfg['sequence_length'])
    required_columns = [timestamp_col] + feature_columns + [target_column]

    val_df = load_single_csv(val_csv, timestamp_col, required_columns, timestamp_format)
    validate_time_interval(
        val_df, timestamp_col, data_cfg.get('time_interval_minutes'),
        data_cfg.get('time_gap_policy', 'warn'), 'val'
    )
    return val_df, feature_columns, target_column, sequence_length


def save_norm_stats(norm_stats_path, min_vals, max_vals, feature_columns, target_column):
    np.savez(
        norm_stats_path,
        min_vals=min_vals,
        max_vals=max_vals,
        columns=np.array(feature_columns + [target_column], dtype=object)
    )
    print(f'归一化统计量已保存到: {norm_stats_path}', flush=True)


def load_norm_stats(norm_stats_path, expected_columns=None):
    if expected_columns is None:
        if not os.path.exists(norm_stats_path):
            raise FileNotFoundError(f'找不到归一化统计量文件: {norm_stats_path}')
        stats = np.load(norm_stats_path, allow_pickle=True)
        return stats['min_vals'].astype(np.float32), stats['max_vals'].astype(np.float32)

    return align_norm_stats_to_columns(norm_stats_path, expected_columns)


def save_model_state_dict(model, model_path):
    cpu_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    tmp_model_path = model_path + '.tmp'
    torch.save(cpu_state_dict, tmp_model_path)
    os.replace(tmp_model_path, model_path)


def parse_save_epochs(train_cfg):
    save_epochs = train_cfg.get('save_epochs', [])

    if save_epochs is None:
        return []

    if isinstance(save_epochs, str):
        save_epochs = [item.strip() for item in save_epochs.split(',') if item.strip()]

    if not isinstance(save_epochs, (list, tuple)):
        raise ValueError('train.save_epochs 必须为列表、元组、逗号分隔字符串，或为空。')

    parsed = []
    for item in save_epochs:
        epoch = int(item)
        if epoch <= 0:
            raise ValueError(f'train.save_epochs 中存在非法 epoch: {epoch}，必须为正整数。')
        parsed.append(epoch)

    return sorted(set(parsed))


def evaluate_loader_loss(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            batch_size = batch_x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    return total_loss / max(total_samples, 1)


def export_test_predictions_csv(config, exp_dir, test_df, feature_columns, sequence_length, pred, true):
    eval_cfg = config.get('eval', {})
    data_cfg = config['data']

    result_csv_file = str(eval_cfg.get('result_csv_file', 'test_predictions.csv'))
    timestamp_col = data_cfg['timestamp_col']
    target_horizon_minutes = int(data_cfg.get('target_horizon_minutes', 15))

    n_samples = len(test_df) - sequence_length + 1
    if n_samples <= 0:
        print('测试样本数为0，跳过结果CSV导出。', flush=True)
        return

    result_base_df = test_df.iloc[sequence_length - 1:].reset_index(drop=True).copy()

    if len(result_base_df) != len(pred):
        raise ValueError(
            f'结果导出长度不一致: result_base_df={len(result_base_df)}, pred={len(pred)}, true={len(true)}'
        )

    result_df = result_base_df[feature_columns].copy()
    result_df['target_timestamp'] = result_base_df[timestamp_col] + pd.to_timedelta(target_horizon_minutes, unit='m')
    result_df['y_pred'] = pred
    result_df['y_true'] = true

    result_csv_path = os.path.join(exp_dir, result_csv_file)
    result_df.to_csv(result_csv_path, index=False, encoding='utf-8-sig')
    print(f'测试结果CSV已保存到: {result_csv_path}', flush=True)


def train_flow(config, script_dir):
    set_seed(int(config.get('seed', 42)))
    runtime_cfg = config.get('runtime', {})
    train_cfg = config['train']

    exp_dir, model_path, norm_stats_path = get_experiment_paths(config, script_dir)

    device, backend = select_device(runtime_cfg)
    print(f'device: {device}', flush=True)
    print(f'backend: {backend}', flush=True)
    if backend == 'npu':
        print(f'NPU 可用: {torch.npu.is_available()}, 当前设备: {torch.npu.current_device()}', flush=True)

    train_df, val_df, feature_columns, target_column, sequence_length, min_vals, max_vals = load_train_val_data(config, script_dir)
    save_norm_stats(norm_stats_path, min_vals, max_vals, feature_columns, target_column)

    X_train, y_train = prepare_split_dataset(
        train_df, feature_columns, target_column, sequence_length, min_vals, max_vals, 'train'
    )
    X_val, y_val = prepare_split_dataset(
        val_df, feature_columns, target_column, sequence_length, min_vals, max_vals, 'val'
    )

    train_csv = resolve_path_from_cfg(
    config["paths"], get_path_base(config, script_dir),
    direct_key='train_csv',
    dir_key='data_dir',
    file_key='train_file'
    )
    val_csv = resolve_path_from_cfg(
        config["paths"], get_path_base(config, script_dir),
        direct_key='val_csv',
        dir_key='data_dir',
        file_key='val_file'
    )

    print(f'train.csv: {train_csv}', flush=True)
    print(f'val.csv: {val_csv}', flush=True)
    print(f'训练样本数: {len(X_train)}', flush=True)
    print(f'验证样本数: {len(X_val)}', flush=True)

    batch_size = int(train_cfg.get('batch_size', 256))
    shuffle = bool(train_cfg.get('shuffle', True))
    num_workers = int(train_cfg.get('num_workers', 0))

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    n_features = len(feature_columns)
    model = build_model(config, n_features, backend).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get('lr', 1e-3)))
    epochs = int(train_cfg.get('epochs', 150))
    patience = int(train_cfg.get('early_stop_patience', 0))
    save_epochs = parse_save_epochs(train_cfg)
    if save_epochs:
        print(f'额外保存指定 epoch checkpoint: {save_epochs}', flush=True)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    wait = 0
    total_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        total_samples = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            batch_size_now = batch_x.size(0)
            running_loss += loss.item() * batch_size_now
            total_samples += batch_size_now

        train_loss = running_loss / max(total_samples, 1)
        val_loss = evaluate_loader_loss(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            save_model_state_dict(model, model_path)
        else:
            wait += 1

        if epoch in save_epochs:
            epoch_model_path = os.path.join(exp_dir, f'checkpoint_{epoch}.pth')
            save_model_state_dict(model, epoch_model_path)
            print(f'已额外保存 epoch {epoch} 模型到: {epoch_model_path}', flush=True)

        epoch_time = time.time() - epoch_start
        total_time = time.time() - total_start
        print(
            f'epoch: {epoch:03d}/{epochs:03d}, '
            f'train_loss: {train_loss:.8f}, '
            f'val_loss: {val_loss:.8f}, '
            f'best_val_loss: {best_val_loss:.8f}, '
            f'epoch_time: {epoch_time:.2f}s, '
            f'total_time: {total_time:.2f}s',
            flush=True
        )

        if patience > 0 and wait >= patience:
            print(f'触发早停：连续 {patience} 个 epoch 验证集损失未提升。', flush=True)
            break

    print(f'最佳模型已保存到: {model_path}', flush=True)

    np.savetxt(os.path.join(exp_dir, 'train_loss.txt'), np.array(train_losses, dtype=np.float32))
    np.savetxt(os.path.join(exp_dir, 'val_loss.txt'), np.array(val_losses, dtype=np.float32))

    plt.figure('Train_Val_Loss')
    plt.plot(train_losses, label='train_loss')
    plt.plot(val_losses, label='val_loss')
    plt.legend()
    plt.savefig(os.path.join(exp_dir, 'Train_Val_Loss.png'))
    plt.close()


def eval_flow(config, script_dir):
    set_seed(int(config.get('seed', 42)))
    runtime_cfg = config.get('runtime', {})

    exp_dir, model_path, norm_stats_path = get_experiment_paths(config, script_dir)

    device, backend = select_device(runtime_cfg)
    print(f'device: {device}', flush=True)
    print(f'backend: {backend}', flush=True)
    if backend == 'npu':
        print(f'NPU 可用: {torch.npu.is_available()}, 当前设备: {torch.npu.current_device()}', flush=True)

    test_df, feature_columns, target_column, sequence_length = load_test_data(config, script_dir)
    val_df, val_feature_columns, val_target_column, val_sequence_length = load_val_data_for_eval(config, script_dir)

    if val_feature_columns != feature_columns or val_target_column != target_column or val_sequence_length != sequence_length:
        raise ValueError(
            '验证集配置与测试集配置不一致，无法用于 WS 校准: '
            f'val_feature_columns={val_feature_columns}, test_feature_columns={feature_columns}, '
            f'val_target_column={val_target_column}, test_target_column={target_column}, '
            f'val_sequence_length={val_sequence_length}, test_sequence_length={sequence_length}'
        )

    norm_columns = get_norm_columns(feature_columns, target_column)
    min_vals, max_vals = load_norm_stats(norm_stats_path, expected_columns=norm_columns)

    X_val, y_val = prepare_split_dataset(
        val_df, feature_columns, target_column, sequence_length, min_vals, max_vals, 'val'
    )
    X_test, y_test = prepare_split_dataset(
        test_df, feature_columns, target_column, sequence_length, min_vals, max_vals, 'test'
    )

    print(f'验证样本数: {len(X_val)}', flush=True)
    print(f'测试样本数: {len(X_test)}', flush=True)

    batch_size = int(config.get('eval', {}).get('batch_size', config.get('train', {}).get('batch_size', 256)))
    num_workers = int(config.get('eval', {}).get('num_workers', 0))
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    n_features = len(feature_columns)
    model = build_model(config, n_features, backend)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'找不到模型文件: {model_path}')
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        raise RuntimeError(
            '模型参数加载失败。通常是因为当前配置的特征维度与 checkpoint 训练时使用的特征维度不一致。'
            '若你现在启用了多特征输入，请使用同一组特征重新训练并生成对应的 checkpoint 与 norm_stats。'
            f'\n原始报错: {e}'
        ) from e
    model = model.to(device)
    model.eval()

    val_pred_norm, val_true_norm = predict_loader_norm(model, val_loader, device)
    pred_norm, true_norm = predict_loader_norm(model, test_loader, device)

    target_min = min_vals[-1]
    target_max = max_vals[-1]

    val_pred_denorm = unnormalize_data(val_pred_norm, target_min, target_max)
    val_true_denorm = unnormalize_data(val_true_norm, target_min, target_max)
    test_pred_denorm = unnormalize_data(pred_norm, target_min, target_max)
    test_true_denorm = unnormalize_data(true_norm, target_min, target_max)

    val_pred_denorm = np.maximum(0, ensure_2d_array(val_pred_denorm))
    val_true_denorm = np.maximum(0, ensure_2d_array(val_true_denorm))
    test_pred_denorm = np.maximum(0, ensure_2d_array(test_pred_denorm))
    test_true_denorm = np.maximum(0, ensure_2d_array(test_true_denorm))

    pred = test_pred_denorm.reshape(-1)
    true = test_true_denorm.reshape(-1)

    ws_alpha = float(config.get('eval', {}).get('ws_alpha', 0.1))
    ws_summary = compute_conformal_winkler_metrics(
        val_true=val_true_denorm,
        val_pred=val_pred_denorm,
        test_true=test_true_denorm,
        test_pred=test_pred_denorm,
        alpha=ws_alpha,
    )
    overall_ws = float(ws_summary['overall_ws'])
    overall_coverage = float(ws_summary['overall_coverage'])
    overall_sharpness = float(ws_summary['overall_sharpness'])
    q_radius_arr = np.asarray(ws_summary['q_radius'], dtype=np.float64).reshape(-1)
    q_radius_value = float(q_radius_arr[0]) if q_radius_arr.size == 1 else [float(v) for v in q_radius_arr]

    mse, rmse, mae, nrmse, nmae, max_true = eval_metrics(true, pred)
    print(
        'MSE:{:.6f}, RMSE:{:.6f}, MAE:{:.6f}, nRMSE:{:.6f}, nMAE:{:.6f}, y_max:{:.6f}, '
        'WS:{:.6f}, Coverage:{:.6f}, Sharpness:{:.6f}, alpha:{:.3f}, q_radius:{}'.format(
            mse, rmse, mae, nrmse, nmae, max_true,
            overall_ws, overall_coverage, overall_sharpness, ws_alpha, q_radius_value
        ),
        flush=True
    )

    metrics_path = os.path.join(exp_dir, 'test_metrics.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write('MSE:{:.6f}\n'.format(mse))
        f.write('RMSE:{:.6f}\n'.format(rmse))
        f.write('MAE:{:.6f}\n'.format(mae))
        f.write('nRMSE:{:.6f}\n'.format(nrmse))
        f.write('nMAE:{:.6f}\n'.format(nmae))
        f.write('y_max:{:.6f}\n'.format(max_true))
        f.write('WS:{:.6f}\n'.format(overall_ws))
        f.write('Coverage:{:.6f}\n'.format(overall_coverage))
        f.write('Sharpness:{:.6f}\n'.format(overall_sharpness))
        f.write('WS_alpha:{:.6f}\n'.format(ws_alpha))
        if isinstance(q_radius_value, list):
            f.write('WS_q_radius:{}\n'.format(','.join('{:.6f}'.format(v) for v in q_radius_value)))
        else:
            f.write('WS_q_radius:{:.6f}\n'.format(q_radius_value))
    print(f'测试指标已保存到: {metrics_path}', flush=True)

    export_test_predictions_csv(config, exp_dir, test_df, feature_columns, sequence_length, pred, true)

    plot_points = int(config.get('eval', {}).get('plot_points', 300))
    plt.figure('Test_Results')
    xz = list(range(min(len(pred), plot_points)))
    plt.plot(xz, pred[:plot_points], label='test_pred')
    plt.plot(xz, true[:plot_points], label='test_true')
    plt.xlabel('time')
    plt.ylabel('power')
    plt.legend()
    fig_path = os.path.join(exp_dir, 'Test_results.png')
    plt.savefig(fig_path)
    plt.close()
    print(f'测试结果图已保存到: {fig_path}', flush=True)


def read_metrics_file(metrics_path):
    metrics = {}
    if not os.path.exists(metrics_path):
        return metrics
    with open(metrics_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' not in line:
                continue
            key, value = line.strip().split(':', 1)
            metrics[key] = value
    return metrics


def batch_eval_flow(config, script_dir):
    runtime_meta = config.get('_runtime', {})
    root_task_dir = runtime_meta.get('task_dir')
    if not root_task_dir:
        raise ValueError('batch_eval requires a task directory')

    paths_cfg = config.get('paths', {})
    batch_cfg = config.get('batch_eval', {})
    path_base = get_path_base(config, script_dir)
    batch_data_dir = resolve_path(paths_cfg.get('batch_data_dir'), path_base)
    if not batch_data_dir:
        raise ValueError('paths.batch_data_dir must be set for batch_eval mode')
    if not os.path.isdir(batch_data_dir):
        raise FileNotFoundError(f'batch_data_dir not found: {batch_data_dir}')

    pattern = str(batch_cfg.get('pattern', '*.csv'))
    csv_files = sorted(glob.glob(os.path.join(batch_data_dir, pattern)))
    if not csv_files:
        raise FileNotFoundError(f'No files matched pattern {pattern} in {batch_data_dir}')

    print(f'batch_data_dir: {batch_data_dir}', flush=True)
    print(f'batch files: {len(csv_files)}', flush=True)

    rows = []
    for idx, csv_path in enumerate(csv_files, start=1):
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        safe_name = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in base_name)
        item_dir = os.path.join(root_task_dir, f'item_{idx:03d}_{safe_name}')
        os.makedirs(item_dir, exist_ok=False)

        print('=' * 80, flush=True)
        print(f'batch item {idx}/{len(csv_files)}: {csv_path}', flush=True)
        print(f'item output: {item_dir}', flush=True)

        item_config = copy.deepcopy(config)
        item_config['mode'] = 'eval'
        item_config.setdefault('paths', {})['test_csv'] = csv_path
        item_config['_runtime']['task_dir'] = item_dir

        eval_flow(item_config, script_dir)

        metrics_path = os.path.join(item_dir, 'test_metrics.txt')
        row = {'item': idx, 'name': base_name, 'test_csv': csv_path, 'item_dir': item_dir}
        row.update(read_metrics_file(metrics_path))
        rows.append(row)

    summary_path = os.path.join(root_task_dir, 'summary.csv')
    pd.DataFrame(rows).to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f'batch summary saved to: {summary_path}', flush=True)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config, config_path = load_config(script_dir)
    mode = str(config.get('mode', 'train')).strip().lower()
    task_dir = create_task_dir(config, config_path, script_dir)
    log_path = setup_task_logger(task_dir)
    config_snapshot_path = save_config_snapshot(config, task_dir)

    print(f'config: {config_path}', flush=True)
    print(f'task_dir: {task_dir}', flush=True)
    print(f'task_log: {log_path}', flush=True)
    print(f'config_snapshot: {config_snapshot_path}', flush=True)

    if mode == 'train':
        train_flow(config, script_dir)
    elif mode in ('eval', 'evaluate', 'test'):
        eval_flow(config, script_dir)
    elif mode in ('batch_eval', 'batch-eval', 'batch'):
        batch_eval_flow(config, script_dir)
    else:
        raise ValueError(f'不支持的 mode: {mode}')


if __name__ == '__main__':
    main()
