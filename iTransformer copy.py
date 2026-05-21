import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
from typing import Optional, Union, Tuple
from collections import namedtuple


import math
from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import os

# 归一化函数，针对每一列进行归一化，并保存最小值和最大值
def normalize_data(data):
    data_np = np.array(data)
    normalized_data = np.zeros_like(data_np)
    min_vals = np.zeros(data_np.shape[1])
    max_vals = np.zeros(data_np.shape[1])
    
    for i in range(data_np.shape[1]):
        min_val = np.min(data_np[:, i])
        max_val = np.max(data_np[:, i])
        min_vals[i] = min_val
        max_vals[i] = max_val
        normalized_data[:, i] = (data_np[:, i] - min_val) / (max_val - min_val)
    
    return normalized_data, min_vals, max_vals

# 反归一化函数，针对每一列进行反归一化
def unnormalize_data(normalized_data, min_vals, max_vals):
    unnormalized_data = np.zeros_like(normalized_data)   
    unnormalized_data = normalized_data * (max_vals - min_vals) + min_vals
    
    return unnormalized_data

def create_sliding_window(data, sequence_length, n_next):
    X_list, y_list = [], []
    # 确保步长为1
    step = 1  
    # 遍历数据集，创建滑动窗口
    for i in range(0, len(data) - sequence_length - sequence_length + 1, step):
        # 创建长度为sequence_length的窗口
        X_list.append(data.iloc[i:(i + sequence_length):n_next, :].values)

        # 预测窗口后的n_next个数据点，这里n_next=1，所以只预测下一个数据点
        # y_list.append(data.iloc[i + sequence_length + n_next - 1, -1])
        # y_list.append(data.iloc[(i + sequence_length):(i + 2*sequence_length):n_next, -1])
        
        start = i + sequence_length
        end = i + 2 * sequence_length
        # 获取从start到end的所有目标数据
        target_data = data.iloc[start:end, -1].values
        # 将数据分成每组n_next个，然后对每组求和
        # NOTE(Ming)：这里为了兼容求和，简单除去了求和量。后续需要特别注意
        sums = [sum(target_data[j:j+n_next]) / n_next for j in range(0, len(target_data), n_next)]
        y_list.append(sums)

    # 将列表转换为NumPy数组并返回
    return np.array(X_list), np.array(y_list)


# 辅助函数和类
def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

Statistics = namedtuple('Statistics', ['mean', 'variance', 'gamma', 'beta'])

# 简化版Attend模块
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

# RevIN模块
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

# Attention模块
class Attention(nn.Module):
    def __init__(self, dim, dim_head=32, heads=4, dropout=0., flash=True):
        super().__init__()
        self.scale = dim_head ** -0.5
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

# FeedForward模块
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


path = open(r'C:\Users\admin\Desktop\solar\ai_project_template-main\data\environ-data-59-cleaned.csv')
df = pd.read_csv(path,encoding='gbk')
# 时间戳转换
date = pd.to_datetime(df['timeStr'])
df = df.assign(
    date=date,
    month=date.dt.month.astype(int),
    day_of_month=date.dt.day.astype(int),
    hour_of_day=date.dt.hour.astype(int),
    minute_of_day=date.dt.minute.astype(int)
)
# Select and filter columns
datetime_columns = ['date', 'month', 'day_of_month', 'hour_of_day', 'minute_of_day']
environ_columns = ['ambientTemperature', 'relativeHumidity',
                'componentTemperature','tiltTotalRadiation',
                'horizontalDirectRadiation','horizontalScatteredRadiation']
target_columns = ['pvPower']

selected_columns = datetime_columns + environ_columns + target_columns
origin_df = df[selected_columns]
resample_df = origin_df[selected_columns][:1440*100]
# print(resample_df.head())

train_split = 0.95
n_train = int(train_split * len(resample_df))
n_test = len(resample_df) - n_train

features = [item for item in selected_columns if item not in ('date', 'month')]
feature_origin= resample_df[features].values
# print(feature_origin)
feature_norm,  min_vals, max_vals = normalize_data(feature_origin)
# Transfom on both Training and Test data
scaled_array = pd.DataFrame(feature_norm, columns=features)
# print(scaled_array)
sequence_length = 1440
output_length = 96        #多步此处要修改
n_next = 15           #多步此处要修改
X, y = create_sliding_window(scaled_array, 
                            sequence_length, n_next)
X[np.isnan(X)] = 0   #将归一化后为NAN(分母为0)替换为0
y[np.isnan(y)] = 0

X_train = X[:n_train]
y_train = y[:n_train]
# 打乱训练集（保持测试集顺序不变）
shuffle_idx = np.random.permutation(n_train)
X_train = X_train[shuffle_idx]
y_train = y_train[shuffle_idx]

X_test = X[n_train:]
y_test = y[n_train:]


#转化数据类型
X_train = Variable(torch.from_numpy(X_train))
X_train = X_train.float()
y_train = Variable(torch.from_numpy(y_train))
y_train = y_train.float()
 
X_test = Variable(torch.from_numpy(X_test))
X_test = X_test.float()
y_test = Variable(torch.from_numpy(y_test))
y_test = y_test.float()

# 训练集的train_loader
traindataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset=traindataset,batch_size=1440*15, shuffle=True)    #batch_size=100,后续模型训练和测试的时候每次输入模型的维度torch.Size([100, 40, 4, 1])
 
# 测试集的test_loader
testdataset = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = DataLoader(dataset=testdataset, batch_size=1440, shuffle=False)     


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device  = 'cpu'
print("device: {}".format(device))

# 主模型类
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

        # 输入处理
        self.input_proj = nn.Sequential(
            nn.Linear(input_height, hidden_size * self.num_tokens_per_variate),
            Rearrange('b v (n d) -> b (v n) d', n=self.num_tokens_per_variate),
            nn.LayerNorm(hidden_size)
        )

        # 可逆实例归一化
        self.reversible_instance_norm = RevIN(
            num_variates=input_width,
            affine=reversible_instance_norm_affine
        ) if use_reversible_instance_norm else None

        # 记忆token
        self.mem_tokens = nn.Parameter(torch.randn(num_mem_tokens, hidden_size)) if num_mem_tokens > 0 else None

        # Transformer层
        self.layers = ModuleList([
            ModuleList([
                Attention(hidden_size, dim_head=dim_head, heads=heads, dropout=dropout, flash=flash_attn),
                nn.LayerNorm(hidden_size),
                FeedForward(hidden_size, mult=ff_mult, dropout=dropout),
                nn.LayerNorm(hidden_size)
            ]) for _ in range(depth)
        ])

        # 输出投影 - 修改为直接输出 [batch, output_dim]
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size * input_width, hidden_size),  # 合并所有variates的特征
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)  # 最终输出维度
        )

    def forward(self, x: Tensor) -> Tensor:
        """输入: [batch, input_height, input_width]
           输出: [batch, output_dim]
        """
        # 1. 重排输入维度 [batch, input_height, input_width] -> [batch, input_width, input_height]
        x = rearrange(x, 'b n v -> b v n')
        
        # 2. 可逆实例归一化
        if exists(self.reversible_instance_norm):
            x, reverse_fn = self.reversible_instance_norm(x)

        # 3. 输入投影 [batch, input_width, input_height] -> [batch, input_width, hidden_size]
        x = self.input_proj(x)  # [batch, input_width * num_tokens, dim]

        # 4. 添加记忆token
        if exists(self.mem_tokens):
            m = repeat(self.mem_tokens, 'm d -> b m d', b=x.shape[0])
            x, mem_ps = pack([m, x], 'b * d')

        # 5. Transformer处理
        for attn, attn_post_norm, ff, ff_post_norm in self.layers:
            x = attn(x) + x
            x = attn_post_norm(x)
            x = ff(x) + x
            x = ff_post_norm(x)

        # 6. 移除记忆token
        if exists(self.mem_tokens):
            _, x = unpack(x, mem_ps, 'b * d')

        # 7. 可逆实例归一化的逆变换
        if exists(self.reversible_instance_norm):
            x = rearrange(x, 'b (v n) d -> n b v d', n=self.num_tokens_per_variate)
            x = reverse_fn(x)
            x = rearrange(x, 'n b v d -> b (v n) d', n=self.num_tokens_per_variate)

        # 8. 展平并映射到输出维度 [batch, input_width * hidden_size] -> [batch, output_dim]
        x = rearrange(x, 'b v d -> b (v d)')  # 展平所有variates的特征
        output = self.output_proj(x)  # [batch, output_dim]

        return output
        
n_features = scaled_array.shape[-1]

flag = 0

if flag == 0:
    model = iTransformer(
        in_channels=1,
        input_height= output_length,
        input_width= n_features,
        hidden_size=128,
        output_dim= output_length
    )                           
    model = model.to(device)
    criterion = torch.nn.MSELoss().to(device)
    # criterion = torch.nn.SmoothL1Loss().to(device)
    # criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 150
    loss_value=[]
    for epoch in range(n_epochs):
        for i,(input_train_pre,labels_train) in enumerate(train_loader):
            model.train()
            batch_x = Variable(input_train_pre).to(device) 
            batch_y = Variable(labels_train).to(device) 
            output = model(batch_x) 
            loss = criterion(output, batch_y).cpu()
            loss.backward()
            optimizer.step()        
            optimizer.zero_grad() 
            loss_value.append(loss.item())
        print('epoch: ', epoch, 'loss: ', loss.item())
        np.savetxt('Train_loss.txt', loss_value)      

        # if epoch % 10 == 0:
        #   loss_value.append(loss.item())
        #   print('epoch: ', e, 'loss: ', loss.item())
    torch.save(model, r'C:\Users\admin\Desktop\solar\PV_predict.pth', _use_new_zipfile_serialization=False)    
    plt.figure('Train_Loss')
    plt.plot(loss_value,label='Loss')
    plt.legend()
    plt.savefig('Train_Loss.png')
    plt.show()


if flag == 1:
    #测试集评估
    test_predict_his = []
    test_true_his = []
    model = torch.load(r'C:\Users\admin\Desktop\solar\PV_predict.pth',
                  weights_only=False)
    model = model.to(device)
    model.eval()
    for i,(test_x,test_y) in enumerate(test_loader):
        batch_test_x = Variable(test_x).to(device)
        batch_test_y = Variable(test_y)
        test_predict = model(batch_test_x).detach().cpu()
        if i == 0:
            test_predict_his = test_predict
            test_true_his = batch_test_y
        else:
            test_predict_his = torch.cat((test_predict_his, test_predict), dim=0)
            test_true_his = torch.cat((test_true_his, batch_test_y), dim=0)

    # test_predict_plot = unnormalize_data(test_predict_his[-1:], min_vals[-1], max_vals[-1])
    # test_predict_plot = np.maximum(0,test_predict_plot)
    # test_true_plot = unnormalize_data(test_true_his[-1:], min_vals[-1], max_vals[-1])
    # test_true_plot = np.maximum(0,test_true_plot)
    test_predict_plot = unnormalize_data(test_predict_his, min_vals[-1], max_vals[-1])
    test_predict_plot = 15 * np.maximum(0,test_predict_plot)
    test_true_plot = unnormalize_data(test_true_his, min_vals[-1], max_vals[-1])
    test_true_plot = 15 * np.maximum(0,test_true_plot)

    from sklearn import metrics
    def eval_metrix(true,pred):
        MAE = metrics.mean_absolute_error(test_true_his, test_predict_his)
        MAPE = metrics.mean_absolute_percentage_error(test_true_his, test_predict_his)
        MSE = metrics.mean_squared_error(test_true_his, test_predict_his)
        RMSE = metrics.root_mean_squared_error(test_true_his, test_predict_his)
        return [MAE,MAPE,MSE,RMSE]


    # 重塑为 (样本数, 96)
    predictions = test_predict_his.reshape(-1, 96)
    # 提取每个样本的第一个预测值和真实值
    y_true_first = test_true_his[:, 0]        # shape: (138168,)
    y_pred_first = predictions[:, 0]     # shape: (138168,)


    MAE,MAPE,MSE,RMSE = eval_metrix(y_true_first,y_pred_first)
    print('MAE:{:.4e},MAPE:{:.2f}%,MSE:{:.4e},RMSE:{:.4e}'.format(MAE,100*MAPE,MSE,RMSE)) 



    plt.figure('训练集结果')
    xz = list(range(0,test_predict_plot[0].numel()))
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] =False
    plt.plot(xz,test_predict_plot[3].reshape(1,-1).squeeze(0),label='测试集预测值')
    plt.plot(xz,test_true_plot[3].reshape(1,-1).squeeze(0),label='测试集真实值')
    plt.xlabel('time')
    plt.ylabel('发电量')
    plt.legend()
    # # 指定保存的相对路径和文件名
    # save_path = './结果/iTransformer/Test_results.png'
    # #确保路径中的所有文件夹都存在
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # # 保存图像
    # plt.savefig(save_path, format='png')
    plt.show()



# # 测试代码
# if __name__ == "__main__":
#     # 参数设置
#     batch_size = 1440
#     input_height = 96  # 时间步长
#     input_width = 11   # 特征维度
#     hidden_size = 128
#     output_dim = 96
    
#     # 创建模型
#     model = iTransformer(
#         in_channels=1,
#         input_height=input_height,
#         input_width=input_width,
#         hidden_size=hidden_size,
#         output_dim=output_dim
#     )
    
#     # 创建随机输入数据
#     x = torch.randn(batch_size, input_height, input_width)
    
#     # 前向传播测试
#     output = model(x)
    
#     # 验证维度
#     print(f"输入维度: {x.shape}")  # 应该输出: torch.Size([32, 24, 10])
#     print(f"输出维度: {output.shape}")  # 应该输出: torch.Size([32, 5])
    
#     # 参数统计
#     num_params = sum(p.numel() for p in model.parameters())
#     print(f"模型参数量: {num_params:,}")