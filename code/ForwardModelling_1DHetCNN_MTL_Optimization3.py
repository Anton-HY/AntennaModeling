import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# 1D HetConv 模块
class HetConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2, groups=4):

        super(HetConv1d, self).__init__()
        # Groupwise 卷积：使用 kernel_size 卷积核，并按照 groups 参数分组
        self.gwc = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                             padding=padding, groups=groups, bias=False)
        # Pointwise 卷积：1×1 卷积实现跨通道信息融合
        self.pwc = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # 将两部分卷积结果相加
        return self.gwc(x) + self.pwc(x)


# 1D HetCNN 模型
class HetCNN_1d_Forward(nn.Module):
    def __init__(self, input_size, output_size_full, output_size_focus1, output_size_focus2, hetconv_groups=4):

        super(HetCNN_1d_Forward, self).__init__()
        # 输入投影：将每个几何参数（1维向量）投影到 embed_dim 维
        self.hetconv1 = HetConv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2, groups=1)
        self.bn1 = nn.BatchNorm1d(16)

        self.hetconv2 = HetConv1d(in_channels=16, out_channels=16, kernel_size=5, padding=2, groups=hetconv_groups)
        self.bn2 = nn.BatchNorm1d(16)

        self.hetconv3 = HetConv1d(in_channels=16, out_channels=64, kernel_size=5, padding=2, groups=hetconv_groups)
        self.bn3 = nn.BatchNorm1d(64)

        self.hetconv4 = HetConv1d(in_channels=64, out_channels=64, kernel_size=5, padding=2, groups=hetconv_groups)
        self.bn4 = nn.BatchNorm1d(64)

        self.hetconv5 = HetConv1d(in_channels=64, out_channels=256, kernel_size=5, padding=2, groups=hetconv_groups)
        self.bn5 = nn.BatchNorm1d(256)

        self.hetconv6 = HetConv1d(in_channels=256, out_channels=256, kernel_size=5, padding=2, groups=hetconv_groups)
        self.bn6 = nn.BatchNorm1d(256)

        # Global Average Pooling，将长度缩减到1
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接层
        self.fc = nn.Linear(256, 512)

        self.dropout = nn.Dropout1d(0.5)

        self.fc_full = nn.Linear(512, output_size_full)
        self.fc_focus1 = nn.Linear(512, output_size_focus1)
        self.fc_focus2 = nn.Linear(512, output_size_focus2)

        self.leaky_relu = nn.LeakyReLU(negative_slope=1e-2)

    def forward(self, data):
        data = self.leaky_relu(self.bn1(self.hetconv1(data)))
        data = self.leaky_relu(self.bn2(self.hetconv2(data)))
        data = self.leaky_relu(self.bn3(self.hetconv3(data)))
        data = self.leaky_relu(self.bn4(self.hetconv4(data)))
        data = self.leaky_relu(self.bn5(self.hetconv5(data)))
        data = self.leaky_relu(self.bn6(self.hetconv6(data)))

        data = self.global_pool(data).squeeze(-1)  # 变成 (batch_size, 128)
        data = self.dropout(self.fc(data))

        output_full = self.fc_full(data)  # 最终输出 (batch_size, output_size)
        output_focus1 = self.fc_focus1(data)
        output_focus2 = self.fc_focus2(data)
        return output_full, output_focus1, output_focus2

# 加载归一化参数（.npy文件）
features = pd.read_csv('data/S11.csv').iloc[:, 0:8].values
features_min = features.min(axis=0, keepdims=True)
features_max = features.max(axis=0, keepdims=True)

targets_min = np.load("models/10/targets_min.npy")
targets_max = np.load("models/10/targets_max.npy")
targets_focus_min1 = np.load("models/10/targets_focus_min1.npy")
targets_focus_max1 = np.load("models/10/targets_focus_max1.npy")
targets_focus_min2 = np.load("models/10/targets_focus_min2.npy")
targets_focus_max2 = np.load("models/10/targets_focus_max2.npy")

# 定义目标函数
def objective(**params):
    """
    输入8个归一化的几何参数，
    使用前向模型预测重点频段 S11（归一化后的输出），
    反归一化后计算该频段的最大 S11，并对两个重点频段内部进行惩罚：
    惩罚条件：从第一个小于 -10dB 到最后一个小于 -10dB的区间内，若有任一数值大于 -10dB，则施加惩罚。
    最终返回的目标为：
         margin = 平均(10 - max(S11)) - λ * (penalty1 + penalty2)
    其中 λ 为惩罚权重（可调）。
    """
    # 归一化的天线几何参数向量，形状为 (8,)
    geom = np.array([params[f'param_{i}'] for i in range(8)], dtype=np.float32)
    geom_tensor = torch.tensor(geom, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)
    with torch.no_grad():
        _, focus1, focus2 = forward_model(geom_tensor)
    focus_pred_orig1 = focus1.cpu().numpy() * (targets_focus_max1 - targets_focus_min1) + targets_focus_min1
    focus_pred_orig2 = focus2.cpu().numpy() * (targets_focus_max2 - targets_focus_min2) + targets_focus_min2

    # 使其成为一维数组（去掉批次维度）
    region1 = focus_pred_orig1.squeeze()  # 预期形状 (N1,)
    region2 = focus_pred_orig2.squeeze()  # 预期形状 (N2,)

    # 计算原有裕度：用最大值计算 margin = 10 - max(S11)
    # 注意：如果S11数值越低越好，max(S11)应为最高（最不负）的数值
    max_s11_1 = np.max(region1)
    max_s11_2 = np.max(region2)
    margin1 = 10 - max_s11_1
    margin2 = 10 - max_s11_2
    base_margin = (margin1 + margin2) / 2.0

    # 计算惩罚项：
    # 对于每个频段，从第一个低于 -10dB 的数值到最后一个低于 -10dB 的数值，
    # 如果区间内有任意数值高于 -10dB，则计算该数值与 -10dB 的差值之和作为惩罚。
    def compute_penalty(region, threshold=-10):
        indices = np.where(region < threshold)[0]
        penalty = 0.0
        if indices.size > 0:
            first_idx = indices[0]
            last_idx = indices[-1]
            subregion = region[first_idx:last_idx + 1]
            # 计算违反阈值的部分：若数值 > threshold，则罚分为 (value - threshold)
            violations = np.maximum(0, subregion - threshold)
            penalty = np.sum(violations)
        return penalty

    penalty1 = compute_penalty(region1, threshold=-10)
    penalty2 = compute_penalty(region2, threshold=-10)
    lambda_penalty = 1.0  # 惩罚系数，可根据需要调整

    final_margin = base_margin - lambda_penalty * (penalty1 + penalty2)
    return final_margin

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 8
output_size_full = 221
output_size_focus1 = 145
output_size_focus2 = 51
forward_model = HetCNN_1d_Forward(input_size, output_size_full, output_size_focus1, output_size_focus2)
model_path = "models/10/HetCNN_1D_MLT_20250417_11-42-27_duration_00-00-46.pth"
forward_model.load_state_dict(torch.load(model_path, map_location=device))
forward_model.eval()
forward_model.to(device)

# WOA 算法实现
def WOA_Optimization(obj_func, dim, lb, ub, n_agents=20, max_iter=100):
    """
    obj_func: 目标函数，输入为 (dim,) 的向量
    dim: 搜索空间维度（8）
    lb, ub: 搜索空间下界与上界（均为数组，长度为 dim）
    n_agents: 鲸鱼数量
    max_iter: 最大迭代次数
    返回最优解和对应目标值
    """
    positions = np.random.uniform(low=lb, high=ub, size=(n_agents, dim))
    best_score = -np.inf
    best_pos = None
    history = []  # 记录每一轮的最佳目标值
    a_linear_component = 2 / max_iter
    for t in range(max_iter):
        for i in range(n_agents):
            params = {f'param_{j}': positions[i, j] for j in range(dim)}
            fitness = obj_func(**params)
            if fitness > best_score:
                best_score = fitness
                best_pos = positions[i].copy()
        history.append(best_score)
        a = 2 - t * a_linear_component
        for i in range(n_agents):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            A = 2 * a * r1 - a
            C = 2 * r2
            p = np.random.rand()
            if p < 0.5:
                if np.linalg.norm(A) < 1:
                    D = np.abs(C * best_pos - positions[i])
                    positions[i] = best_pos - A * D
                else:
                    random_idx = np.random.randint(0, n_agents)
                    random_pos = positions[random_idx]
                    D = np.abs(C * random_pos - positions[i])
                    positions[i] = random_pos - A * D
            else:
                D_prime = np.linalg.norm(best_pos - positions[i])
                b = 1
                l = np.random.uniform(-1, 1, size=dim)
                positions[i] = best_pos + D_prime * np.exp(b * l) * np.cos(2 * np.pi * l)
            positions[i] = np.clip(positions[i], lb, ub)
    return best_pos, best_score, history

# 定义搜索区间
dim = 8
lb = np.zeros(dim)
ub = np.ones(dim)

print("Starting Whale Optimization...")
start_time = time.time()
best_norm_vector, best_score, history = WOA_Optimization(objective, dim, lb, ub, n_agents=20, max_iter=100)
end_time = time.time()
elapsed_time = end_time - start_time
print("Optimization time: {:.6f} seconds".format(elapsed_time))
print("WOA best normalized parameters:", best_norm_vector)
print("WOA best objective (margin):", best_score)


# 绘制寻优过程的曲线
iterations = np.arange(1, len(history) + 1)
history = np.array(history)
# 如果目标函数值越大越好，使用 np.maximum.accumulate(history)
best_so_far = np.maximum.accumulate(history)

plt.figure()
plt.plot(iterations, best_so_far, 'r-', label='Best Value So Far')
plt.xlabel('Iteration')
plt.ylabel('Best Value So Far')
plt.title('Whale Optimization Process')
plt.legend()
plt.show()

# 反归一化参数
best_actual = best_norm_vector * (features_max.flatten() - features_min.flatten()) + features_min.flatten()
print("Best design (actual geometry parameters):")
print(best_actual)
