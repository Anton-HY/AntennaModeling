import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import pyswarms as ps

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

# 使用粒子群优化（PSO）算法进行寻优
def pso_objective(x):
    # x: (n_particles, 8)
    n_particles = x.shape[0]
    results = []
    for i in range(n_particles):
        params = {f'param_{j}': x[i, j] for j in range(8)}
        results.append(-objective(**params))  # PSO求最小化，返回负值
    return np.array(results)

dimensions = 8
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
bounds = (np.zeros(dimensions), np.ones(dimensions))
optimizer_pso = ps.single.GlobalBestPSO(n_particles=20, dimensions=dimensions, options=options, bounds=bounds)

print("Starting PSO optimization...")
start_time = time.time()
cost, pos = optimizer_pso.optimize(pso_objective, iters=100)
end_time = time.time()
elapsed_time = end_time - start_time
print("Optimization time: {:.6f} seconds".format(elapsed_time))
print("PSO best position:", pos)
print("PSO best cost:", cost)

# 将PSO最优解转为最佳设计，并反归一化
best_norm_vector = pos  # PSO返回的是最佳解向量
best_actual = best_norm_vector * (features_max.flatten() - features_min.flatten()) + features_min.flatten()
print("Best design (actual geometry parameters):")
print(best_actual)

# 绘制寻优过程的曲线
iterations = np.arange(1, len(optimizer_pso.cost_history) + 1)
values = -np.array(optimizer_pso.cost_history)
best_so_far = np.maximum.accumulate(values)  # PSO 寻优最小值，取负值后变最大值

plt.figure()
plt.plot(iterations, best_so_far, 'r-', label='Best Value So Far')
plt.xlabel('Iteration')
plt.ylabel('Best Value So Far')
plt.title('Particle Swarm Optimization Process')
plt.legend()
plt.show()
