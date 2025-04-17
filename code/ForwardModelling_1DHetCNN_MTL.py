import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
import copy
import time
from datetime import datetime
import csv

# 数据集类
class AntennaDataset(Dataset):
    def __init__(self, features, targets_full, targets_focus1, targets_focus2):
        self.features = features  # 天线几何作为输入
        self.targets_full = targets_full  # 全频段S11曲线参数作为输出
        self.targets_focus1 = targets_focus1 # 重点频段
        self.targets_focus2 = targets_focus2

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # 返回天线几何参数作为输入，S11参数作为目标
        inputs = self.features[idx]
        target_full = self.targets_full[idx]
        target_focus1 = self.targets_focus1[idx]
        target_focus2 = self.targets_focus2[idx]
        return inputs, target_full, target_focus1, target_focus2

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

def main():
    # 加载CSV文件
    csv_file = 'data/S11.csv'
    data_df = pd.read_csv(csv_file)

    # 提取特征和目标
    targets = data_df.iloc[:, 8:229].values  # 提取S11曲线数据作为目标
    features = data_df.iloc[:, 0:8].values  # 提取天线几何参数作为输入

    # 提取重点频段数据
    index1 = 7  # 对应2.6GHz的索引
    index2 = 152  # 对应17.1GHz的索引
    index3 = 203  # 对应22.2GHz的索引

    targets_focus1 = targets[:, index1:index2]
    targets_focus2 = targets[:, index2:index3]

    # 数据归一化（天线几何参数）
    features_min = features.min(axis=0, keepdims=True)
    features_max = features.max(axis=0, keepdims=True)
    features_normalized = ((features - features_min) / (features_max - features_min)).astype(np.float32)

    targets_min = targets.min()
    targets_max = targets.max()
    targets_normalized = ((targets - targets_min) / (targets_max - targets_min)).astype(np.float32)

    targets_focus_min1 = targets_focus1.min()
    targets_focus_max1 = targets_focus1.max()
    targets_focus_normalized1 = ((targets_focus1 - targets_focus_min1) / (targets_focus_max1 - targets_focus_min1)).astype(np.float32)

    targets_focus_min2 = targets_focus2.min()
    targets_focus_max2 = targets_focus2.max()
    targets_focus_normalized2 = ((targets_focus2 - targets_focus_min2) / (targets_focus_max2 - targets_focus_min2)).astype(np.float32)

    np.save("ForwardModelling/HetCNN_1D_MLT/features_min.npy", features_min)
    np.save("ForwardModelling/HetCNN_1D_MLT/features_max.npy", features_max)

    np.save("ForwardModelling/HetCNN_1D_MLT/targets_min.npy", targets_min)
    np.save("ForwardModelling/HetCNN_1D_MLT/targets_max.npy", targets_max)

    np.save("ForwardModelling/HetCNN_1D_MLT/targets_focus_min1.npy", targets_focus_min1)
    np.save("ForwardModelling/HetCNN_1D_MLT/targets_focus_max1.npy", targets_focus_max1)

    np.save("ForwardModelling/HetCNN_1D_MLT/targets_focus_min2.npy", targets_focus_min2)
    np.save("ForwardModelling/HetCNN_1D_MLT/targets_focus_max2.npy", targets_focus_max2)

    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建数据集
    dataset = AntennaDataset(features_normalized, targets_normalized, targets_focus_normalized1, targets_focus_normalized2)

    # 将数据集按8:1比例分割为训练集、验证集和测试集
    train_size = 512
    val_size = 64

    # 使用random_split函数进行数据集划分
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    test_dataset = val_dataset

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = HetCNN_1d_Forward(input_size=8, output_size_full=221, output_size_focus1=145, output_size_focus2=51).to(device)
    criterion_full = nn.SmoothL1Loss()  # 全频段任务的损失
    criterion_focus1 = nn.SmoothL1Loss()  # 重点频段任务的损失
    criterion_focus2 = nn.SmoothL1Loss()

    # 加权系数
    alpha_full = 0.1  # 全频段任务的权重
    alpha_focus1 = 0.4  # 重点频段任务的权重
    alpha_focus2 = 0.5

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=50, verbose=True)

    # 记录训练和验证结果
    train_losses = []
    val_losses = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')

    # 记录开始时间
    start_time = time.time()
    loss_csv_path = f'ForwardModelling/HetCNN_1D_MLT/HetCNN_1D_MLT_Loss.csv'
    # 创建 CSV 文件，并写入表头
    #with open(loss_csv_path, mode='w', newline='') as file:
    #    writer = csv.writer(file)
    #    writer.writerow(["Epoch", "Train Loss", "Val Loss"])

    # 训练模型
    epochs = 1000
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for inputs, labels_full, labels_focus1, labels_focus2 in train_loader:
            data_batch = inputs.unsqueeze(1).to(device).float()
            labels_full = labels_full.to(device).float()
            labels_focus1 = labels_focus1.to(device).float()
            labels_focus2 = labels_focus2.to(device).float()
            optimizer.zero_grad()
            outputs_full, outputs_focus1, outputs_focus2 = model(data_batch)

            # 计算任务的损失
            loss_full = criterion_full(outputs_full, labels_full)
            loss_focus1 = criterion_focus1(outputs_focus1, labels_focus1)
            loss_focus2 = criterion_focus2(outputs_focus2, labels_focus2)

            # 加权总损失
            loss = alpha_full * loss_full + alpha_focus1 * loss_focus1 + alpha_focus2 * loss_focus2
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        # 验证模型
        model.eval()
        running_val_loss = 0.0

        # 分别记录全频段和重点频段的预测值与实际值
        val_predictions_full = []
        val_actuals_full = []

        val_predictions_focus1 = []
        val_actuals_focus1 = []
        val_predictions_focus2 = []
        val_actuals_focus2 = []

        with torch.no_grad():
            for inputs, labels_full, labels_focus1, labels_focus2 in val_loader:
                data_batch = inputs.unsqueeze(1).to(device).float()
                labels_full = labels_full.to(device).float()
                labels_focus1 = labels_focus1.to(device).float()
                labels_focus2 = labels_focus2.to(device).float()

                outputs_full, outputs_focus1, outputs_focus2 = model(data_batch)

                loss_full = criterion_full(outputs_full, labels_full)
                loss_focus1 = criterion_focus1(outputs_focus1, labels_focus1)
                loss_focus2 = criterion_focus2(outputs_focus2, labels_focus2)

                loss = alpha_full * loss_full + alpha_focus1 * loss_focus1 + alpha_focus2 * loss_focus2
                running_val_loss += loss.item()

                # 记录预测和实际值，用于计算 R²
                # 记录全频段预测和实际值
                val_predictions_full.append(outputs_full.cpu().numpy())
                val_actuals_full.append(labels_full.cpu().numpy())

                # 记录重点频段预测和实际值
                val_predictions_focus1.append(outputs_focus1.cpu().numpy())
                val_actuals_focus1.append(labels_focus1.cpu().numpy())
                val_predictions_focus2.append(outputs_focus2.cpu().numpy())
                val_actuals_focus2.append(labels_focus2.cpu().numpy())

        # 计算验证集平均损失
        train_loss = running_train_loss / len(train_loader)
        val_loss = running_val_loss / len(val_loader)

        # 保存损失，确保变量是全局列表
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        #with open(loss_csv_path, mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([epoch + 1, train_loss, val_loss])

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")

        # 更新学习率调度器
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    # 记录结束时间
    end_time = time.time()
    total_time = end_time - start_time
    total_time_str = time.strftime("%H-%M-%S", time.gmtime(total_time))

    # 恢复最佳模型权重
    model.load_state_dict(best_model_wts)

    # 获取当前时间并格式化，用于文件命名
    current_time = datetime.now().strftime('%Y%m%d_%H-%M-%S')

    # 保存模型
    model_path = f'ForwardModelling/HetCNN_1D_MLT/HetCNN_1D_MLT_{current_time}_duration_{total_time_str}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

    # 绘制训练的损失曲线
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(-0.1, 0.3)
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    # 在测试集上进行预测
    model.eval()
    test_predictions_full = []
    test_actuals_full = []
    test_predictions_focus1 = []
    test_actuals_focus1 = []
    test_predictions_focus2 = []
    test_actuals_focus2 = []

    with torch.no_grad():
        for inputs, labels_full, labels_focus1, labels_focus2 in test_loader:
            data_batch = inputs.unsqueeze(1).to(device).float()
            labels_full = labels_full.to(device).float()
            labels_focus1 = labels_focus1.to(device).float()
            labels_focus2 = labels_focus2.to(device).float()

            outputs_full, outputs_focus1, outputs_focus2 = model(data_batch)
            test_predictions_full_batch = outputs_full.cpu().numpy()
            test_actuals_full_batch = labels_full.cpu().numpy()
            test_predictions_focus_batch1 = outputs_focus1.cpu().numpy()
            test_actuals_focus_batch1 = labels_focus1.cpu().numpy()
            test_predictions_focus_batch2 = outputs_focus2.cpu().numpy()
            test_actuals_focus_batch2 = labels_focus2.cpu().numpy()

            # 累积测试集的预测值和实际值
            test_predictions_full.append(test_predictions_full_batch)
            test_actuals_full.append(test_actuals_full_batch)
            test_predictions_focus1.append(test_predictions_focus_batch1)
            test_actuals_focus1.append(test_actuals_focus_batch1)
            test_predictions_focus2.append(test_predictions_focus_batch2)
            test_actuals_focus2.append(test_actuals_focus_batch2)

    test_predictions_full = np.vstack(test_predictions_full)  # 全频段
    test_actuals_full = np.vstack(test_actuals_full)

    # 重点频段
    test_predictions_focus1 = np.vstack(test_predictions_focus1)
    test_actuals_focus1 = np.vstack(test_actuals_focus1)
    test_predictions_focus2 = np.vstack(test_predictions_focus2)
    test_actuals_focus2 = np.vstack(test_actuals_focus2)

    # 反归一化 S11 数据
    test_predictions_original = test_predictions_full * (targets_max - targets_min) + targets_min
    test_actuals_original = test_actuals_full * (targets_max - targets_min) + targets_min

    # 绘制散点图（测试集数据576组中任意一组）
    sample_index = 1  # 可根据需求调整
    plt.figure(figsize=(8, 6))

    plt.scatter(
        range(test_actuals_original.shape[1]),
        test_actuals_original[sample_index],
        label='Actual', alpha=0.7, marker='o', color='blue')

    plt.scatter(
        range(test_predictions_original.shape[1]),
        test_predictions_original[sample_index],
        label='Predicted', alpha=0.7, marker='x', color='red')

    plt.xlabel('Frequency (GHz)')
    plt.ylabel('S11 Parameter (dB)')
    plt.legend()
    plt.title(f'Scatter Plot for Sample {sample_index}')
    plt.show()

    # 计算测试集的 MSE RMSE MAE 和 R²
    test_mse_full = np.mean((test_predictions_full - test_actuals_full) ** 2)
    test_mse_focus1 = np.mean((test_predictions_focus1 - test_actuals_focus1) ** 2)
    test_mse_focus2 = np.mean((test_predictions_focus2 - test_actuals_focus2) ** 2)
    test_mse = alpha_full * test_mse_full + alpha_focus1 * test_mse_focus1 + alpha_focus2 * test_mse_focus2

    test_rmse_full = np.sqrt(test_mse_full)
    test_rmse_focus1 = np.sqrt(test_mse_focus1)
    test_rmse_focus2 = np.sqrt(test_mse_focus2)
    test_rmse = alpha_full * test_rmse_full + alpha_focus1 * test_rmse_focus1 + alpha_focus2 * test_rmse_focus2

    test_mae_full = np.mean(np.abs(test_predictions_full - test_actuals_full))
    test_mae_focus1 = np.mean(np.abs(test_predictions_focus1 - test_actuals_focus1))
    test_mae_focus2 = np.mean(np.abs(test_predictions_focus2 - test_actuals_focus2))
    test_mae = alpha_full * test_mae_full + alpha_focus1 * test_mae_focus1 + alpha_focus2 * test_mae_focus2


    print(f"Test Mean Squared Error (MSE): {test_mse:.6f}")
    print(f"Test Root Mean Squared Error (RMSE): {test_rmse:.6f}")
    print(f"Test Mean Absolute Error (MAE): {test_mae:.6f}")


if __name__ == '__main__':
    main()