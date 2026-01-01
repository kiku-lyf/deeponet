import os
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import scipy.io
from torch.utils.data import Dataset, DataLoader
import logging

# 设置日志
logger = logging.getLogger("deeponet")
logging.basicConfig(level=logging.INFO)

# 环境变量设置
os.environ["NUMEXPR_MAX_THREADS"] = "20"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ------------------------------------------------------------------------------
# 1. DeepONet 模型定义
# ------------------------------------------------------------------------------
class DeepONet(nn.Module):
    def __init__(self, branch_input_dim=101, trunk_input_dim=2, hidden_dim=50, output_dim=1):
        super().__init__()
        # Branch Net: 处理初始条件 u0 (维度 101)
        self.branch_net = nn.Sequential(
            nn.Linear(branch_input_dim, hidden_dim),
            nn.Tanh(),  # Burgers方程通常Tanh或Sin激活效果更好
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Trunk Net: 处理坐标 (x, t) (维度 2)
        self.trunk_net = nn.Sequential(
            nn.Linear(trunk_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u, y):
        # u: [batch, 101], y: [batch, 2]
        branch_out = self.branch_net(u)
        trunk_out = self.trunk_net(y)
        # 点积操作
        out = torch.sum(branch_out * trunk_out, dim=-1, keepdim=True) + self.bias
        return out


# ------------------------------------------------------------------------------
# 2. Burgers 数据集类
# ------------------------------------------------------------------------------
class BurgersDataset(Dataset):
    def __init__(self, mat_file_path, mode='train', train_ratio=0.8):
        """
        读取 .mat 文件并处理成 (u, y) -> G 格式
        MATLAB数据结构:
        input: (N, 101) -> 初始条件 u0
        output: (N, 101, 101) -> 解 u(t, x) 注意MATLAB里通常是(样本, 时间, 空间)
        tspan: (1, 101)
        x (隐含在代码中): linspace(0, 1, 101)
        """
        if not os.path.exists(mat_file_path):
            raise FileNotFoundError(f"未找到文件: {mat_file_path}，请先运行MATLAB代码生成数据。")

        data = scipy.io.loadmat(mat_file_path)

        # 获取原始数据
        U0 = data['input']  # Shape: (N, 101)
        U_xt = data['output']  # Shape: (N, steps+1, nn) -> (N, 101, 101)
        t_span = data['tspan'].flatten()  # Shape: (101,)

        # 构造空间坐标 x (根据MATLAB代码 nn=101, [0,1])
        nn = U0.shape[1]
        x_span = np.linspace(0, 1, nn)

        num_samples = U0.shape[0]
        split_idx = int(num_samples * train_ratio)

        if mode == 'train':
            self.U0 = U0[:split_idx]
            self.U_xt = U_xt[:split_idx]
        else:
            self.U0 = U0[split_idx:]
            self.U_xt = U_xt[split_idx:]

        # 预处理数据：将数据展平为 (u_sample, [t, x], label) 的形式
        # 为了高效训练，我们需要构建大量的 Pair

        self.u_list = []
        self.y_list = []
        self.G_list = []

        # 生成网格 (t, x)
        # 注意: MATLAB output 是 (N, time, space)
        # Time 轴对应 t_span, Space 轴对应 x_span
        T, X = np.meshgrid(t_span, x_span, indexing='ij')
        # T, X shape: (101, 101)

        # 将网格展平为坐标点 (10201, 2)
        TX_grid = np.stack([X.flatten(), T.flatten()], axis=1)  # [x, t]

        # 展平标签和输入
        # 内存优化提示: 如果数据量极大，建议在 __getitem__ 中动态生成，不要在此处展开
        # 这里为了兼容 DeepONet 简单的 forward 结构，我们直接展开

        logging.info(f"正在处理 {mode} 数据...")

        u_expanded = []
        y_expanded = []
        g_expanded = []

        # 遍历每一个样本轨迹
        for i in range(self.U0.shape[0]):
            # 当前样本的初始条件 u0 (101,)
            u0_curr = self.U0[i]
            # 当前样本的解 (101, 101) -> 展平 -> (10201,)
            # 注意需要与 TX_grid 的顺序对应。TX_grid 是先 Time 后 Space 的 meshgrid 展平
            sol_curr = self.U_xt[i].flatten()

            # 将 u0 重复 10201 次以匹配坐标点数量
            # (这会消耗内存，生产环境通常使用索引映射)
            # 为了演示清晰，此处直接构建
            n_points = TX_grid.shape[0]

            u_expanded.append(np.tile(u0_curr, (n_points, 1)))
            y_expanded.append(TX_grid)
            g_expanded.append(sol_curr[:, None])

        self.u = np.vstack(u_expanded).astype(np.float32)
        self.y = np.vstack(y_expanded).astype(np.float32)
        self.G = np.vstack(g_expanded).astype(np.float32)

        # 转为 Tensor
        self.u = torch.from_numpy(self.u)
        self.y = torch.from_numpy(self.y)
        self.G = torch.from_numpy(self.G)

        logging.info(f"{mode} 数据集形状: u={self.u.shape}, y={self.y.shape}, G={self.G.shape}")

    def __len__(self):
        return self.u.shape[0]

    def __getitem__(self, idx):
        return {"u": self.u[idx], "y": self.y[idx]}, self.G[idx]


def collate_fn(batch):
    inputs = {
        "u": torch.stack([item[0]["u"] for item in batch]),
        "y": torch.stack([item[0]["y"] for item in batch])
    }
    labels = torch.stack([item[1] for item in batch])
    return inputs, labels


# ------------------------------------------------------------------------------
# 3. 训练函数
# ------------------------------------------------------------------------------
def train():
    # 配置参数
    seed = 42
    output_dir = "./output_burgers"
    data_file_path = "Burger2.mat"  # 确保此文件存在
    learning_rate = 1e-3
    epochs = 2000  # 根据数据量调整，Burgers通常需要较多epoch
    batch_size = 5000  # 较大的batch size有助于稳定
    eval_freq = 100

    # 初始化
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型 (Branch Input=101, Trunk Input=2 [x,t])
    model = DeepONet(branch_input_dim=101, trunk_input_dim=2, hidden_dim=50).to(device)

    # 数据检查
    if not os.path.exists(data_file_path):
        print(f"错误: 找不到 {data_file_path}。请运行提供的MATLAB代码生成数据。")
        return

    # 数据加载
    # 注意: 如果MATLAB生成的N=10，train_ratio=0.8意味着只有8个样本用于训练
    # 建议在MATLAB中把 N 改为 1000 左右
    full_dataset_train = BurgersDataset(data_file_path, mode='train')
    full_dataset_valid = BurgersDataset(data_file_path, mode='valid')

    train_loader = DataLoader(full_dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(full_dataset_valid, batch_size=batch_size, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    criterion = nn.MSELoss()

    best_loss = float("inf")

    logger.info("开始训练...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            u = inputs["u"].to(device)
            y = inputs["y"].to(device)  # y is [x, t]
            G_true = labels.to(device)

            optimizer.zero_grad()
            G_pred = model(u, y)
            loss = criterion(G_pred, G_true)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        if epoch % eval_freq == 0:
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    u = inputs["u"].to(device)
                    y = inputs["y"].to(device)
                    G_true = labels.to(device)
                    G_pred = model(u, y)
                    valid_loss += criterion(G_pred, G_true).item()

            avg_train_loss = train_loss / len(train_loader)
            avg_valid_loss = valid_loss / len(valid_loader) if len(valid_loader) > 0 else 0

            logger.info(f"Epoch {epoch:04d} | Train Loss: {avg_train_loss:.4e} | Valid Loss: {avg_valid_loss:.4e}")

            if avg_valid_loss < best_loss:
                best_loss = avg_valid_loss
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))

    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))
    logger.info("Training completed")


# ------------------------------------------------------------------------------
# 4. 结果可视化 (Heatmap)
# ------------------------------------------------------------------------------
def plot_burgers_results(model, data_file, save_dir, device):
    """
    随机选取测试集中的一个样本，绘制真实解 vs 预测解的热力图
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # 加载数据用于提取单个样本
    data = scipy.io.loadmat(data_file)
    U0 = data['input']
    U_xt_true = data['output']  # (N, 101, 101)
    t_span = data['tspan'].flatten()
    x_span = np.linspace(0, 1, 101)

    # 选取最后一个样本作为测试
    idx = -1
    u_test = U0[idx]  # (101,)
    ground_truth = U_xt_true[idx]  # (101, 101) [Time, Space]

    # 构造预测所需的网格输入
    T, X = np.meshgrid(t_span, x_span, indexing='ij')
    # Shape (101, 101). Note: MATLAB output is (Time, Space) usually, verify dims.
    # MATLAB Code: output(j, k, :) = u{k}(X). k is time index.
    # So dim 0 is time, dim 1 is space.

    TX_grid = np.stack([X.flatten(), T.flatten()], axis=1)  # [x, t]
    TX_tensor = torch.tensor(TX_grid, dtype=torch.float32).to(device)

    # 构造 Branch 输入 (重复 u0)
    u_tensor = torch.tensor(u_test, dtype=torch.float32).repeat(TX_tensor.shape[0], 1).to(device)

    with torch.no_grad():
        pred_flat = model(u_tensor, TX_tensor).cpu().numpy()

    # 重塑为 2D 图像 (101, 101)
    pred_img = pred_flat.reshape(101, 101)

    # 绘图
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # Ground Truth
    h1 = ax[0].imshow(ground_truth, interpolation='nearest', cmap='jet',
                      extent=[0, 1, 1, 0], aspect='auto')
    ax[0].set_title("Ground Truth (MATLAB)")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("t")
    plt.colorbar(h1, ax=ax[0])

    # Prediction
    h2 = ax[1].imshow(pred_img, interpolation='nearest', cmap='jet',
                      extent=[0, 1, 1, 0], aspect='auto')
    ax[1].set_title("DeepONet Prediction")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("t")
    plt.colorbar(h2, ax=ax[1])

    # Error
    err = np.abs(ground_truth - pred_img)
    h3 = ax[2].imshow(err, interpolation='nearest', cmap='jet',
                      extent=[0, 1, 1, 0], aspect='auto')
    ax[2].set_title(f"Absolute Error (Max: {err.max():.4f})")
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("t")
    plt.colorbar(h3, ax=ax[2])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "burgers_comparison.png"))
    plt.close()
    logger.info(f"结果图已保存至 {save_dir}")


# ------------------------------------------------------------------------------
# 主程序
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. 确保你有 Burger.mat 文件。如果没有，请先运行你的 MATLAB 代码生成它。
    # 建议在 MATLAB 中将 N 设置得大一些 (例如 1000)，否则训练数据太少。



    if os.path.exists("Burger.mat"):
        train()

        # 加载最佳模型进行绘图
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DeepONet(branch_input_dim=101, trunk_input_dim=2, hidden_dim=50).to(device)
        model.load_state_dict(torch.load('./output_burgers/best_model.pth'))

        plot_burgers_results(model, "Burger.mat", "./result_burgers", device)
    else:
        print("请先在同目录下放置 'Burger.mat' 数据文件。")
