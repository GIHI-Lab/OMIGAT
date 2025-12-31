#OMIGAT: A Fusion Model with Collaborative Attention and Graph Convolution Enhanced by Multi-Omics Features for Identifying Cancer Driver Genes

---

## 📌 主要特性

- **多图结构处理**: 同时处理先验图和KNN图，捕捉局部和全局关系
- **特征增强注意力**: 自适应融合原始特征、度中心性和聚类系数
- **交互注意力机制**: 实现不同图结构之间的信息交互
- **随机掩码正则化**: 增强模型鲁棒性和泛化能力
- **焦点损失函数**: 有效处理类别不平衡问题
- **双路径架构**: 结合SAGEConv和TransformerConv的优势

---

## 🏗️ 架构概览

### 1. 特征处理模块
- **FeatureWiseAttention**: 特征级注意力机制，动态融合三种特征表示
- **RandomMasking**: 随机掩码技术，防止过拟合

### 2. 双图处理路径
- **先验图路径**: 使用用户定义的图结构
- **KNN图路径**: 基于特征相似度构建的k最近邻图

### 3. 交互注意力层
- **InteractionAttention**: 跨图注意力机制，促进不同图结构间的信息交换

### 4. 融合预测模块
- 自适应权重融合SAGE和Transformer特征
- MLP分类器输出预测结果

---

## 📦 安装要求

```bash
pip install torch torch-geometric
```

**环境要求**:
- Python 3.10
- PyTorch >= 1.9
- PyTorch Geometric >= 2.0
- CUDA支持（可选，用于GPU加速）

---

## 🚀 快速开始

### 1. 模型初始化

```python
from OMIGAT import OMIGATNet

# 初始化模型
model = OMIGATNet(
    in_channels=100,          # 输入特征维度
    hidden_channels=128,      # 隐藏层维度
    num_layers=2,             # 网络层数
    dropout=0.2,              # dropout率
    focal_alpha=0.25,         # 焦点损失alpha参数
    focal_gamma=2.0           # 焦点损失gamma参数
)

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
```

### 2. 准备数据

```python
from torch_geometric.data import Data
import torch

# 示例数据
num_nodes = 1000
in_channels = 100

# 节点特征
x = torch.randn(num_nodes, in_channels)

# 先验图边（用户定义的图结构）
edge_index_prior = torch.randint(0, num_nodes, (2, 2000))

# KNN图边（可自动生成或手动提供）
edge_index_knn = torch.randint(0, num_nodes, (2, 3000))

# 创建数据对象
data = Data(
    x=x,
    edge_index=edge_index_prior,
    edge_index_knn=edge_index_knn
)
```

### 3. 前向传播

```python
# 模型前向传播
output = model(data)
print(f"输出形状: {output.shape}")  # [num_nodes, 1]
```

### 4. 训练模型

```python
# 定义损失函数
target = torch.randint(0, 2, (num_nodes, 1)).float()
loss = model.compute_loss(output, target)

# 反向传播
loss.backward()
print(f"损失值: {loss.item():.4f}")
```

---

## ⚙️ 模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `in_channels` | int | 必需 | 输入特征维度 |
| `hidden_channels` | int | 128 | 隐藏层维度 |
| `num_layers` | int | 2 | SAGE和Transformer层数 |
| `num_heads` | int | 4 | Transformer注意力头数 |
| `dropout` | float | 0.2 | dropout概率 |
| `focal_alpha` | float | 0.25 | 焦点损失alpha参数 |
| `focal_gamma` | float | 2.0 | 焦点损失gamma参数 |

---

## 🔧 核心模块详解

### FeatureWiseAttention
```python
class FeatureWiseAttention(torch.nn.Module):
    """
    特征级注意力机制
    输入:
        - x: 原始节点特征 [N, C]
        - edge_index: 图边索引 [2, E]
    输出:
        - 融合后的特征 [N, C]
    """
```

### InteractionAttention
```python
class InteractionAttention(torch.nn.Module):
    """
    交互注意力机制
    输入:
        - x1: 第一个图特征 [N, C]
        - x2: 第二个图特征 [N, C]
    输出:
        - 交互后的特征 [N, C]
    """
```

### RandomMasking
```python
class RandomMasking(torch.nn.Module):
    """
    随机掩码正则化
    训练时随机屏蔽部分特征
    增强模型鲁棒性
    """
```

---

## 📊 性能优化

### 内存优化
- 特征融合时使用内存高效的计算方式
- 支持mini-batch训练

### 计算效率
- 并行处理多个图结构
- 优化的注意力计算

### 扩展性
- 支持大规模图数据
- 易于添加新的图卷积层

---

## 🎯 应用场景

### 1. 生物信息学
- 蛋白质相互作用预测
- 疾病基因关联分析
- 药物靶点发现

### 2. 社交网络
- 用户关系预测
- 社区检测
- 影响力分析

### 3. 推荐系统
- 用户-物品交互预测
- 协同过滤增强

### 4. 知识图谱
- 实体链接预测
- 关系推理

---

## 📈 实验配置建议

```python
# 推荐的训练配置
config = {
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'epochs': 200,
    'patience': 20,  # 早停耐心值
    'batch_size': 32,
    'k_for_knn': 10,  # KNN图的k值
    'masking_prob': 0.2,  # 随机掩码概率
}
```

---

## 🧪 实验示例

```python
# 完整训练循环示例
import torch.optim as optim
from torch_geometric.loader import DataLoader

# 创建数据加载器
loader = DataLoader([data], batch_size=1, shuffle=True)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 训练循环
for epoch in range(100):
    model.train()
    total_loss = 0
    
    for batch in loader:
        optimizer.zero_grad()
        output = model(batch)
        target = torch.randint(0, 2, (batch.num_nodes, 1)).float()
        loss = model.compute_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}')
```

---

## 🚨 注意事项

1. **输入特征**: 确保输入特征已标准化或归一化
2. **图构建**: KNN图可以自动生成或手动提供
3. **内存使用**: 大规模图建议使用GPU和分批处理
4. **超参数调优**: 根据具体任务调整超参数
5. **类别平衡**: 对于极端不平衡数据，调整焦点损失参数

---

## 🔍 调试建议

```python
# 调试代码示例
def debug_model(model, data):
    """模型调试函数"""
    print(f"输入特征形状: {data.x.shape}")
    print(f"先验图边数: {data.edge_index.shape[1]}")
    print(f"KNN图边数: {data.edge_index_knn.shape[1]}")
    
    # 前向传播调试
    with torch.no_grad():
        output = model(data)
        print(f"输出形状: {output.shape}")
        print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
```

---


## 🤝 贡献指南

我们欢迎各种形式的贡献：

1. **问题报告**: 使用GitHub Issues报告bug或问题
2. **功能请求**: 提出新功能或改进建议
3. **代码贡献**: 提交Pull Request
4. **文档改进**: 帮助完善文档和示例

### 开发环境设置
```bash
# 克隆仓库
git clone https://github.com/your-username/OMIGAT.git
cd OMIGAT

# 安装开发依赖
pip install -r requirements-dev.txt
****
