import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling


# 边分类模型
class EdgeClassifier(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeClassifier, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.classifier = torch.nn.Linear(2 * out_channels, 2)

    def forward(self, x, edge_index):
        x = F.relu(self.conv(x, edge_index))
        pos_edge_index = edge_index
        total_edge_index = torch.cat(
            [
                pos_edge_index,
                negative_sampling(edge_index, num_neg_samples=pos_edge_index.size(1)),
            ],
            dim=1,
        )
        # 在计算边特征时，简单进行源节点特征和目标节点特征的concat。可以考虑其他方法（点乘等等），也可以在这里加MLP用以学习更多的节点-边模式
        edge_features = torch.cat(
            [x[total_edge_index[0]], x[total_edge_index[1]]], dim=1
        )
        return self.classifier(edge_features)


# 加载数据集
dataset = Planetoid(root="./dataset", name="Cora")
data = dataset[0]

# 创建train_mask和test_mask
edges = data.edge_index.t().cpu().numpy()
num_edges = edges.shape[0]
train_mask = torch.zeros(num_edges, dtype=torch.bool)
test_mask = torch.zeros(num_edges, dtype=torch.bool)
train_size = int(0.8 * num_edges)
train_indices = torch.randperm(num_edges)[:train_size]
train_mask[train_indices] = True
test_mask[~train_mask] = True

# 定义模型和优化器/训练/测试
model = EdgeClassifier(dataset.num_features, 64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    pos_edge_index = data.edge_index
    # 手动创建边标签的正例与负例。这是一个二分类问题, 1表示正例, 0表示负例
    pos_labels = torch.ones(pos_edge_index.size(1), dtype=torch.long)
    neg_labels = torch.zeros(pos_edge_index.size(1), dtype=torch.long)
    labels = torch.cat([pos_labels, neg_labels], dim=0).to(logits.device)
    # 在创建模型时是根据所有的边创建正负样本。但是在训练过程当中，只取出train_mask的正负样本计算损失，
    new_train_mask = torch.cat([train_mask, train_mask], dim=0)
    loss = F.cross_entropy(logits[new_train_mask], labels[new_train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test():
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pos_edge_index = data.edge_index
        pos_labels = torch.ones(pos_edge_index.size(1), dtype=torch.long)
        neg_labels = torch.zeros(pos_edge_index.size(1), dtype=torch.long)
        labels = torch.cat([pos_labels, neg_labels], dim=0).to(logits.device)
        new_test_mask = torch.cat([test_mask, test_mask], dim=0)

        predictions = logits[new_test_mask].max(1)[1]
        correct = predictions.eq(labels[new_test_mask]).sum().item()
        return correct / len(predictions)


for epoch in range(1, 1001):
    loss = train()
    acc = test()
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}")
