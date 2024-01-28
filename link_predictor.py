"""
链接预测比节点分类更复杂，需要使用节点嵌入对边缘进行预测。
1.编码器通过处理具有两个卷积层的图来创建节点嵌入。
2.在原始图上随机添加负链接。这使得模型任务变为对原始边的正链接和新增边的负链接进行二元分类。
3.解码器使用节点嵌入对所有边(包括负链接)进行链接预测(二元分类)。它从每条边上的一对节点计算节点嵌入的点积。然后聚合整个嵌入维度的值，并在每条边上创建一个表示边存在概率的值。
"""
import torch
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling

dataset = Planetoid(root="dataset", name="Cora")
graph = dataset[0]


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(
            dim=-1
        )  # product of a pair of nodes on each edge

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


def train_link_predictor(
    model, train_data, val_data, optimizer, criterion, n_epochs=100
):
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)

        # sampling training negatives for every training epoch
        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index,
            num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1),
            method="sparse",
        )

        edge_label_index = torch.cat(
            [train_data.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat(
            [
                train_data.edge_label,
                train_data.edge_label.new_zeros(neg_edge_index.size(1)),
            ],
            dim=0,
        )

        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()

        val_auc = eval_link_predictor(model, val_data)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val AUC: {val_auc:.3f}")

    return model


@torch.no_grad()
def eval_link_predictor(model, data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()

    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


# 对于这个链接预测任务，我们希望将链接/边随机分割为训练数据、有效数据和测试数据,使用PyG的RandomLinkSplit模块来做到这一点
split = T.RandomLinkSplit(
    num_val=0.05,
    num_test=0.1,
    is_undirected=True,
    add_negative_train_samples=False,
    neg_sampling_ratio=1.0,
)
"""
1、在edge_index上执行分割，这样训练和验证分割不包括来自验证和测试分割的边(即只有来自训练分割的边)，而测试分割不包括来自测试分割的边。这是因为编码器使用edge_index和x来创建节点嵌入，这种方式确保了在对验证/测试数据进行预测时，节点嵌入上没有目标泄漏。

2、向每个分割数据添加两个新属性(edge_label和edge_label_index)。它们是与每次分割相对应的边标签和边索引。Edge_label_index将用于解码器进行预测，edge_label将用于模型评估。

3、向val_data和test_data添加与正链接相同数量的负链接(neg_sampling_ratio=1.0)。它们被添加到edge_label和edge_label_index属性中，但没有添加到edge_index中，因为我们不希望在编码器(或节点嵌入创建)上使用负链接。我们没有向这里的训练集添加负链接(设置add_negative_train_samples=False)，因为会在上面的train_link_predictor的训练循环中添加它们。训练过程中的这种随机化应该会使模型更健壮。
"""
train_data, val_data, test_data = split(graph)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(dataset.num_features, 128, 64).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()
model = train_link_predictor(model, train_data, val_data, optimizer, criterion)

test_auc = eval_link_predictor(model, test_data)
print(f"Test: {test_auc:.3f}")
