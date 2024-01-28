import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATConv, GCNConv, HeteroConv, Linear, SAGEConv

dataset = OGB_MAG(root="dataset", preprocess="metapath2vec", transform=T.ToUndirected())
data = dataset[0]

train_loader = NeighborLoader(
    data,
    # Sample 15 neighbors for each node and each edge type for 2 iterations:
    num_neighbors={key: [15] * 2 for key in data.edge_types},
    # Use a batch size of 128 for sampling training nodes of type "paper":
    batch_size=128,
    input_nodes=("paper", data["paper"].train_mask),
)


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    ("paper", "cites", "paper"): GCNConv(-1, hidden_channels),
                    ("author", "writes", "paper"): SAGEConv((-1, -1), hidden_channels),
                    ("paper", "rev_writes", "author"): GATConv(
                        (-1, -1), hidden_channels, add_self_loops=False
                    ),
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return self.lin(x_dict["author"])


model = HeteroGNN(hidden_channels=64, out_channels=dataset.num_classes, num_layers=2)
# print(model)
print(data)
with torch.no_grad():  # Initialize lazy modules.
    out = model(data.x_dict, data.edge_index_dict)
    print(data.x_dict.keys())

    # 计算准确率
    # pred = out.argmax(dim=-1)
    # acc = pred.eq(data.y_dict["paper"]).sum().item() / data["paper"].sum().item()
    # print("Accuracy: {:.4f}".format(acc))
