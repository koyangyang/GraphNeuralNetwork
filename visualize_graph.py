import random

import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx


def visualize_graph(dataset):
    # 可视化图结构
    graph = nx.Graph()  # 创建一个空图

    edge_index = dataset.edge_index  # 获取边的索引

    source = edge_index[0]  # 起点
    target = edge_index[1]  # 终点

    for src, tge in zip(source, target):
        graph.add_edge(src.item(), tge.item())

    plt.rcParams["figure.dpi"] = 300  # 设置分辨率
    fig, ax1 = plt.subplots(figsize=(10, 10))

    nx.draw_networkx(G=graph, ax=ax1, font_size=6, node_size=150)
    plt.show()


def convert_to_networkx(graph, n_sample=None):
    g = to_networkx(graph, node_attrs=["x"])
    y = graph.y.numpy()

    if n_sample is not None:
        sampled_nodes = random.sample(g.nodes, n_sample)
        g = g.subgraph(sampled_nodes)
        y = y[sampled_nodes]

    return g, y


def plot_graph(g):
    g, y = convert_to_networkx(g, n_sample=1000)
    plt.figure(figsize=(9, 7))
    nx.draw_spring(g, node_size=30, arrows=False, node_color=y)
    plt.show()
