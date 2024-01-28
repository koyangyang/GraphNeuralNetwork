from collections import Counter

from pygod.detector import DOMINANT
from pygod.utils import load_data
from sklearn.metrics import average_precision_score, roc_auc_score

# 对于这个异常检测任务，需要使用的是PyGOD库，它是建立在PyG之上的一个图异常值检测库。可以通过PyGOD模块加载已经进行了异常值注入的Cora数据集。
graph = load_data("inj_cora")
# 0:正常，1:仅上下文异常，2:结构异常，3:上下文和结构都异常
# Counter({0: 2570, 1: 68, 2: 68, 3: 2})
# 如果你对这些异常值是如何注入的感兴趣，可以查看关于异常值生成器模块的PyGOD文档，该文档解释了操作细节。
# 需要注意的是标签y将只用于模型评估，而不是用于训练标签，因为我们正在训练一个无监督的模型。
print(Counter(graph.y.tolist()))


def train_anomaly_detector(model, graph):
    return model.fit(graph)


def eval_anomaly_detector(model, graph):
    outlier_scores = model.decision_function(graph)
    auc = roc_auc_score(graph.y.numpy(), outlier_scores)
    ap = average_precision_score(graph.y.numpy(), outlier_scores)
    print(f"AUC Score: {auc:.3f}")
    print(f"AP Score: {ap:.3f}")


graph.y = graph.y.bool()
model = DOMINANT()
model = train_anomaly_detector(model, graph)
eval_anomaly_detector(model, graph)
