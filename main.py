import torch
from spuco.utils import set_seed
from spuco.robust_train import ERM
from spuco.datasets import SpuCoMNIST, SpuriousFeatureDifficulty
import torchvision.transforms as T
from spuco.group_inference import Cluster, ClusterAlg
from spuco.utils import Trainer
from torch.optim import SGD
from spuco.robust_train import GroupBalanceBatchERM, ClassBalanceBatchERM
from spuco.models import model_factory
from spuco.evaluate import Evaluator
import os

set_seed(0)
device = torch.device("cuda:0")

classes = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
difficulty = SpuriousFeatureDifficulty.MAGNITUDE_LARGE

trainset = SpuCoMNIST(
    root="data/mnist/",
    spurious_feature_difficulty=difficulty,
    spurious_correlation_strength=0.99,
    classes=classes,
    split="train"
)
trainset.initialize()

testset = SpuCoMNIST(
    root="data/mnist/",
    spurious_feature_difficulty=difficulty,
    classes=classes,
    split="test"
)
testset.initialize()

print("Train a model using ERM:")
model = model_factory("lenet", trainset[0][0].shape, trainset.num_classes).to(device)
erm = ERM(
    model=model,
    num_epochs=1,
    trainset=trainset,
    batch_size=64,
    optimizer=SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True),
    device=device,
    verbose=True
)
erm.train()
evaluator = Evaluator(
    testset=testset,
    group_partition=testset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=64,
    model=model,
    device=device,
    verbose=True
)
evaluator.evaluate()
print("The worst group accuracy is {}, \nthe average accuracy is {}, \nthe evaluation of spurious attribute prediction is {}".format(evaluator.worst_group_accuracy, evaluator.average_accuracy, evaluator.evaluate_spurious_attribute_prediction()))
print("Cluster inputs based on the output they produce for ERM")
logits = erm.trainer.get_trainset_outputs()
cluster = Cluster(
    Z=logits,
    class_labels=trainset.labels,
    cluster_alg=ClusterAlg.KMEANS,
    num_clusters=2,
    device=device,
    verbose=True
)
group_partition = cluster.infer_groups()

evaluator = Evaluator(
    testset=trainset,
    group_partition=group_partition,
    group_weights=trainset.group_weights,
    batch_size=64,
    model=model,
    device=device,
    verbose=True
)
evaluator.evaluate()
print("The worst group accuracy is {}, \nthe average accuracy is {}, \nthe evaluation of spurious attribute prediction is {}".format(evaluator.worst_group_accuracy, evaluator.average_accuracy, evaluator.evaluate_spurious_attribute_prediction()))

print(" Retrain using \"Group-Balancing\" to ensure in each batch each group appears equally.")
model = model_factory("lenet", trainset[0][0].shape, trainset.num_classes).to(device)
group_balance_erm = GroupBalanceBatchERM(
    model=model,
    num_epochs=5,
    trainset=trainset,
    group_partition=group_partition,
    batch_size=64,
    optimizer=SGD(model.parameters(), lr=1e-3, weight_decay=5e-4, momentum=0.9, nesterov=True),
    device=device,
    verbose=True
)
group_balance_erm.train()

evaluator = Evaluator(
    testset=testset,
    group_partition=testset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=64,
    model=model,
    device=device,
    verbose=True
)
evaluator.evaluate()
print("The worst group accuracy is {}, \nthe average accuracy is {}, \nthe evaluation of spurious attribute prediction is {}".format(evaluator.worst_group_accuracy, evaluator.average_accuracy, evaluator.evaluate_spurious_attribute_prediction()))
