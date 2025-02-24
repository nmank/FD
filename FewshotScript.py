import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, Flowers102, EuroSAT
from torchvision.models import alexnet
import torch.nn.functional as F
from tqdm import tqdm

from easyfsl.samplers import TaskSampler
from easyfsl.datasets import WrapFewShotDataset

from AlexNetLastTwoLayers import AlexNetLastTwoLayers
from PrototypicalNetworks import PrototypicalNetworks
from PIL import Image

import copy

from statistics import mean

from matplotlib import pyplot as plt

import pandas as pd

import random

def evaluate(model_, val_loader, DEVICE):
    model_.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # No gradients needed during validation
        for val_support_images, val_support_labels, val_query_images, val_query_labels, _ in val_loader:
            # Obtain validation predictions
            val_preds = model_(val_support_images.to(DEVICE), val_support_labels.to(DEVICE), val_query_images.to(DEVICE))
            
            # Count correct predictions
            correct += (val_preds.argmax(dim=2).reshape(-1) == val_query_labels.to(DEVICE)).sum().item()
            total += val_query_labels.size(0)
            

    # Calculate validation accuracy
    val_accuracy = correct / total
    return val_accuracy


if __name__ == '__main__':

    # CHOOSE A DATASET
    # flowers, eurosat, cifar10
    dataset = 'cifar10'

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'your device is {DEVICE}')

    random_seed = 0

    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    N_WAY = 5  # Number of classes in a task
    N_QUERY = 10  # Number of images per class in the query set
    N_EVALUATION_TASKS = 100

    transform = transforms.Compose(
            [
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
            ]
    )


    if dataset == 'flowers':
        test_data = WrapFewShotDataset(
            Flowers102(
            root="../data",
            transform=transform,
            download=True,
            split = 'test')
        )
    elif dataset == 'eurosat':
        test_data = WrapFewShotDataset(
            EuroSAT(
            root="../data",
            transform=transform,
            download=False)
        )
    elif dataset == 'cifar10':
        test_data = WrapFewShotDataset(
            CIFAR10(
            root="../data",
            transform=transform,
            download=True,
            train = False )  
        )


    #takes about 2hr with one GPU
    results = pd.DataFrame(columns = ['Method', 'N Shots', 'Accuracy'])


    for N_SHOT in [3,5,7]:
        print(f'Starting {N_SHOT}')
        test_sampler = TaskSampler(
                test_data, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS
            )

        test_loader = DataLoader(
            test_data,
            batch_sampler=test_sampler,
            num_workers=0,
            pin_memory=True,
            collate_fn=test_sampler.episodic_collate_fn,
            shuffle = False
        )

        for random_seed in tqdm(range(20)):

            # # Eval Protonets
            random.seed(random_seed)
            alexnet_backbone1 = alexnet(pretrained = True)
            alexnet_backbone1.classifier[6] = nn.Flatten()
            proto_model = PrototypicalNetworks(alexnet_backbone1, head = 'ProtoNet').to(DEVICE)
            proto_acc = evaluate(proto_model, test_loader, DEVICE)
            row = pd.DataFrame(columns = results.columns,
                            data = [['ProtoNets', N_SHOT, proto_acc]])
            results = pd.concat([results, row])

            # # Eval Protonets
            random.seed(random_seed)
            alexnet_backbone11 = alexnet(pretrained = True)
            alexnet_backbone11 = AlexNetLastTwoLayers(alexnet_backbone11).to(DEVICE)
            proto_model = PrototypicalNetworks(alexnet_backbone11, head = 'ProtoNet', both = True).to(DEVICE)
            proto_acc = evaluate(proto_model, test_loader, DEVICE)
            row = pd.DataFrame(columns = results.columns,
                            data = [['ProtoNetsMod', N_SHOT, proto_acc]])
            results = pd.concat([results, row])

            # # Eval Subspace Nets
            random.seed(random_seed)
            alexnet_backbone2 = alexnet(pretrained = True)
            alexnet_backbone2.classifier[6] = nn.Flatten()
            subspace_model = PrototypicalNetworks(alexnet_backbone2, head = 'SubspaceNet').to(DEVICE)
            subspace_acc = evaluate(subspace_model, test_loader, DEVICE)
            row = pd.DataFrame(columns = results.columns,
                            data = [['SubspaceNets', N_SHOT, subspace_acc]])
            results = pd.concat([results, row])

            # # Eval Subspace Nets
            random.seed(random_seed)
            alexnet_backbone21 = alexnet(pretrained = True)
            alexnet_backbone21 = AlexNetLastTwoLayers(alexnet_backbone21).to(DEVICE)
            subspace_model = PrototypicalNetworks(alexnet_backbone21, head = 'SubspaceNet', both = True).to(DEVICE)
            subspace_acc = evaluate(subspace_model, test_loader, DEVICE)
            row = pd.DataFrame(columns = results.columns,
                            data = [['SubspaceNetsMod', N_SHOT, subspace_acc]])
            results = pd.concat([results, row])

            # Eval Flag Nets
            random.seed(random_seed)
            alexnet_backbone3 = alexnet(pretrained = True)
            alexnet_backbone3 = AlexNetLastTwoLayers(alexnet_backbone3).to(DEVICE)
            flag_model = PrototypicalNetworks(alexnet_backbone3, head = 'FlagNet', fl_type = [N_SHOT-1,N_SHOT-1],both = True).to(DEVICE)
            flag_acc = evaluate(flag_model, test_loader, DEVICE)
            row = pd.DataFrame(columns = results.columns,
                            data = [['FlagNets', N_SHOT, flag_acc]])
            results = pd.concat([results, row])
            # print(results)

        results.to_csv(f'../results/{dataset}_probing_rebuttal_test.csv')