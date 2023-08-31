import os
import argparse
import math

parser = argparse.ArgumentParser()

# General Configurations
parser.add_argument('--dir-result', type=str, default='.')
parser.add_argument('--project-name', type=str, default='proj')
parser.add_argument('--seed', type=int, default=1024)
parser.add_argument('--cpu', type=bool, default=False)
parser.add_argument('--dataset', type=str, default='facebookpagepage')
parser.add_argument('--train-type', type=str, default="features")

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--model', type=str, default="baseline_mlp")
parser.add_argument('--lr_init', type=float, default=1e-4)
parser.add_argument('--val-epoch', type=int, default=10)
parser.add_argument('--input-type', type=str, choices=["features", "sampling", "slaps"], default="features")


parser.add_argument('--relational-train-length', type=int, default=5000)

# GraphSage Model Configurations
parser.add_argument('--graph-sample-layers', type=int, default=2)
parser.add_argument('--knn-k', type=int, default=15)
parser.add_argument('--sage-sample-percent', type=int, default=5)


#SLAPS Model Configurations
parser.add_argument('--slaps-mlp-layers', type=int, default=2)
parser.add_argument('--slaps-mlp-out', type=int, default=40)
parser.add_argument('--slaps-mlp-hidden', type=int, default=80)
parser.add_argument('--slaps-sparse-graph', type=bool, default=True)
parser.add_argument('--slaps-knn-metric', type=str, choices=['cosine', 'euclidean', 'haversine', 'l1', 'manhattan'], default='cosine')
parser.add_argument('--slaps-sparse', type=bool, default=False)
parser.add_argument('--slaps-mlp-epochs', type=int, default=100)

parser.add_argument('--slaps-dae-layers', type=int, default=3)
parser.add_argument('--slaps-dae-hidden', type=int, default=80)
parser.add_argument('--slaps-dae-out', type=int, default=40)
parser.add_argument('--slaps-dae-dropout', type=float, default=0.1)

parser.add_argument('--slaps-gcn-layers', type=int, default=3)
parser.add_argument('--slaps-gcn-hidden', type=int, default=80)
parser.add_argument('--slaps-gcn-out', type=int, default=40)
parser.add_argument('--slaps-gcn-dropout', type=float, default=0.1)


args = parser.parse_args()
args.dir_root = os.getcwd()