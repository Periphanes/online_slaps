import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dir-result', type=str, default='.')
parser.add_argument('--project-name', type=str, default='proj')
parser.add_argument('--seed', type=int, default=1024)
parser.add_argument('--cpu', type=bool, default=False)
parser.add_argument('--dataset', type=str, default='facebookpagepage')

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--model', type=str, default="baseline_mlp")
parser.add_argument('--lr_init', type=float, default=1e-4)
parser.add_argument('--val-epoch', type=int, default=10)

parser.add_argument('--graph-sample-layers', type=int, default=2)

args = parser.parse_args()
args.dir_root = os.getcwd()