import time
import argparse
import os

def get_dataset_name():
    return ['elecNormNew']

model_names = ['DSTNA']
file_names = get_dataset_name()
arg_parser = argparse.ArgumentParser(description='EODL_Enhanced main script')


exp_group = arg_parser.add_argument_group('exp', 'experiment setting')

exp_group.add_argument('-r','--root', default='dataset/',type=str, help='root path to the experiment result' )

exp_group.add_argument('-f','--filename', default='result.json',type=str, help='filename of the experiment results file''(default: result.json)')

exp_group.add_argument('--bs', default=1, type=int,help='batch size')

exp_group.add_argument('--lr', default=0.01, type=float,help='learning rate')

data_group = arg_parser.add_argument_group('data', 'dataset setting')

data_group.add_argument('-d','--dataset',default='elecNormNew',type=str,help='datasets: ' +' | '.join(file_names) + ' (default: elecNormNew)')

net_group = arg_parser.add_argument_group('network', 'network setting')

net_group.add_argument('-m','--model',default='EODL_Enhanced',type=str,help='model architecture: ' +' | '.join(model_names) )

net_group.add_argument('--ln',default=10,type=int,help='layers number (default:10)')

net_group.add_argument('--nHn',default=100, type=int,help='neurons number of hidden layers (default:100)')

net_group.add_argument('--beta', default=0.80, type=float,help='declay factor of prediction weight vector ')

net_group.add_argument('--theta', default=0.01, type=float,help='concept drift detection threshold')

net_group.add_argument('--smooth', default=0.2, type=float,help='concept drift detection threshold')

net_group.add_argument('-p', default=0.99, type=float,help='weighted coefficient of exponential moving average')

net_group.add_argument('--local-pool-size', default=50, type=int,help='maximum size of local memory pool (default:50)')

net_group.add_argument('--global-pool-size', default=100, type=int,help='maximum size of global memory pool (default:100)')

net_group.add_argument('--similarity-threshold', default=0.7, type=float,help='threshold for sample similarity in local memory pool (default:0.85)')

net_group.add_argument('--window-size', default=100, type=int, help='size of sliding window for JS divergence calculation (default:100)')

net_group.add_argument('--kly', default=2.0, type=float, help='coefficient for JS divergence threshold (default:2.0)')

net_group.add_argument('--memory-train-freq', default=500, type=int, help='frequency of training with memory pools (default:50)')