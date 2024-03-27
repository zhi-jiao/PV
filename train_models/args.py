import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--num-gpu', type=int, default=0, help='the number of the gpu to use')
parser.add_argument('--epochs', type=int, default=200, help='train epochs')
parser.add_argument('--batch-size', type=int, default=16, help='batch size')
parser.add_argument('--data-path', type=str, default='../levels/high.csv')
parser.add_argument('--filename', type=str, default='high')
parser.add_argument('--train-ratio', type=float, default=0.6, help='the ratio of training dataset')
parser.add_argument('--test-ratio', type=float, default=0.2, help='the ratio of testing dataset')
parser.add_argument('--his-length', type=int, default=6, help='the length of history time series of input')
parser.add_argument('--pred-length', type=int, default=1, help='the length of target time series for prediction')
parser.add_argument('--step', type=int, default=1, help='the length of target time series for prediction')

parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')

parser.add_argument('--log', action='store_true', help='if write log to files')
args = parser.parse_args()