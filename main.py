import argparse
import torch

parser = argparse.ArgumentParser(description='block')

parser.add_argument('--model', type=str, required=True, default='transformer', help='实验模型选择')
parser.add_argument('--data_path', type=str, required=True, default='/data/', help='数据文件存放路径')
parser.add_argument('--data_name', typr=str, default='data.nc', help='数据文件名')
parser.add_argument('--checkpoints', type=str, default='/checkpoints/', help='模型存放路径')

parser.add_argument('--image_size', type=int, default=256, help='图片大小')
parser.add_argument('--patch_size', type=int, default=32, help='patch尺寸')
parser.add_argument('--dim', type=int, default=1024, help='transformer隐变量纬度')
parser.add_argument('--e_depth', type=int, default=6, help='编码器层数')
parser.add_argument('--d_depth', type=int, default=6, help='解码器层数')
parser.add_argument('--heads', type=int, default=8, help='多头注意力层的head数')
parser.add_argument('--mlp_dim', type=int, default=2048, help='多层感知机的隐藏层')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--emb_dropout', type=float, default=0.1, help='embedding dropout rate')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')

parser.add_argument('--use_gpu', type=bool, default=True, help='是否使用gpu训练')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', type=bool, action='store_true', default=False, help='使用多个gpu并行计算')
parser.add_argument('--devices', type=str, default='0,1', help='多个gpu的device_ids')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]