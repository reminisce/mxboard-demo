import mxnet as mx
from mxboard import SummaryWriter

with SummaryWriter(logdir='./logs') as sw:
    sw.add_graph(mx.sym.load('./data/Inception-BN-symbol.json'))
