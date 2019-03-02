import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from lstm import Evaluation, check_cuda

print('---> check cuda')
use_cuda, _, _ = check_cuda()
print('cuda: %s' % 'yes' if use_cuda else 'no')
if use_cuda: 
	loc = 'cuda'
else: 
	loc = 'cpu'

print('---> loading checkpoint')
checkpoint = torch.load('checkpoint0.pth.tar', map_location=loc)
evaluate = checkpoint['history']
losses = []
print('----> Reading history')
for epoch in evaluate.history:
    print('epoch %i:' % epoch, end=' ')
    count = len(evaluate.history[epoch])
    for i, (train_loss, val_loss) in enumerate(evaluate.history[epoch]):
        print('[%i]' % i, end='->')
        losses.append((epoch-1+(i+1)/count, train_loss, val_loss))
    print('', end='\n')
plotx, train_loss, val_loss = zip(*losses)

# plot
plt.plot(plotx, train_loss, '.-', label='train')
plt.plot(plotx, val_loss, '.-', label='val')
fontsize = 12
plt.legend(fontsize=fontsize)
plt.xlabel('epoch', fontsize=fontsize)
plt.ylabel('loss', fontsize=fontsize)
plt.savefig('loss.png', fontsize=fontsize)
