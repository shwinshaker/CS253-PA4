import torch
import torch.nn as nn
from music_dataloader import createLoaders
import numpy as np
import rnn
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

epoch = 30
learning_rate = 0.01
hidden_size = 100

# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 1, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")





# load data
train_loader, val_loader, test_loader, one_hot_length = createLoaders(extras=extras)

RNN = rnn.RNN(hidden_size, one_hot_length, computing_device)
RNN = RNN.to(computing_device)
print("Model on CUDA?", next(RNN.parameters()).is_cuda)
print("Model on CUDA?", next(RNN.parameters()).is_cuda, file=open("output.txt", "a"))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(RNN.parameters())

# Track the loss across training
chunk_train_loss = []

# calculate training and validation loss per N times through the whole training process
batch_train_loss=[]
batch_validation_loss = []

best_val_loss = math.inf

# record training and validation loss per epoch for plotting
plot_train_loss = []
plot_validation_loss = []
for e in range(epoch):

    N = 2000
    # calculate training and validation loss per N times, reinitialize at the beginning of every epoch
    epoch_train_loss = []
    epoch_val_loss = []

    for count, (input,target) in enumerate(train_loader,0):
        # Put the minibatch data in CUDA Tensors and run on the GPU if supported
        input = input[0]
        target = target[0]
        input, target = input.to(computing_device), target.to(computing_device)
        # Zero out the stored gradient (buffer) from the previous iteration
        optimizer.zero_grad()

        # Perform the forward pass through the network and compute the loss
        output = RNN(input)
        output = output.to(computing_device)

        loss = criterion(output,torch.max(target.long(),1)[1])

        # Automagically compute the gradients and backpropagate the loss through the network
        loss.backward()

        # Update the weights
        optimizer.step()

        chunk_train_loss.append(loss.item())

        if count % N == 0 and count != 0 :
            with torch.no_grad():
                # calculating validation error with the weights updated after the 500th minibatch.
                N_loss_val = 0.0

                for count_val, (input_val, target_val) in enumerate(val_loader, 0):
                    # Put the minibatch data in CUDA Tensors and run on the GPU if supported
                    input_val = input_val[0]
                    target_val = target_val[0]
                    input_val, target_val = input_val.to(computing_device), target_val.to(computing_device)
                    output_val = RNN(input_val)

                    output_val= output_val.to(computing_device)

                    loss_val = criterion(output_val, torch.max(target_val.long(),1)[1])
                    N_loss_val += loss_val

                batch_train_loss.append(loss.item())
                epoch_train_loss.append(loss.item())
                print('Epoch %d, count_index %d training loss: %.3f' %
                      (e + 1, count, loss.item()))
                print('Epoch %d, count_index %d training loss: %.3f' %
                      (e + 1, count, loss.item()),file=open("output.txt", "a"))


                N_loss_val /= (count_val+1)
                batch_validation_loss.append(N_loss_val)
                epoch_val_loss.append(N_loss_val)
                print('Epoch %d, count_index %d validation loss: %.3f' %
                      (e + 1, count, N_loss_val))

                print('Epoch %d, count_index %d validation loss: %.3f' %
                      (e + 1, count, N_loss_val),file=open("output.txt", "a"))



    print("Finished", e + 1, "epochs of training")
    print("Finished", e + 1, "epochs of training",file=open("output.txt", "a"))


    # save model
    if (N_loss_val < best_val_loss):
        torch.save(RNN, 'model.ckpt')
        best_val_loss = N_loss_val
    # else:
    #     for g in optimizer.param_groups:
    #         g['lr'] = g['lr'] * 0.1

    print('Epoch %d, training loss: %.3f' %
          (e + 1, sum(epoch_train_loss)/len(epoch_train_loss)))
    plot_train_loss.append(sum(epoch_train_loss)/len(epoch_train_loss))

    print('Epoch %d, training loss: %.3f' %
          (e + 1, sum(epoch_train_loss) / len(epoch_train_loss)),file=open("output.txt", "a"))


    print('Epoch %d, validation loss: %.3f' %
          (e + 1, sum(epoch_val_loss)/len(epoch_val_loss)))

    print('Epoch %d, validation loss: %.3f' %
          (e + 1, sum(epoch_val_loss)/len(epoch_val_loss)), file=open("output.txt", "a"))

    plot_validation_loss.append(sum(epoch_val_loss)/len(epoch_val_loss))


def plot(i1,i2):
  x=np.linspace(1,len(i1),len(i1))
  plt.xlabel('epoch')
  plt.title('loss on training and validation set')
  plt.plot(x,i1,label='train')
  plt.plot(x,i2,label='validation')
  plt.legend()
  plt.savefig('training-validation-loss')

plot(plot_train_loss,plot_validation_loss)

# evaluate on the test set
model = torch.load('model.ckpt')
N_loss_test = 0.0
for count_test, (input_test, target_test) in enumerate(test_loader, 0):
    # Put the minibatch data in CUDA Tensors and run on the GPU if supported
    input_test = input_test[0]
    target_test = target_test[0]
    input_test, target_test = input_test.to(computing_device), target_test.to(computing_device)
    output_test = model(input_test)

    output_test = output_test.to(computing_device)

    loss_test = criterion(output_test, torch.max(target_test.long(),1)[1])
    N_loss_test += loss_test

N_loss_test /= (count_test+1)
print('training loss on the test set: %.3f' % N_loss_test)
print('training loss on the test set: %.3f' % N_loss_test, file=open("output.txt", "a"))



