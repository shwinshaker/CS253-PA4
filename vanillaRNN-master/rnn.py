import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim

'''
form: X*W

here self.length means the length of the one hot encoding (93)

input: original input dimension in data loader for one example : 100*93
W_xh: 93*k, where k equals to the number of neurons

H: 100*k
W_hh: k*k

W_ho: k*93
output: calculated here: 100*93
        
'''




class RNN(nn.Module):
    def __init__(self, hidden_size,one_hot_length,computing_device):

        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.hidden = torch.zeros(1,self.hidden_size)
        self.hidden = self.hidden.to(computing_device)
        self.length = one_hot_length

        self.i2h = nn.Linear(self.length,hidden_size)
        self.h2h = nn.Linear(hidden_size,hidden_size)
        self.h2o = nn.Linear(hidden_size,self.length)


        torch_init.xavier_normal_(self.i2h.weight)
        torch_init.xavier_normal_(self.h2h.weight)
        torch_init.xavier_normal_(self.h2o.weight)



        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        output = torch.zeros(len(input),self.length)
        for i in range(len(input)):
            # apply weights to inputs and apply weights to hidden

            input_hidden = self.i2h(input[i])

            hidden_hidden = self.h2h(self.hidden)


            # W_hh(h_t-1) + W_xh(X_t)
            hidden = input_hidden + hidden_hidden

            # apply activation function
            self.hidden = func.tanh(hidden)

            # hidden to output
            output[i] = self.h2o(self.hidden)

        # output = torch.FloatTensor(output)
        self.hidden = self.hidden.detach()

        return output
