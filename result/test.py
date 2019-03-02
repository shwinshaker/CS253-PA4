from lstm import validate_model, build_model, Evaluation
from utils import String_Encoder
import torch
'''
            learning_rate   hiden_size      chunk_size      test_loss     
model 1:    0.1             75              100             2.30                            
model 2:    0.1             100             100             2.19
model 3:    0.1             150             100             2.11
model 4:    0.01            100             100             2.86
model 5:    1.0             100             100             2.40
model 6:    0.1             100             75              2.27
model 7:    0.1             100             150             2.00
model 8:    0.1             150             150             2.19
model 9:    0.1             150             200             2.01
model 10:   0.1             200             150             2.14
model 11:   0.1             200             200             1.89
'''

def main():
    for i in range(1,11):
        print('--------------------model num:',i,'--------------------')
        print('----> loadin setup')
        init = torch.load('init'+str(i)+'.pth.tar')
        encoder = init['encoder']
        dataloaders = init['loaders']
        hidden_size = init['hidden_size']
        model, criterion, _ = build_model(input_dim=encoder.length, hidden_dim=hidden_size)

        print('---> loading best model')
        path = 'model_best'+str(i)+'.pth.tar'
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])

        print('test loss: %.8f'  % validate_model(model, criterion, dataloaders['test'],
                                                  verbose=True))


if __name__ == "__main__":
    main()
