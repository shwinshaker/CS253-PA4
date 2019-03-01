from lstm import validate_model, build_model, Evaluation
from utils import String_Encoder
import torch


def main():
    print('----> loadin setup')
    init = torch.load('init.pth.tar')
    encoder = init['encoder']
    dataloaders = init['loaders']
    hidden_size = init['hidden_size']
    model, criterion, _ = build_model(input_dim=encoder.length, hidden_dim=hidden_size)

    print('---> loading best model')
    path = 'model_best.pth.tar'
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])

    print('test loss: %.8f'  % validate_model(model, criterion, dataloaders['test'],
                                              verbose=True))


if __name__ == "__main__":
    main()
