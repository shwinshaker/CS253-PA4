from lstm import validate_model, build_model, Evaluation, check_cuda
from utils import String_Encoder
import torch


def main():
    print('---> check cuda')
    use_cuda, device, extras = check_cuda()

    print('----> loading setup')
    init = torch.load('init0.pth.tar')
    encoder = init['encoder']
    dataloaders = init['loaders']
    hidden_size = init['hidden_size']
    model, criterion, _ = build_model(input_dim=encoder.length, hidden_dim=hidden_size, 
                                      device=device)

    print('---> loading best model')
    path = 'model_best0.pth.tar'
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])

    print('test loss: %.8f'  % validate_model(model, criterion, dataloaders['test'],
                                              verbose=True, device=device))


if __name__ == "__main__":
    main()
