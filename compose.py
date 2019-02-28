from music_dataloader import load_input_label
from lstm import build_model, Evaluation
from utils import String_Encoder
import torch
import numpy as np
from torch.nn import functional as F

def generate_music(model, encoder, length=100):
    
    # evaluation mode
    model.eval()
    
    # init_hidden
    model.hidden = model._init_hidden()
    
    # init note
    init_notes = load_input_label('pa4Data/test.txt')[0][:110] 
    #'<start>\nX:19\nT:Dusty Miller, The'
    init_seq = []
    for w in init_notes:
        init_seq.append(encoder.get_one_hot(w))
        
    init_seq = torch.tensor(init_seq, dtype=torch.float)
    init_seq = init_seq.view(1,len(init_seq),-1)
    # print(init_seq)
    
    def _get_indices(output, temperature=1):
        # print(output.squeeze().shape)
        # F.softmax(output, dim=2).detach().numpy().argsort(axis=2)
        dim = output.shape[1]
        opt_soft = F.softmax(output / temperature, 
                             dim=1).detach().numpy()
        inds = []
        for opt in opt_soft:
            assert(opt.shape==(93,))
            inds.append(np.random.choice(dim, 1, p=opt).squeeze())
        return inds
    
    def _to_input(output):
        characters = []
        inputs = []
        for ind in _get_indices(output):
            character = encoder.get_character(ind)
            characters.append(character)
            inputs.append(encoder.get_one_hot(character))
        inputs = torch.tensor(inputs, dtype=torch.float)
        inputs = inputs.view(1, len(characters), -1)
        assert(inputs.shape[-1] == 93)
        return characters, inputs
    
    notes = []
    with torch.no_grad():
        outputs = model(init_seq)
        characters, inputs = _to_input(outputs)
        notes.extend(characters)
        # pick the last one
        input = inputs[:, -1, :].view(1, 1, -1)
        for _ in range(length):
            output = model(input)
            character, input = _to_input(output)
            notes.extend(character)
    
    return ''.join(notes)


def main():
    print('----> loading setup')
    init = torch.load('init.pth.tar')
    encoder = init['encoder']
    model, _, _ = build_model(input_dim=encoder.length)

    print('---> loading best model')
    path = 'model_best.pth.tar'
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])

    with open("music.txt", "w") as f:
        f.write(generate_music(model, encoder, length=200))


if __name__ == "__main__":
    main()







