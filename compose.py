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
    
    # init note -> try anything, but start with <start>
    init_notes = load_input_label('pa4Data/test.txt')[0][:110] 
    #'<start>\nX:19\nT:Dusty Miller, The'

    # convert initial notes to tensor
    init_seq = []
    for w in init_notes:
        init_seq.append(encoder.get_one_hot(w))
        
    init_seq = torch.tensor(init_seq, dtype=torch.float)
    init_seq = init_seq.view(1,len(init_seq),-1)
    
    def _get_indices(output, temperature=1):
        # temperature based sampling
        # high temperature means low deterministic
        # pick indices on the output probability by softmax
        dim = output.shape[1]
        opt_soft = F.softmax(output/temperature, dim=1).detach().numpy()
        inds = []
        probs = []
        for opt in opt_soft:
            assert(opt.shape==(93,))
            ind = np.random.choice(dim, 1, p=opt).squeeze()
            inds.append(ind)
            probs.append(opt[ind])
        return inds, probs
    
    def _to_input(output):
        # convert a lstm output to input
        # to feed back to the net
        characters = []
        inputs = []
        inds, probs = _get_indices(output)
        for ind in inds:
            character = encoder.get_character(ind)
            inputs.append(encoder.get_one_hot(character))
            characters.append(character)
        inputs = torch.tensor(inputs, dtype=torch.float)
        inputs = inputs.view(1, len(characters), -1)
        assert(inputs.shape[-1] == 93)
        return characters, inputs, probs
    
    notes = []
    confs = [1.]
    notes.extend(list(init_notes))
    confs *= len(init_notes)
    with torch.no_grad():
        outputs = model(init_seq)
        characters, inputs, probs = _to_input(outputs)
        # record the last output <- predicted
        notes.append(characters[-1])
        confs.append(probs[-1])
        # pick the last output as next input
        input = inputs[:, -1, :].view(1, 1, -1)
        for _ in range(length):
            # loop production
            # output -> input -> output -> ..
            output = model(input)
            character, input, prob = _to_input(output)
            notes.extend(character)
            confs.extend(prob)
    
    return ''.join(notes), ' '.join(['%.2f' % f for f in confs])


def main():
    print('----> loading setup')
    init = torch.load('init.pth.tar')
    encoder = init['encoder']
    model, _, _ = build_model(input_dim=encoder.length)

    print('---> loading best model')
    path = 'model_best.pth.tar'
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])

    print('---> Music sheet generated to music.txt')
    notes, confs = generate_music(model, encoder, length=200)
    notes_s = [s.replace('\n', '\\n') for s in list(notes)]
    notes_r = ' '.join([n+':'+c for n, c in zip(notes_s, confs.split())])
    print(notes)
    with open("music.txt", "w") as f:
        f.write(notes)
        f.write('\n------------------\n')
        f.write(notes_r)


if __name__ == "__main__":
    main()







