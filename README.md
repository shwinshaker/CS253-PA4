# CS253-PA4

### Dependencies
* Python3
* Pytorch
* numpy

### Manual
* To train a lstm model, simply type `python lstm.py`. It will encode the music data, build a model, and train and validate it. The hyperparameters are incorporated in the code, including number of epochs, chunk size, hidden layer size, optimizer type and learning rate. Other options includes resume, which resume the training from a checkpoint, and debug, which allows a few data through the model to test its feasibility.

* To test a model, type `python test.py`. It will read the setup parameters and best model from the checkpoint, and evaluate its performance on the test set.

* To generate music, type `python compose.py`. Given few initial notes, it will generate a required length of notes using the best model.

### List
---
`lstm.py`: preprocessing, build model, train and validate model

`compose.py`: generate music from best model

`test.py`: test the model on test set (loss)

`plot_loss.py`: plot the train and validation curve from recorded history.

`feature_extraction.ipynb`: plot the activation heatmap of a hidden neuron

---
`utils.py`: encoding utils for dataloader

`music_dataloader.py`: dataloader

![loss curve](loss.png)
![feature map](feature.png)