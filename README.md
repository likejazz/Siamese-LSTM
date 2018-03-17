# Siamese-LSTM

Using MaLSTM model(Siamese networks + LSTM with Manhattan distance) to detect semantic similarity between question pairs. Training dataset used is a subset of the original Quora Question Pairs Dataset(~363K pairs used).

It is Keras implementation based on [Original Paper(PDF)](http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf) and [Excellent Medium Article](https://medium.com/mlreview/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07).

![](https://cloud.githubusercontent.com/assets/9861437/20479493/6ea8ad12-b004-11e6-89e4-53d4d354d32e.png)

## Prerequisite

- Paper, Articles
    - [Siamese Recurrent Architectures for Learning Sentence Similarity](http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf)
    - [How to predict Quora Question Pairs using Siamese Manhattan LSTM](https://medium.com/mlreview/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07)
- Data
    - [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
    - [Kaggle's Quora Question Pairs Dataset](https://www.kaggle.com/c/quora-question-pairs/data)
- References
    - [aditya1503/Siamese-LSTM](https://github.com/aditya1503/Siamese-LSTM) Original author's GitHub
    - [dhwajraj/deep-siamese-text-similarity](https://github.com/dhwajraj/deep-siamese-text-similarity) TensorFlow based implementation

Kaggle's `test.csv` is too big, so I had extracted only the top 20 questions and created a file called `test-20.csv` and It is used in the `predict.py`.

You should put all data files to `./data` directory.

## How to Run
### Training
```
$ python3 train.py
```

### Predicting
It uses `test-20.csv` file mentioned above.
```
$ python3 predict.py
```

## The Results
I have tried with various parameters such as number of hidden states of LSTM cell, activation function of LSTM cell and repeated count of epochs. As a result, I have reached about 82.25% accuracy after 150 epochs 6h 22m later. 

```
363861/363861 [==============================] - 162s 444us/step - loss: 0.0857 - acc: 0.8993 - val_loss: 0.1345 - val_acc: 0.8225
Training time finished.
150 epochs in     22965.92
```

Training time(NVIDIA Tesla P40 GPU) per 1 epoch(max seq length=20) is *2min 23secs*, 10% data was used as the validation set.
