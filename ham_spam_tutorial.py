import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torchtext
from torchtext.legacy.data import Field, BucketIterator, TabularDataset # Later versions must use '.legacy'
import torch.optim as optim
from datetime import datetime

import rnn # custom neural network class

def load_and_clean(random_state=123): 
    df = pd.read_csv('./spam.csv', encoding='latin-1')

    # Debug Quick look to ensure file is correct
    print(df.head())

    # Dropping not required columns
    df = df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis= 1)

    # Rename
    df = df.rename(index = str, columns = {'v1': 'labels', 'v2': 'text'})

    print(df.head())

    #* Split dataset into training and test datasets 
    train_df, test_df = train_test_split(df, test_size = 0.2, random_state = random_state)
    train_df.reset_index(drop=True)
    test_df.reset_index(drop=True)

    #debug
    print("[DEBUG] train_df.shape:",train_df.shape)
    print("[DEBUG] test_df.shape:",test_df.shape)

    #* Flush resultant datasets to file
    train_df.to_csv('./spam_train.csv', index=False)
    test_df.to_csv('./spam_test.csv', index=False)

#* Tokenization
#nltk.download('punkt')

def vocab_setup():
    # Torchtext datasets
    TEXT = torchtext.legacy.data.Field(tokenize = word_tokenize) # or you could just use Field(...) if you imported earlier
    LABEL = torchtext.legacy.data.LabelField(dtype = torch.float)
    datafields = [("labels", LABEL), ("text", TEXT)]

    tor_train, tor_test = torchtext.legacy.data.TabularDataset.splits(
        path = '.\\data',
        train = 'spam_train.csv',
        test = 'spam_test.csv',
        format = 'csv',
        skip_header = True,
        fields = datafields
    )

    #debug
    print("[DEBUG] no. of training examples:",len(tor_train))
    print("[DEBUG] no. of testing examples:", len(tor_test))

    #* Building Vocab
    TEXT.build_vocab(tor_train, max_size=10500) # 10500 is arbitrary? Top 10500, the rest will be unknowns (UNK).
    LABEL.build_vocab(tor_train)

    return TEXT, LABEL, tor_train, tor_test

def load_datasets_and_train(TEXT, LABEL, tor_train, tor_test, num_epochs=5):
    """ #debug
    print("[DEBUG] Unique tokens in TEXT:", len(TEXT.vocab))
    print("[DEBUG] Unique tokens in LABEL:", len(LABEL.vocab))
    print("[DEBUG] TEXT most common words:", TEXT.vocab.freqs.most_common(50))

    #debug - integer to string mapping or REVERSE
    print("[DEBUG] TEXT.vocab.itos[:10]:",TEXT.vocab.itos[:10])
    print("[DEBUG] TEXT.vocab.stoi:", TEXT.vocab.stoi) """

    #* Iterator
    batch_size = 64
    train_iterator, test_iterator = torchtext.legacy.data.BucketIterator.splits(
        (tor_train, tor_test),
        batch_size = batch_size,
        sort_key = lambda x: len(x.text),
        sort_within_batch = False
    )

    #* Instantiate RNN 
    input_dim = len(TEXT.vocab)
    embedding_dim = 100
    hidden_dim = 256
    output_dim = 1 # 0 = Ham; 1 = Spam
    nn_type = "lstm" # or rnn for normal

    model = rnn.RNN(input_dim, embedding_dim, hidden_dim, output_dim, nn_type)
    optimizer = optim.Adam(model.parameters(), lr=1e-6) # Adam optimizer with learning rate 1 x 10^-6
    criterion = torch.nn.BCEWithLogitsLoss() # Loss-function - Cross-entropy calculation for binary classification with sigmoid function for predictions

    #* Training
    print("[+] Starting training for {NUM_EPOCH} epoch(s)...".format(NUM_EPOCH=num_epochs))
    for epoch in range(num_epochs):
        train_loss, train_acc = rnn.RNN.train_once(model, train_iterator, optimizer, criterion) # default rnn (not LSTM)

        print('[*] Epoch: {EPOCH} -> Train Loss = {TRAIN_LOSS} / Train Acc: {TRAIN_ACC}%'.format(
            EPOCH=epoch+1, TRAIN_LOSS=round(train_loss, 3), TRAIN_ACC=round(train_acc*100 ,2) 
        ))
    
    #* Evalulation
    print(model.eval())

    test_loss, test_acc = rnn.RNN.evaluate(model, test_iterator, criterion)
    print("[~] Evaluation of current model -> Test Loss: {TEST_LOSS} / Test Acc: {TEST_ACC}%".format(
        TEST_LOSS=test_loss, TEST_ACC=round(test_acc*100,2)
    ))

    return model, test_acc

def run():
    #* Basics Set up
    TEXT, LABEL, tor_train, tor_test = vocab_setup()

    """ #*0. Load and clean
    load_and_clean(random_state=42)
    #TODO: find out difference between random_state=123 vs. 42.

    #* 1. Train and save
    num_epochs = 5
    trained_model, test_acc = load_datasets_and_train(TEXT, LABEL, tor_train, tor_test, num_epochs=num_epochs)
    torch.save(trained_model, "./trained/trained_model_{DATE_TIME}_{ACCURACY}percent.pt".format(
        DATE_TIME=datetime.now().strftime("%m-%d-%Y_%H%M"), ACCURACY=round(test_acc*100)
        )) """

    
    #* 2. Load and Predict
    loaded_model = torch.load("./trained/trained_model_07-22-2021_2307_71percent.pt")

    # Test text
    test_text_1 = "I don't wanna talk about this crap anymore. Let's drop it, shall we?"
    print("[#] Predicting:", test_text_1)
    score = rnn.RNN.predict(loaded_model, TEXT, test_text_1)

    # Decide if ham or spam
    if score < 0.5:
        print("[/] It was determined to be a legit message. (Score: {SCORE})".format(SCORE=score))
    else:
        print("[!] It's f##king spam! (Score: {SCORE})".format(SCORE=score))


if __name__ == "__main__":
    run()