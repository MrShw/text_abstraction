import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.optim as optim
from seq2seq.model_elements.batcher import train_batch_generator
import time

def train_model(model, word_to_id, params):

    epochs = params['seq2seq_train_epochs']
    batch_size = params['batch_size']

    pad_index = word_to_id['<PAD>']
    unk_index = word_to_id['<UNK>']
    start_index = word_to_id['<START>']

    params['vocab_size'] = len(word_to_id)

    # mac用gpu会出现loss nan的情况
    device = torch.device('cpu' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    print(f'Model has passed to {device}...')

    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    def loss_function(pred, real):

        pad_mask = torch.eq(real, pad_index)
        unk_mask = torch.eq(real, unk_index)
        mask = torch.logical_not(torch.logical_or(pad_mask, unk_mask))
        pred = pred.transpose(2, 1)
        real = real * mask
        loss_ = criterion(pred, real)

        return torch.mean(loss_)

    def train_step(enc_input, dec_target):
        initial_hidden_state = model.encoder.initialize_hidden_state()
        initial_hidden_state = initial_hidden_state.to(device)
        optimizer.zero_grad()

        enc_output, enc_hidden = model.encoder(enc_input, initial_hidden_state)
        dec_input = torch.tensor([start_index] * batch_size)
        dec_input = dec_input.unsqueeze(1)

        dec_hidden = enc_hidden

        dec_input = dec_input.to(device)
        dec_hidden = dec_hidden.to(device)
        enc_output = enc_output.to(device)
        dec_target = dec_target.to(device)

        predictions, _ = model(dec_input, dec_hidden, enc_output, dec_target)
        loss = loss_function(predictions, dec_target)
        loss.backward()
        optimizer.step()

        return loss.item()

    dataset, steps_per_epoch = train_batch_generator(batch_size)

    for epoch in range(epochs):
        print("*" * 40)
        print(f"this is the {epoch+1} epoch, {epochs} in total.")
        start_time = time.time()
        total_loss = 0

        for batch, (inputs, targets) in enumerate(dataset):
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets = targets.type_as(inputs)
            batch_loss = train_step(inputs, targets)
            total_loss = total_loss + batch_loss

            if (batch + 1) % 20 == 0:
                print(f'Epoch {epoch + 1} Batch {batch + 1} Loss {batch_loss:.4f}')

        if (epoch + 1) % 2 == 0:
            MODEL_PATH = root_path + '/src/saved_model/' + 'model_' + str(epoch) + '.pt'
            torch.save(model.state_dict(), MODEL_PATH)
            print('*****************************************')
            print(f'** The model has saved for epoch {epoch + 1} **')
            print(f'** Epoch {epoch + 1} Total Loss {total_loss:.4f} **')
            print('*****************************************')

        print(f'Time taken for 1 epoch {time.time() - start_time} sec\n')
