import torch
from torch import nn
import datetime
import matplotlib.pyplot as plt
from data_util import random_batches

def train(batches, model, optimizer):

    model.train()
    loss_fn = nn.CrossEntropyLoss()
    n_batches = len(batches)
    train_loss = 0

    for x, y in batches:

        # propagate forward
        logits = model(x)
        loss = loss_fn(logits, y)
        train_loss += loss.item()

        # backpropagate
        loss.backward()
        # for param in model.parameters():
        #     print(param.grad)
        optimizer.step()
        optimizer.zero_grad()

    # return avg train loss per sample
    train_loss /= n_batches
    return train_loss


def test(batches, model):

    model.eval() 
    loss_fn = nn.CrossEntropyLoss() 
    n_batches = len(batches) 
    n_samples = n_batches * batches[0][0].size(0)
    test_loss, correct = 0, 0

    # testing
    with torch.no_grad():
        for x, y in batches:
            logits = model(x)
            test_loss += loss_fn(logits, y).item()
            correct += (logits.argmax(1) == y).type(torch.float).sum().item()

    # return avg test loss per sample and accuracy
    test_loss /= n_batches
    correct /= n_samples
    return test_loss, correct*100

def random_epochs(train_data, val_data, model, optimizer, 
    n_epochs, n_train_bats, n_val_bats, batch_size, target_col, device, fix_seed=False):

    if fix_seed:
        np.random.seed(0)

    print(f'epoch | train_loss | val_loss | accuracy(%)', flush=True)

    for e in range(n_epochs):

        print(f'{e+1:<6}|', end='', flush=True)

        train_batches = random_batches(n_train_bats, train_data, batch_size, target_col=target_col, device=device)
        train_loss = train(train_batches, model, optimizer)
        print(f' {train_loss:>9.6f}  |', end='', flush=True)

        val_batches = random_batches(n_val_bats, val_data, batch_size, target_col=target_col, device=device)
        val_loss, accuracy = test(val_batches, model)
        print(f'{val_loss:>9.6f} |{accuracy:>9.2f}', flush=True)

    timestamp = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')[:19]
    save_as = 'trained_models/knots_' + timestamp
    torch.save(model.state_dict(), save_as)
    print('final model saved', flush=True)


        





