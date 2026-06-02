import torch
import pandas as pd
from model import mlp, mlp_dropout
from data_util import random_batches
from train_test import random_epochs, train

# hyperparameters
learning_rate = 3e-4
batch_size = 32
weight_decay = 1e-4

# pick accelerator
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'

# load the data
train_data = pd.read_csv('invariants/alexander_exp_3/3-16_hyp_alexander_exp_3_train.csv').select_dtypes('number')
val_data = pd.read_csv('invariants/alexander_exp_3/3-16_hyp_alexander_exp_3_val.csv').select_dtypes('number')
print('data loaded...')

# instantiate model
model = mlp_dropout(in_width=3, out_width=50, hidden_width=400, depth=3, dropout=0.2).to(device)
print('model created...') 

# loss function and optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

random_epochs(train_data, val_data, model, optimizer, 
	n_epochs=100, n_train_bats=1000, n_val_bats=50, 
	batch_size=batch_size, target_col='alexander_const', device=device)

# overfitting a few examples
# batches = random_batches(2, train_data, batch_size=2, device=device, target_col='alexander_const', fix_seed=True)
# print(batches)
# for _ in range(100):
#     print(train(batches, model, optimizer))
