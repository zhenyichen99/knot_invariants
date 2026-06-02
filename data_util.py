import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def transfer_data(from_path, to_path, col_names, index_col=None):
	# transfer data in the col_name column from the csv file from_path to to_path

	from_df = pd.read_csv(from_path, index_col=index_col)
	to_df = pd.read_csv(to_path, index_col=index_col)

	for col_name in col_names:
		to_df[col_name] = from_df[col_name]
		print(f'{col_name} copied to dataframe...', flush=True)
	
	to_df.to_csv(to_path, index=(index_col!=None))
	print('transfer complete')

def extract_cols(from_path, to_path, col_names):
	# extract the columns in the list col_names and store in to_path

	from_df = pd.read_csv(from_path)
	to_df = from_df[col_names]
	print('columns extracted...', flush=True)

	to_df.to_csv(to_path, index=False)
	print('data saved', flush=True)

def change_col_type(path, col_name, dtype):
	df = pd.read_csv(path)
	df = df.astype({col_name: dtype})
	df.to_csv(path, index=False)
	print(f'column {col_name} data type changed to {dtype}')

def info(path):
	df = pd.read_csv(path)
	print(df.info(verbose=True, show_counts=True))

def count_missing_values(path):
	df = pd.read_csv(path)
	print(df.isnull().sum())

def drop_nan(path):
	'''
	path: a csv file
	drop all rows with missing values
	'''
	df = pd.read_csv(path)
	df.dropna(inplace=True)
	df.to_csv(path, index=False)
	print('deleted all rows with missing values')

def kill_tiny(path, col_name, threshold=1e-12):
	'''
	path: a csv file
	set all values below threshold in the column col_name to zero
	'''
	df = pd.read_csv(path)
	df_num = pd.read_csv(path).select_dtypes(include='number')
	print('csv read into dataframe...', flush=True)

	df_num.where(df_num > threshold, 0, inplace=True)
	for col in df_num.columns:
		df[col] = df_num[col]
	print(f'values below {threshold} in the {col_name} column set to zero...', flush=True)

	df.to_csv(path, index=False)
	print('new data saved')

def find_range(path, col_name):
	df = pd.read_csv(path)
	col_min = df[col_name].min()
	col_max = df[col_name].max()
	print(f'min {col_name} = {col_min}\nmax {col_name} = {col_max}')

def shift_by_min(path, col_name):
	'''
	shift the values in the column col_name so that min=0
	'''
	df = pd.read_csv(path)
	col_min = df[col_name].min()
	df[col_name] -= col_min
	print('data shifted by minimum...', flush=True)
	df.to_csv(path, index=False)
	print('shifted data saved', flush=True)

def count_values(path, col_name):
	df = pd.read_csv(path)
	print(df[col_name].value_counts())

def normalize(path):
	'''
	path: a csv file
	for each column, subtract the mean and divide by the variance,
	then write into a new csv file ending in '_nomalized.csv'
	'''
	assert path[-4:] == '.csv', "'path' should end with '.csv'"
	df = pd.read_csv(path)
	df_num = pd.read_csv(path).select_dtypes(include='number')
	print('csv read into dataframe...', flush=True)

	scaler = StandardScaler()
	scaled_data = scaler.fit_transform(df_num)
	df_num = pd.DataFrame(scaled_data, columns=df_num.columns)

	for col in df_num.columns:
		df[col] = df_num[col]
	print('data normalized...', flush=True)

	df.to_csv(f'{path[:-4]}_normalized.csv', index=False)
	print('normalized data saved')

def split_data(path, p_train=90, p_val=5, p_test=5):
	'''
	path: a csv file
	p_#: integer percentage of # set in the train/val/test split
	split every 100 rows according to the given ratio and then
	write them into three csv files
	'''

	assert path[-4:] == '.csv', "'path' should end with '.csv'"
	assert p_train + p_val + p_test == 100, 'percentages should add up to 100'

	df = pd.read_csv(path)
	l = len(df)
	n_val = int(l*p_val/100)
	n_test = int(l*p_test/100)
	n_train = len(df) - n_val - n_test

	df_val = df.iloc[:n_val]
	df_test = df.iloc[n_val:n_val+n_test]
	df_train = df.iloc[n_val+n_test:]
	print('dataframes created...', flush=True)
	# for i in range(l):
	# 	r = i % 100
	# 	if r < p_train:
	# 		df_train.loc[len(df_train)] = df.loc[i]
	# 	elif r < p_train + p_val:
	# 		df_val.loc[len(df_val)] = df.loc[i]
	# 	else:
	# 		df_test.loc[len(df_test)] = df.loc[i]

	prepath = path[:-4]
	df_train.to_csv(f'{prepath}_train.csv', index=False)
	print('train data saved...', flush=True)
	df_val.to_csv(f'{prepath}_val.csv', index=False)
	print('val data saved...', flush=True)
	df_test.to_csv(f'{prepath}_test.csv', index=False)
	print('test data saved...\nsplit complete', flush=True)

def random_batch(df, batch_size, target_col, device='cpu', fix_seed=False):
	'''
	df: pd.DataFrame
	batch_size: int
	return a pytorch tensor of shape (batch_size, len(df.columns))
	'''

	if fix_seed:
		np.random.seed(0)
	indices = np.random.randint(0, len(df), size=batch_size)
	rows = df.iloc[indices]

	inputs = rows[[col for col in df.columns if col != target_col]]
	targets = rows[target_col]

	inputs = torch.tensor(inputs.to_numpy(), dtype=torch.float, device=device)
	targets = torch.tensor(targets.to_numpy(), dtype=torch.int, device=device)

	return inputs, targets

def random_batches(n_batches, data, batch_size, target_col, device='cpu', fix_seed=False):
	# generate a length = n_batches list of random batches 

	if fix_seed:
		np.random.seed(0)

	batches = []
	for _ in range(n_batches):
		batches.append(random_batch(data, batch_size, target_col=target_col, device=device))

	return batches

