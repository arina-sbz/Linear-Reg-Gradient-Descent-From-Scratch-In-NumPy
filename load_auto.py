import numpy as np
import pandas as pd

def load_auto(selectAll = True):
	"""
	Load the Auto dataset and preprocess it.

	parameters:
    selectAll: If True, all feature columns except 'name' are used. If False, only the 'horsepower' column is used.

	returns: 
	X_train: the features of the train data
	X_train_normalized: the normalized version of the train data
	Y_train: the taget column of the train data
	"""
	
	# import data
	Auto = pd.read_csv('Auto.csv', na_values='?', dtype={'ID': str}).dropna().reset_index()
	columns = ['cylinders','displacement','horsepower','weight', 'acceleration','year','origin']
	normalized_data = Auto.copy()
	# normalize values and save them in normalized_data
	normalized_data[columns] = normalized_data[columns].apply(lambda x: (x - x.mean()) / x.std())

	# Extract relevant data features
	chosen_features = columns if selectAll else ['horsepower']
	X_train = Auto[chosen_features].values
	X_train_normalized = normalized_data[chosen_features].values
	
	Y_train = Auto[['mpg']].values

	return X_train, X_train_normalized, Y_train
