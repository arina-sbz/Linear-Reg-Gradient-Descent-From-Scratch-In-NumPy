import numpy as np
import pandas as pd

def load_auto(selectAll = True):

	# import data
	Auto = pd.read_csv('Auto.csv', na_values='?', dtype={'ID': str}).dropna().reset_index()
	columns = ['cylinders','displacement','horsepower','weight', 'acceleration','year','origin']
	normalized_data = Auto.copy()
	# normalize values and save them in normalized_data
	normalized_data[columns] = normalized_data[columns].apply(lambda x: (x - x.mean()) / x.std())

	# Extract relevant data features
	if selectAll:
		X_train = Auto[['cylinders','displacement','horsepower','weight', 'acceleration','year','origin']].values
		X_train_normalized = normalized_data[['cylinders','displacement','horsepower','weight', 'acceleration','year','origin']].values
	else:
		X_train = Auto[['horsepower']].values
		X_train_normalized = normalized_data[['horsepower']].values
	
	Y_train = Auto[['mpg']].values

	return X_train, X_train_normalized, Y_train
