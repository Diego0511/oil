import pandas as pd
import os
import json
import random
from tqdm import tqdm
import numpy as np
import math

class Dataset:
	
	def __init__(self, path, blank_fill = 'avg', normailze = 'none', logwise = False, window_size = 7, use_KG = True, use_log = True, test_last = False):
		
		if blank_fill == 'avg':
			self.fill_blank = self.fill_blank_by_avg
		elif blank_fill == 'zero':
			self.fill_blank = self.fill_blank_by_zero
		elif blank_fill == 'smooth':
			self.fill_blank = self.fill_blank_by_neighbor

		if normailze == 'none':
			self.normailze = self.none_normalize
		elif normailze == 'maxmin':
			self.normailze = self.maxmin_normalize
		elif normailze == 'meanstd':
			self.normailze = self.meanstd_normalize
		self.path = path

		self.logwise = logwise

		self.window_size = window_size

		self.use_KG = use_KG
		self.use_log = use_log
		self.test_last = test_last
		
		self.train_x = None
		self.train_y = None
		self.valid_x = None
		self.valid_y = None
		self.train_scale = None
		self.valid_scale = None
		self.X = None
        self.Y = None
        self.S = None
		self.columns = None
		self.read_data()

	def get_colnames(self):
		return self.columns

	def read_data(self):
		
		files = os.listdir(self.path)
		files = [file for file in files if file[-4:] == '.csv']
		dfs = [pd.read_csv(file) for file in files]
		
		#product_embeddings = json.loads(open("oil_embs.json").read())
		type_embeddings = {"12":[0,0,0,0,0,0,0,1],"14":[0,0,0,0,0,0,1,0],"15":[0,0,0,0,0,1,0,0],
         "16":[0,0,0,0,1,0,0,0], "51":[0,0,0,1,0,0,0,0], "52":[0,0,1,0,0,0,0,0], "54":[0,1,0,0,0,0,0,0], "56":[1,0,0,0,0,0,0,0]}

		X, Y, S = [], [], []

		valid_ids = []

		for file_id in tqdm(range(len(files))):

			#split the file name according to '_' and take first part as type
			type = files[file_id].split("_")[0]
			#split the file name according to '_' and take second part as product
			#product = files[file_id].split("_")[1]
			
 
			#product, station = tuple(files[file_id][:-4].split("_"))
			#pemb, semb = product_embeddings[product], type_embeddings[type]
			semb = type_embeddings[type]
			df = dfs[file_id]
			self.columns = list(df.columns)
			self.columns = self.columns[1:-1]
			data = df[[col for col in df.columns if col != 'date']]
			ys = [self.float(data.iloc[i,3]) for i in range(data.shape[0])]
			
			
			if self.test_last == False: 
				idxes = [i for i in range(len(ys)) if ys[i] != None]
				random.shuffle(idxes)
				L = int(len(idxes)*0.8)+1
				valid_idxes = set(idxes[L:])
				train_idxes = idxes[:L]
				tmp_arr = [ys[idx] for idx in train_idxes]
				avg = sum(tmp_arr)/len(tmp_arr)
				scale = avg
				ys = [v/avg if v != None else None for v in ys]

			else: 
				tmp_arr = [v for v in ys if v != None]
				test_size = int(len(tmp_arr)*(1 - 0.8))  
				k, count = len(ys), 0
				while count < test_size:
					k -= 1
					if ys[k] != None:
						count += 1
				train_bar = k
				tmp_arr = [v for v in ys[:train_bar] if v != None]
				avg = sum(tmp_arr)/len(tmp_arr)
				ys = [v/avg if v != None else None for v in ys]
				scale = avg
				valid_idxes = {j for j in range(train_bar,len(ys)) if ys[j] != None}


			if self.use_log:
				ys = [math.log(v*1000+1) if v != None else None for v in ys]

			previous = {"confirmedcount":0, "suspectedcount":0, "curedcount":0, "deadcount":0}
			col2idx = {"confirmedcount":6, "suspectedcount":7, "curedcount":8, "deadcount":9}
			for i in range(data.shape[0]):
				for col in ["confirmedcount","suspectedcount","curedcount","deadcount"]:
					v = data[col][i]
					int_v = self.int(v)
					if int_v == None:
						int_v = previous[col]
					new = int_v - previous[col]
					previous[col] = int_v
					data.iloc[i, col2idx[col]] = new

			self.fill_blank(data, [3,4,5] + list(range(10,21)))
			
			# fill weather
			for i in range(21,38):
				for j in range(data.shape[0]):
					int_value = self.int(data.iloc[j,i])
					if int_value == None:
						data.iloc[j,i] = 0
					else:
						data.iloc[j,i] = int_value

			records = []
			for i in range(data.shape[0]):
				record = data.iloc[i,:].tolist()
				int1, int2 = self.int(record[1]), self.int(record[2])
				if int1 == None:
					int1 = random.randint(1,7)
				if int2 == None:
					int2 = 0 if int1 in {1,2,3,4,5} else 1
				record = record[3:] + self.onehot(int1-1,7) + self.onehot(int2,4)
				record[0] = self.float(record[0])
				records.append(record)

			assert len(records) == len(ys)
			for i in range(self.window_size, len(records)):
				if ys[i] != None: 
					x = []
					for j in range(i - self.window_size, i):
						x = x + records[j]
					
					x = x + semb if self.use_KG else x
					X.append(x)
					Y.append(ys[i])
					S.append(scale)
					if i in valid_idxes:
						valid_ids.append(len(Y) - 1)

		X = np.array(X)
		Y = np.array(Y)
		S = np.array(S)
		print(X.shape, Y.shape, S.shape)

		valid_idxes = np.array(valid_ids)
		valid_ids = set(valid_ids)
		train_idxes = np.array([i for i in range(X.shape[0]) if i not in valid_ids])

		self.train_x = X[train_idxes,:] 	
		self.train_y = Y[train_idxes] 		
		self.valid_x = X[valid_idxes,:]		
		self.valid_y = Y[valid_idxes]		
		self.train_scale = S[train_idxes]	
		self.valid_scale = S[valid_idxes]	
		self.X = X
        self.Y = Y
        self.S = S
		print(self.train_x.shape, self.valid_x.shape)


	def onehot(self, x, l):
		tmp = [0]*l
		return tmp

	def float(self, v):
		try:
			return float(v)
		except:
			return None

	def int(self, v):
		try:
			return int(v)
		except:
			return None

	def fill_blank_by_avg(self, data, fill_columns):
		for i in fill_columns: #receipts, oil, temperature
			tmp = [self.float(data.iloc[j,i]) for j in range(data.iloc[:,i].shape[0])]
			tmp = [value for value in tmp if value != None]
			avg = sum(tmp)/len(tmp)
			std = (sum([(x - avg)**2 for x in tmp])/len(tmp))**0.5
			for j in range(data.shape[0]):
				float_value = self.float(data.iloc[j,i])
				if float_value == None:
					data.iloc[j,i] = (random.random() - 1)*1e-3*std + avg
				else:
					data.iloc[j,i] = float_value

	def fill_blank_by_zero(self, data, fill_columns):
		for i in fill_columns:
			print(i)
			tmp = [self.float(data.iloc[j,i]) for j in range(data.iloc[:,i].shape[0])]
			tmp = [value for value in tmp if value != None]
			avg = sum(tmp)/len(tmp)
			std = (sum([(x - avg)**2 for x in tmp])/len(tmp))**0.5
			for j in range(data.shape[0]):
				float_value = self.float(data.iloc[j,i])
				if float_value == None:
					data.iloc[j,i] = (random.random() - 1)*1e-3*std + 0.0
				else:
					data.iloc[j,i] = float_value

	def fill_blank_by_neighbor(self, data, fill_columns):
		raise NotImplementedError

	def none_normalize(self):
		raise NotImplementedError

	def maxmin_normalize(self):
		raise NotImplementedError

	def meanstd_normalize(self):
		raise NotImplementedError


if __name__ == "__main__":
	dataset = Dataset(blank_fill = 'avg',path='./')