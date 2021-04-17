import numpy as np
import pandas as pd
import keras
from PIL import Image
from skimage.transform import resize, rotate
from sklearn.model_selection import train_test_split
from skimage.filters import gaussian
import itertools
from tqdm import tqdm
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, Subtract, Lambda
from keras import backend as K


def augment(X, y):
	# add blur
	# with 25% chance
	if np.random.randint(0, 4, 1)[0] == 0:
		# make blur if clear and change GT
		if y == 0:
			X = gaussian(X, sigma=np.random.randint(5, 8, 1)[0], preserve_range=True, multichannel=True)
			y = 1


	# make black and white
	# with 33% chance
	if np.random.randint(0, 3, 1)[0] == 0:
		# make black and white
		X = np.repeat(np.mean(X, axis=-1, keepdims=True), 3, axis=-1)


	# rotation
	X = rotate(X, angle=np.random.randint(-359, 360, 1)[0], preserve_range=True)
	return X, y


# normal classification dataset
class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, labels, batch_size=32, dim=(224,224), n_channels=3,
				 n_classes=2, shuffle=False, aug=False):
		'Initialization'
		self.dim = dim
		self.batch_size = batch_size
		self.labels = labels
		self.list_IDs = list_IDs
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.aug = aug
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.indexes) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Generate data
		X, y = self.__data_generation(indexes)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.tile(np.arange(len(self.list_IDs)), 10)
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, indexes):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
		y = np.empty((self.batch_size), dtype=int)

		# Generate data
		for i, indx in enumerate(indexes):
			# Store sample
			X[i,] = resize(np.array(Image.open('dataset/train/' + self.list_IDs[indx])), self.dim, preserve_range=True)

			# Store class
			y[i] = self.labels[indx]

		
		# data augmentation
		if self.aug == True:
			for i, indx in enumerate(indexes):
				# with 50% chance
				if np.random.randint(0, 2, 1)[0] == 0:
					# make blur if clear and change GT
					if y[i] == 0:
						X[i,] = gaussian(X[i,], sigma=np.random.randint(5, 8, 1)[0], preserve_range=True)
						y[i] = 1

				# with 33% chance
				if np.random.randint(0, 3, 1)[0] == 0:
					# make black and white
					X[i,] = np.repeat(np.mean(X[i,], axis=-1, keepdims=True), 3, axis=-1)

				# rotation
				X[i,] = rotate(X[i,], angle=np.random.randint(-359, 360, 1)[0], preserve_range=True)


		return X/255.0, keras.utils.to_categorical(y, num_classes=self.n_classes)


# Pair selection - all possible pairs
class DataGeneratorPair(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, labels, batch_size=32, dim=(224,224), n_channels=3,
				 n_classes=2, shuffle=False, aug=False):
		'Initialization'
		self.dim = dim
		self.batch_size = batch_size
		self.labels = labels
		self.list_IDs = list_IDs
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.aug = aug
		self.cache = {}

		for img in tqdm(self.list_IDs):
			self.cache['dataset/train/' + img] = resize(np.array(Image.open('dataset/train/' + img)), self.dim, preserve_range=True)

		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.indexes) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Generate data
		X, y = self.__data_generation(indexes)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		# self.indexes = np.tile(np.arange(len(self.list_IDs)), 10)
		self.indexes = list(itertools.combinations(range(len(self.list_IDs)), 2))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, indexes):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X1 = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
		X2 = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
		y1 = np.empty((self.batch_size), dtype=int)
		y2 = np.empty((self.batch_size), dtype=int)
		y = np.empty((self.batch_size), dtype=int)

		# Generate data
		for i, indx in enumerate(indexes):
			# Store sample
			X1[i,] = self.cache['dataset/train/' + self.list_IDs[indx[0]]]
			X2[i,] = self.cache['dataset/train/' + self.list_IDs[indx[1]]]
			
			# Store class
			y1[i] = self.labels[indx[0]]
			y2[i] = self.labels[indx[1]]

		
		# data augmentation
		if self.aug == True:
			for i, indx in enumerate(indexes):
				X1[i,], y1[i] = augment(X1[i,].copy(), y1[i].copy())
				X2[i,], y2[i] = augment(X2[i,].copy(), y2[i].copy())


		# generate the ground truth
		for i, indx in enumerate(indexes):
			y[i] = int(y1[i] == y2[i])


		return [X1/255.0, X2/255.0], keras.utils.to_categorical(y, num_classes=self.n_classes)


# Pair selection - pick up hard easy and hard negative samples
class DataGeneratorPairHard(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, labels, model, score_model, batch_size=32, dim=(224,224), n_channels=3,
				 n_classes=2, shuffle=False, aug=False):
		'Initialization'
		self.dim = dim
		self.batch_size = batch_size
		self.labels = labels
		self.list_IDs = list_IDs
		self.model = model
		self.score_model = score_model
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.aug = aug
		self.cache = {}

		for img in tqdm(self.list_IDs):
			self.cache['dataset/train/' + img] = resize(np.array(Image.open('dataset/train/' + img)), self.dim, preserve_range=True)

		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.indexes) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Generate data
		X, y = self.__data_generation(indexes)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		# self.indexes = np.tile(np.arange(len(self.list_IDs)), 10)
		indexes = list(itertools.combinations(range(len(self.list_IDs)), 2))
		dist = np.zeros((len(self.list_IDs), len(self.list_IDs)))
		self.newImages = self.labels.copy().tolist()
		self.newLabels = self.labels.copy()
		features = []

		
		for i in tqdm(range(len(self.list_IDs)), desc="Getting the features"):
			X1 = self.cache['dataset/train/' + self.list_IDs[i]]
			Y1 = self.labels[i]

			if self.aug == True:
				X1, Y1 = augment(X1.copy(), Y1.copy())

			self.newImages[i] = X1
			self.newLabels[i] = Y1
			f = self.model.predict(X1[np.newaxis, :, :, :]/255.0)
			features.append(f)


		for i in tqdm(indexes, desc="Calculating dist matrix"):
			X1 = features[i[0]]
			Y1 = self.newLabels[i[0]]
			X2 = features[i[1]]
			Y2 = self.newLabels[i[1]]

			d = self.score_model.predict([X1, X2])[0][0]
			dist[i[0], i[1]] = d
			dist[i[1], i[0]] = d

		# print(dist)

		samemask = np.zeros(dist.shape)
		for i in range(len(self.list_IDs)):
			for j in range(len(self.list_IDs)):
				if i == j:
					samemask[i][j] = 0
				else:
					samemask[i][j] = np.float32(self.newLabels[i] == self.newLabels[j])


		diffmask = np.zeros(dist.shape)
		for i in range(len(self.list_IDs)):
			for j in range(len(self.list_IDs)):
				if i == j:
					diffmask[i][j] = 0
				else:
					diffmask[i][j] = np.float32(self.newLabels[i] != self.newLabels[j])


		samedist = dist * samemask
		diffdist = (1.0-dist) * diffmask

		same = np.argmax(samedist, axis=0)
		diff = np.argmax(diffdist, axis=0)

		self.indexes = []
		for i in range(len(self.list_IDs)):
			self.indexes.append((i, same[i]))
			self.indexes.append((i, diff[i]))

		# print(self.indexes)
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, indexes):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X1 = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
		X2 = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
		y1 = np.empty((self.batch_size), dtype=int)
		y2 = np.empty((self.batch_size), dtype=int)
		y = np.empty((self.batch_size), dtype=int)

		# Generate data
		for i, indx in enumerate(indexes):
			# Store sample
			X1[i,] = self.newImages[indx[0]]
			X2[i,] = self.newImages[indx[1]]

			# Store class
			y1[i] = self.newLabels[indx[0]]
			y2[i] = self.newLabels[indx[1]]


		# generate the ground truth
		for i, indx in enumerate(indexes):
			y[i] = int(y1[i] == y2[i])


		return [X1/255.0, X2/255.0], keras.utils.to_categorical(y, num_classes=self.n_classes)


# Triplet selection - all possible triplet
class DataGeneratorTriplet(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, labels, batch_size=32, dim=(224,224), n_channels=3,
				 n_classes=2, shuffle=False, aug=False):
		'Initialization'
		self.dim = dim
		self.batch_size = batch_size
		self.labels = labels
		self.list_IDs = list_IDs
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.aug = aug
		self.cache = {}

		for img in tqdm(self.list_IDs):
			self.cache['dataset/train/' + img] = resize(np.array(Image.open('dataset/train/' + img)), self.dim, preserve_range=True)

		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.indexes) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Generate data
		X, y = self.__data_generation(indexes)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		# self.indexes = np.tile(np.arange(len(self.list_IDs)), 10)
		self.indexes = list(itertools.combinations(range(len(self.list_IDs)), 3))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, indexes):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X1 = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
		X2 = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
		X3 = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
		y1 = np.empty((self.batch_size), dtype=int)
		y2 = np.empty((self.batch_size), dtype=int)
		y3 = np.empty((self.batch_size), dtype=int)
		y = np.empty((self.batch_size), dtype=int)

		# Generate data
		for i, indx in enumerate(indexes):
			# Store sample
			X1[i,] = self.cache['dataset/train/' + self.list_IDs[indx[0]]]
			X2[i,] = self.cache['dataset/train/' + self.list_IDs[indx[1]]]
			X3[i,] = self.cache['dataset/train/' + self.list_IDs[indx[2]]]
			
			# Store class
			y1[i] = self.labels[indx[0]]
			y2[i] = self.labels[indx[1]]
			y3[i] = self.labels[indx[2]]

		
		# data augmentation
		if self.aug == True:
			for i, indx in enumerate(indexes):
				X1[i,], y1[i] = augment(X1[i,].copy(), y1[i].copy())
				X2[i,], y2[i] = augment(X2[i,].copy(), y2[i].copy())
				X3[i,], y3[i] = augment(X3[i,].copy(), y3[i].copy())


		return [X1/255.0, X2/255.0, X3/255.0], keras.utils.to_categorical(y, num_classes=self.n_classes)


# Triplet selection - pick up hard easy and hard negative samples
class DataGeneratorTripletHard(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, labels, model, score_model, batch_size=32, dim=(224,224), n_channels=3,
				 n_classes=2, shuffle=False, aug=False):
		'Initialization'
		self.dim = dim
		self.batch_size = batch_size
		self.labels = labels
		self.list_IDs = list_IDs
		self.model = model
		self.score_model = score_model
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.aug = aug
		self.cache = {}

		for img in tqdm(self.list_IDs):
			self.cache['dataset/train/' + img] = resize(np.array(Image.open('dataset/train/' + img)), self.dim, preserve_range=True)

		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.indexes) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Generate data
		X, y = self.__data_generation(indexes)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		# self.indexes = np.tile(np.arange(len(self.list_IDs)), 10)
		indexes = list(itertools.combinations(range(len(self.list_IDs)), 2))
		dist = np.zeros((len(self.list_IDs), len(self.list_IDs)))
		self.newImages = self.labels.copy().tolist()
		self.newLabels = self.labels.copy()
		features = []

		
		for i in tqdm(range(len(self.list_IDs)), desc="Getting the features"):
			X1 = self.cache['dataset/train/' + self.list_IDs[i]]
			Y1 = self.labels[i]

			if self.aug == True:
				X1, Y1 = augment(X1.copy(), Y1.copy())

			self.newImages[i] = X1
			self.newLabels[i] = Y1
			f = self.model.predict(X1[np.newaxis, :, :, :]/255.0)
			features.append(f)


		for i in tqdm(indexes, desc="Calculating dist matrix"):
			X1 = features[i[0]]
			Y1 = self.newLabels[i[0]]
			X2 = features[i[1]]
			Y2 = self.newLabels[i[1]]

			d = self.score_model.predict([X1, X2])[0][0]
			dist[i[0], i[1]] = d
			dist[i[1], i[0]] = d

		# print(dist)

		samemask = np.zeros(dist.shape)
		for i in range(len(self.list_IDs)):
			for j in range(len(self.list_IDs)):
				if i == j:
					samemask[i][j] = 0
				else:
					samemask[i][j] = np.float32(self.newLabels[i] == self.newLabels[j])


		diffmask = np.zeros(dist.shape)
		for i in range(len(self.list_IDs)):
			for j in range(len(self.list_IDs)):
				if i == j:
					diffmask[i][j] = 0
				else:
					diffmask[i][j] = np.float32(self.newLabels[i] != self.newLabels[j])


		samedist = dist * samemask
		diffdist = (1.0-dist) * diffmask

		same = np.argmax(samedist, axis=0)
		diff = np.argmax(diffdist, axis=0)

		self.indexes = []
		for i in range(len(self.list_IDs)):
			self.indexes.append((i, same[i], diff[i]))

		# print(self.indexes)
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, indexes):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X1 = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
		X2 = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
		X3 = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
		y = np.empty((self.batch_size), dtype=int)

		# Generate data
		for i, indx in enumerate(indexes):
			# Store sample
			X1[i,] = self.newImages[indx[0]]
			X2[i,] = self.newImages[indx[1]]
			X3[i,] = self.newImages[indx[2]]


		return [X1/255.0, X2/255.0, X3/255.0], keras.utils.to_categorical(y, num_classes=self.n_classes)



if __name__ == '__main__':
	params = {'dim': (224,224),
		  'batch_size': 1,
		  'n_classes': 2,
		  'n_channels': 3,
		  'shuffle': True,
		  'aug': True}
	
	datalist = pd.read_csv('dataset/train/train.txt', header=None, delimiter=' ')
	X_train, X_test, y_train, y_test = train_test_split(datalist[0].values, datalist[1].values, stratify=datalist[1], test_size=0.99)

	# print(datalist.head())
	# print(datalist.describe())
	# print(np.sum(datalist[1]), np.sum(y_train), np.sum(y_test))

	# dataGen = DataGenerator(X_train, y_train, **params)

	# import matplotlib.pyplot as plt

	# for i in range(len(dataGen)):
	# 	x, y = dataGen[i]
	# 	print(x.shape, y.shape)
	# 	# if y[0][0] == 1:
	# 	# 	plt.subplot(1, 2, 1)
	# 	# 	plt.imshow(x[0])
	# 	# 	plt.subplot(1, 2, 2)
	# 	# 	sig = np.random.randint(5, 8, 1)
	# 	# 	plt.imshow(gaussian(x[0], sigma=sig[0], preserve_range=True))
	# 	# 	plt.show()



	# dataGen = DataGeneratorPair(X_train, y_train, **params)

	# import matplotlib.pyplot as plt

	# for i in range(len(dataGen)):
	# 	x, y = dataGen[i]
	# 	print(len(x), x[0].shape, x[1].shape, y.shape, y)
	# 	plt.subplot(1, 2, 1)
	# 	plt.imshow(x[0][0])
	# 	plt.subplot(1, 2, 2)
	# 	plt.imshow(x[1][0])
	# 	plt.show()


	# np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
	# print(X_train.shape, y_train)

	# dist = np.random.rand(X_train.shape[0], X_train.shape[0])
	# print(dist)

	# samemask = np.zeros((y_train.shape[0], y_train.shape[0]))
	# for i in range(y_train.shape[0]):
	# 	for j in range(y_train.shape[0]):
	# 		if i == j:
	# 			samemask[i][j] = 0
	# 		else:
	# 			samemask[i][j] = np.float32(y_train[i] == y_train[j])


	# diffmask = np.zeros((y_train.shape[0], y_train.shape[0]))
	# for i in range(y_train.shape[0]):
	# 	for j in range(y_train.shape[0]):
	# 		if i == j:
	# 			diffmask[i][j] = 0
	# 		else:
	# 			diffmask[i][j] = np.float32(y_train[i] != y_train[j])


	# samedist = dist * samemask
	# diffdist = (1.0-dist) * diffmask

	# print(samedist)
	# print(np.argmax(samedist, axis=0))
	# print(diffdist)
	# print(np.argmax(diffdist, axis=0))

	base_model = ResNet50(weights='imagenet', include_top=False)
	base_model.summary()
	print(base_model.output_shape)


	score_inp_left = Input(shape=base_model.output_shape[1:])
	score_inp_right = Input(shape=base_model.output_shape[1:])
	xl = GlobalAveragePooling2D()(score_inp_left)
	xr = GlobalAveragePooling2D()(score_inp_right)
	d1 = Subtract()([xl, xr])
	d1 = Lambda(lambda val: K.abs(val))(d1)
	x = Dense(1024, activation='relu')(d1)
	predictions = Dense(2, activation='softmax')(x)
	score_model = Model(inputs=[score_inp_left, score_inp_right], outputs=predictions)
	score_model.summary()


	left_input = Input(params['dim'] + (params['n_channels'],))
	right_input = Input(params['dim'] + (params['n_channels'],))
	bl = base_model(left_input)
	br = base_model(right_input)
	p = score_model([bl, br])
	model = Model(inputs=[left_input, right_input], outputs=p)
	model.summary()

	

	dataGen = DataGeneratorPairHard(X_train, y_train, base_model, score_model, **params)

	import matplotlib.pyplot as plt

	for i in range(len(dataGen)):
		x, y = dataGen[i]
		print(len(x), x[0].shape, x[1].shape, y.shape, y)
		plt.subplot(1, 2, 1)
		plt.imshow(x[0][0])
		plt.subplot(1, 2, 2)
		plt.imshow(x[1][0])
		plt.show()