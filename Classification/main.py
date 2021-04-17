import os
import argparse
import numpy as np
import pandas as pd
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from datetime import datetime
from sklearn.model_selection import train_test_split
from data_loader import DataGenerator


def get_base_model(name='resnet50'):
	if name == 'resnet50':
		base_model = ResNet50(weights='imagenet', include_top=False)
	elif name == 'mobilenetv2':
		base_model = MobileNetV2(weights='imagenet', include_top=False)

	return base_model


def date_str():
	return datetime.now().__str__().replace("-", "_").replace(" ", "_").replace(":", "_")


if __name__ == '__main__':

	# params
	parser = argparse.ArgumentParser()

	# Model configuration.
	parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone architecture')
	parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

	config = parser.parse_args()


	# callbacks
	date_str_ = date_str()

	if not os.path.exists('models/{}_pair_batch_{}_{}/'.format(config.backbone, config.batch_size, date_str_)):
		os.makedirs('models/{}_pair_batch_{}_{}/'.format(config.backbone, config.batch_size, date_str_))

	model_checkpoint = ModelCheckpoint('models/{}_pair_batch_{}_{}/'.format(config.backbone, config.batch_size, date_str_) + '{epoch:03d}-{val_loss:.6f}-{val_acc:.4f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
	early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='min')
	tensorboard_logs = TensorBoard(log_dir='./tensorboard_logs/{}_pair_batch_{}_{}'.format(config.backbone, config.batch_size, date_str_), histogram_freq=0, batch_size=config.batch_size, write_graph=True, write_grads=False, write_images=True)
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
	callbacks = [model_checkpoint, early_stopping, tensorboard_logs, reduce_lr]


	# data
	train_params = {'dim': (224,224),
		  'batch_size': config.batch_size,
		  'n_classes': 2,
		  'n_channels': 3,
		  'shuffle': True,
		  'aug': True}

	val_params = {'dim': (224,224),
		  'batch_size': config.batch_size,
		  'n_classes': 2,
		  'n_channels': 3,
		  'shuffle': False,
		  'aug': False}

	datalist = pd.read_csv('dataset/train/train.txt', header=None, delimiter=' ')
	X_train, X_test, y_train, y_test = train_test_split(datalist[0].values, datalist[1].values, stratify=datalist[1], test_size=0.2)

	train_generator = DataGenerator(X_train, y_train, **train_params)
	val_generator = DataGenerator(X_test, y_test, **val_params)


	print("Training size =", len(train_generator))
	print("Validation size =", len(val_generator))


	# models
	# add a global spatial average pooling layer
	base_model = get_base_model()
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	# let's add a fully-connected layer
	x = Dense(1024, activation='relu')(x)
	# and a logistic layer -- let's say we have 2 classes
	predictions = Dense(2, activation='softmax')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)

	# first: train only the top layers (which were randomly initialized)
	# i.e. freeze all convolutional InceptionV3 layers
	for layer in base_model.layers:
		layer.trainable = False

	# compile the model (should be done *after* setting layers to non-trainable)
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	# train the model on the new data for a few epochs
	model.fit_generator(generator=train_generator,
					validation_data=val_generator,
					epochs=1,
					max_queue_size=8,
					shuffle=True,
					use_multiprocessing=True,
					workers=4)

	# unfreeze
	for layer in base_model.layers:
		layer.trainable = True

	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	# we train our model again
	model.fit_generator(generator=train_generator,
					validation_data=val_generator,
					epochs=1000,
					callbacks=callbacks,
					max_queue_size=8,
					shuffle=True,
					use_multiprocessing=True,
					workers=4)