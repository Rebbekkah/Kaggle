import os, os.path
import pandas as pd
import numpy as np
import imageio
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import matplotlib.image as mpimg
from tensorflow import keras
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

#classifier = Sequential()


#import tensorflow.keras.layers.Maxpool2D
'''
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.python.keras.models import Sequential
'''



def folder(path_test, path_train, path_val):
	test_normal = []
	test_pneu = []
	train_normal = []
	train_pneu = []
	val_normal = []
	val_pneu = []

	list_path = [path_test, path_train, path_val]
	for path in list_path:
		if os.path.exists(os.path.join(path, "NORMAL")):
			new_path = os.path.join(path, "NORMAL")
			for file in os.listdir(new_path):
				if str(new_path) == os.path.join(path_test, "NORMAL"):
					test_normal.append(os.path.join(new_path, file))						
				elif str(new_path) == os.path.join(path_train, "NORMAL"):
					train_normal.append(os.path.join(new_path, file))
				elif str(new_path) == os.path.join(path_val, "NORMAL"):
					val_normal.append(os.path.join(new_path, file))
		if os.path.exists(os.path.join(path, "PNEUMONIA")):
			new_path = os.path.join(path, "PNEUMONIA")
			for file in os.listdir(new_path):
				if str(new_path) == os.path.join(path_test, "PNEUMONIA"):
					test_pneu.append(os.path.join(new_path, file))		
				elif str(new_path) == os.path.join(path_train, "PNEUMONIA"):
					train_pneu.append(os.path.join(new_path, file))
				elif str(new_path) == os.path.join(path_val, "PNEUMONIA"):
					val_pneu.append(os.path.join(new_path, file))
	return test_normal, test_pneu, train_normal, train_pneu, val_normal, val_pneu



def as_df(test_normal, test_pneu, train_normal, train_pneu, val_normal, val_pneu) :

	test = test_normal + test_pneu
	train = train_normal + train_pneu
	val = val_normal + val_pneu
	
	liste = [[test_normal, test_pneu], [train_normal, train_pneu], [val_normal, val_pneu]]
	list_df = []
	print("---------")
	print(liste[2])
	for normal, pneu in liste :
		cls = []
		for index in range(len(normal)) :
			cls.append("normal")
		for index in range(len(pneu)) :
			cls.append("pneumonia")

		list_df.append(pd.DataFrame({'class' : cls, 'image' : normal+pneu}))
		
		print(list_df)
		# print(len(list_df))
	return list_df
	

def data_visualisation(dataframes) :
	hist = []
	# camembert = []
	# labels = 'normal', 'pneumonia'
	#print(labels)
	for df in dataframes :
		hist.append(df['class'].hist())
		#camembert.append(df['class'].pie())
		#plt.show()

	# for plot in hist :
	# 	#plot_hist = plt.plot(plot)
	# 	plt.show()

	return hist


def data_splitting(dataframes) :
	df_train = dataframes[1]
	train_len = 0.80*len(df_train)

	shuffled_train =  df_train.sample(frac = 1).reset_index(drop = True)
	print("SHUFFELD DATA ----------------------", shuffled_train)
	train_data = shuffled_train.iloc[1:int(train_len), :]
	print("TRAIN DATA ----------------------------")
	print(train_data)
	val_data = shuffled_train.iloc[int(train_len):len(shuffled_train), :]
	print("VAL DATA ----------------------------")
	print(val_data)

	return train_data, val_data


def Data_Augmentation(train_df, val_df, list_df) :
	print("--------------------DATA AUGMENTATION ------------------------")

	test_df = list_df[0]
	print("DATAFRAMES TYPE", type(test_df), type(train_df),type(val_df), test_df)
	train_datagen = ImageDataGenerator(rescale = 1. / 255, shear_range = 0.2, zoom_range = 0.2,
			horizontal_flip = True, rotation_range = 0.1, width_shift_range = 0.1, height_shift_range = 0.1)
	# print(type(train_datagen))
	# print(train_datagen)

	val_datagen = ImageDataGenerator(rescale = 1/ 255.)
	# print("----------------", type(val_datagen), "----------------")
	# print(val_datagen)
	print("CALIBRATION")
	calibration_train = train_datagen.flow_from_dataframe(dataframe = train_df, directory = path_train, ################changer les paths
			x_col = 'image', y_col = 'class', class_mode = 'binary',
			batch_size = BATCH_SIZE, seed = SEED_SET, target_size = (IMG_SIZE, IMG_SIZE))
	print("TRAIN", type(calibration_train), calibration_train, "\n")

	calibration_val = val_datagen.flow_from_dataframe(dataframe = val_df, directory = path_val, x_col = 'image',
			 y_col = 'class', class_mode = 'binary',
			batch_size = BATCH_SIZE, seed = SEED_SET, target_size = (IMG_SIZE, IMG_SIZE))
	print("VALIDATION", type(calibration_val), calibration_val, "\n")


	test_datagen = ImageDataGenerator(rescale = 1/ 255.)
	calibration_test = test_datagen.flow_from_dataframe(dataframe = test_df, directory = path_test, x_col = 'image', 
				y_col = 'class', class_mode = 'binary', 
			batch_size = 1, shuffle = False, target_size = (IMG_SIZE, IMG_SIZE))
	print("test", type(calibration_test), calibration_test, "\n")
	#for i in calibration_val :
	#	print(i)

	return calibration_test, calibration_train, calibration_val


def Custom() :
	# callbacks pour éviter le surapprentissage
	cb = callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0.0000000001, patience = 50, restore_best_weights = True)
	control_learning_rate = callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.25, patience = 10, min_lr = 0.1, 
			min_delta = 0.0000001, cooldown = 5, verbose = 1)

	return cb, control_learning_rate

def model():
	print("CNN----------------------------------")
	filters = 4
	#inputs = layers.Input(shape = (IMG_SIZE, IMG_SIZE, 3))
	inputs = layers.Input(shape = (IMG_SIZE, IMG_SIZE, 3))

	print("DIM IMPUTS {}".format(inputs))


	#Première couche
	x = layers.Conv2D(filters = filters, kernel_size = 4, strides = (1, 1), padding = 'valid')(inputs)
	x = layers.BatchNormalization()(x) #Peut être utilisé avant ou après fonction d'activation/Maxpooling
	x = layers.Activation('relu')(x)
	x = layers.MaxPooling2D(pool_size = 2, strides = 1, padding = 'valid')(x)
	#x = layers.MaxPooling2D()(x)
	#print(type(x))
	#print(x)
	x = layers.Dropout(0.3)(x)

	#Seconde couche
	x = layers.Conv2D(filters = filters*2, kernel_size = 4, strides = (1, 1), padding = 'valid')(inputs)
	x = layers.BatchNormalization()(x) 
	x = layers.Activation('relu')(x)
	x = layers.MaxPooling2D(pool_size = 2, strides = 1, padding = 'valid')(x)
	x = layers.Dropout(0.3)(x)

	#Troisième couche
	x = layers.Conv2D(filters = filters*3, kernel_size = 4, strides = (1, 1), padding = 'valid')(inputs)
	#x = layers.Conv2D(filters = filters*3, kernel_size = 4, strides = (1, 1), padding = 'valid')(inputs)
	x = layers.BatchNormalization()(x) 
	x = layers.Activation('relu')(x)
	x = layers.MaxPooling2D(pool_size = 2, strides = 1, padding = 'valid')(x)
	x = layers.Dropout(0.5)(x)


	x = layers.Flatten()(x)
	x = layers.Dense(filters*3, activation = 'relu')(x)
	x = layers.Dropout(0.6)(x)


	#Dernière couche --> sigmoid activation car résultat binaire
	output = layers.Dense(1, activation = 'sigmoid')(x)

	my_model = keras.Model(inputs = [inputs], outputs = output)

	#print(type(my_model), my_model)
	return my_model
	'''
	#Input shape = [width, height, color channels]
	inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
	print("DIM IMPUTS {}".format(inputs))


	# Block One
	x = layers.Conv2D(filters=16, kernel_size=3, padding='valid')(inputs)
	x = layers.BatchNormalization()(x)
	x = layers.Activation('relu')(x)
	x = layers.MaxPool2D()(x)
	x = layers.Dropout(0.2)(x)

	# Block Two
	x = layers.Conv2D(filters=32, kernel_size=3, padding='valid')(x)
	x = layers.BatchNormalization()(x)
	x = layers.Activation('relu')(x)
	x = layers.MaxPool2D()(x)
	x = layers.Dropout(0.2)(x)
	
	# Block Three
	x = layers.Conv2D(filters=64, kernel_size=3, padding='valid')(x)
	x = layers.Conv2D(filters=64, kernel_size=3, padding='valid')(x)
	x = layers.BatchNormalization()(x)
	x = layers.Activation('relu')(x)
	x = layers.MaxPool2D()(x)
	x = layers.Dropout(0.4)(x)

	# Head
	#x = layers.BatchNormalization()(x)
	x = layers.Flatten()(x)
	x = layers.Dense(64, activation='relu')(x)
	x = layers.Dropout(0.5)(x)
	
	#Final Layer (Output)
	output = layers.Dense(1, activation='sigmoid')(x)
	
	my_model = keras.Model(inputs=[inputs], outputs=output)
	
	return my_model
	'''


def model_info(model, train, val, callback, plateau) :
	model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.005), 
		loss = 'binary_crossentropy', metrics = ['accuracy'], steps_per_execution = 1)
	#model.compile(optimizer = keras.optimizers.Nadam(learning_rate = 0.005), 
	#	loss = 'binary_crossentropy', steps_per_execution = 1)
	model_summary = model.summary()
	print(model_summary)

	print("LEN TRAIN {}".format(len(train)))
	print("LEN val {}".format(len(val)))

	print(train)
	print("SHAPE TRAIN ---> {}".format(np.shape(train)))
	
	print(train['image'])
	print("PAHT TRAIN ---->  {}".format(path_train))

	print("----------------")
	#train = np.expand_dims(train, axis = (0, 1))
	#train = np.expand_dims(train, axis = 1)
	#print("SHAPE TRAIN EXPAND ---> {}".format(np.shape(train)))

	#train.reshape([-1, IMG_SIZE, IMG_SIZE, 3])
	#print(train)

	#print(val)
	#val = np.expand_dims(val, axis = (0, 1))
	#print(val)
	#val = np.expand_dims(val, axis = 0)

	history = model.fit(train, batch_size = BATCH_SIZE, epochs = 30, validation_data = val,
		callbacks = [callback, plateau], steps_per_epoch = (len(train)/BATCH_SIZE),
		validation_steps = (len(val)/BATCH_SIZE))

	print(history)


if __name__ == "__main__":
	path_test = "/Users/rgoulanc/Desktop/Rebecca/FAC/M2BI/KAGGLE/chest_xray/test"
	path_train = "/Users/rgoulanc/Desktop/Rebecca/FAC/M2BI/KAGGLE/chest_xray/train"
	path_val = "/Users/rgoulanc/Desktop/Rebecca/FAC/M2BI/KAGGLE/chest_xray/val"
	#path_test = "/home/sdv/m2bi/rgoulancourt/Kaggle/data/archive/chest_xray/test"
	#path_train = "/home/sdv/m2bi/rgoulancourt/Kaggle/data/archive/chest_xray/train"
	#path_val = "/home/sdv/m2bi/rgoulancourt/Kaggle/data/archive/chest_xray/val"

	# Important parameters
	IMG_SIZE = 224
	BATCH_SIZE = 32
	SEED_SET = 42


	test_normal, test_pneu, train_normal, train_pneu, val_normal, val_pneu = folder(path_test, path_train, path_val)
	all_df = as_df(test_normal, test_pneu, train_normal, train_pneu, val_normal, val_pneu)
	graph = data_visualisation(all_df)
	train_data, val_data = data_splitting(all_df)
	callback, lr = Custom()
	img_train, img_test, img_val = Data_Augmentation(train_data, val_data, all_df)

	get_model = model()
	#info = model_info(get_model, augment_train)
	info = model_info(get_model, train_data, val_data, callback, lr)
	#augment_test, augment_train, augment_val = Data_Augmentation(train_data, val_data, all_df)




