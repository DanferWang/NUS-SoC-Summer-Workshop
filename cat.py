from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import Model,load_model
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import SGD

import numpy as np
import os.path

MODEL_FILE="cat.hd5"

def create_model(num_hidden,num_classes):
	base_model=InceptionV3(include_top=False,weights='imagenet')
	x=base_model.output
	x=GlobalAveragePooling2D()(x)
	x=Dense(num_hidden,activation='relu')(x)
	predictions=Dense(num_classes,activation='softmax')(x)

	for layer in base_model.layers:
		layer.trainable=False
	
	model=Model(inputs=base_model.input,outputs=predictions)

	return model

def load_existing(model_file):
	model=load_model(model_file)
	numlayers=len(model.layers)

	for layer in model.layers[:numlayers-3]:
		layer.trainable=False
	
	for layer in model.layers[numlayers-3:]:
		layer.trainable=True

	return model

def train(model_file,train_path,validation_path,num_hidden=200,num_classes=5,steps=32,num_epochs=20,save_period=1):
	if os.path.exists(model_file):
		print "\n***Existing model found at {}. Loading.***\n\n".format(model_file)
		model=load_existing(model_file)
	else:
		print "\n***Creating new model ***\n\n"
		model=create_model(num_hidden,num_classes)
	
	model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

	checkpoint=ModelCheckpoint(model_file,period=save_period)

	train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

	test_datagen=ImageDataGenerator(rescale=1./255)

	train_generator=train_datagen.flow_from_directory(train_path,target_size=(249,249),batch_size=32,class_mode="categorical")

	validation_generator=test_datagen.flow_from_directory(validation_path,target_size=(249,249),batch_size=32,class_mode='categorical')

	model.fit_generator(train_generator,steps_per_epoch=steps,epochs=num_epochs,callbacks=[checkpoint],validation_data=validation_generator,validation_steps=50)

	for layer in model.layers[:249]:
		layer.trainable=False
	
	for layer in model.layers[249:]:
		layer.trainable=True
	
	model.compile(optimizer=SGD(lr=0.0001,momentum=0.9),loss='categorical_crossentropy',metrics=['accuracy'])

	model.fit_generator(train_generator,steps_per_epoch=steps,epochs=num_epochs,callbacks=[checkpoint],validation_data=validation_generator,validation_steps=50)

'''
def prepare_data(train_or_test_dir):
    x_train_test = []
    ima1 = os.listdir(train_or_test_dir)
    for i in ima1:
        x_train_test.append(convert_image_array(i, train_or_test_dir))
    x_train_test = np.array(x_train_test)
    y_train_test = []
    for filename in ima1:
        y_train_test.append(int(filename.split('_')[0]))
    y_train_test = np.array(y_train_test)
    y_train_test = np_utils.to_categorical(y_train_test)
    x_train_test = x_train_test.astype('float32')
    x_train_test /= 255

    return x_test, y_test
'''

def main():
	train(MODEL_FILE,train_path="cat_photos/training",validation_path="cat_photos/training",steps=120,num_hidden=10)
    '''
    x_test, y_test = prepare_data("cat_photos/testing")
    test_loss,test_acc=model.evaluate(x_test, y_test)
    print("Test accuracy:%3.2f,loss:%3.2f"%(test_acc,test_loss))
    '''

if __name__=="__main__":
	main()