import matplotlib
matplotlib.use('Agg')
import os
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
from skimage import transform
from skimage.transform import AffineTransform
import scipy.ndimage as ndimage
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Input,TimeDistributed,BatchNormalization,MaxPooling2D,Activation,ZeroPadding2D,Add
from tensorflow.keras.layers import Conv2D,Dropout,Flatten,Dense,LSTM,concatenate,AveragePooling2D,GlobalMaxPooling2D
from tensorflow.keras.models import Model,Sequential,load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import itertools



class Categorizers():

	def __init__(self):

		self.extension_image=('.png','.PNG','.jpeg','.JPEG','.jpg','.JPG','.tiff','.TIFF','.bmp','.BMP')
		self.extension_video=('.avi','.mpg','.wmv','.mp4','.mkv','.m4v','.mov')
		self.classnames=None


	def rename_label(self,file_path,new_path,resize=None):

		# file_path: the folder that stores the sorted, unprepared examples
		# new_path: the folder that stores all prepared examples, which can be directly used for training a Categorizer
		# resize: if not None, resize the frames in animations / pattern images to the target size

		folder_list=[i for i in os.listdir(file_path) if os.path.isdir(os.path.join(file_path,i))]

		if len(folder_list)<2:

			print('You need at least 2 categories of behaviors!')
			print('Preparation aborted!')

		else:

			print('Behavior names are: '+str(folder_list))
			previous_lenth=None
			imagedata=False

			for folder in folder_list:

				name_list=[i for i in os.listdir(os.path.join(file_path,folder)) if i.endswith('.avi')]

				if len(name_list)==0:
					name_list=[i for i in os.listdir(os.path.join(file_path,folder)) if i.endswith('.jpg')]
					imagedata=True
			
				for i in name_list:

					if imagedata:

						image=os.path.join(file_path,folder,i)
						new_image=os.path.join(new_path,str(name_list.index(i))+'_'+folder+'.jpg')
						image=cv2.imread(image)
						if resize is not None:
							image=cv2.resize(image,(resize,resize),interpolation=cv2.INTER_AREA)
						cv2.imwrite(new_image,image)

					else:

						animation=os.path.join(file_path,folder,i)
						pattern_image=os.path.join(file_path,folder,os.path.splitext(i)[0]+'.jpg')
						current_length=0

						new_animation=os.path.join(new_path,str(name_list.index(i))+'_'+folder+'.avi')
						new_pattern_image=os.path.join(new_path,str(name_list.index(i))+'_'+folder+'.jpg')
						writer=None
						capture=cv2.VideoCapture(animation)
						fps=round(capture.get(cv2.CAP_PROP_FPS))
						while True:
							retval,frame=capture.read()
							current_length+=1
							if frame is None:
								break
							if resize is not None:
								frame=cv2.resize(frame,(resize,resize),interpolation=cv2.INTER_AREA)
							if writer is None:
								(h,w)=frame.shape[:2]
								writer=cv2.VideoWriter(new_animation,cv2.VideoWriter_fourcc(*'MJPG'),fps,(w,h),True)
							writer.write(frame)
						capture.release()
						writer.release()
						pattern_image=cv2.imread(pattern_image)
						if resize is not None:
							pattern_image=cv2.resize(pattern_image,(resize,resize),interpolation=cv2.INTER_AREA)
						cv2.imwrite(new_pattern_image,pattern_image)
						if previous_lenth is None:
							previous_lenth=current_length
						else:
							if previous_lenth!=current_length:
								previous_lenth=current_length
								print('Inconsistent duration of animation detected at: '+str(i)+'. Check the duration of animations!')

			print('All prepared training examples stored in: '+str(new_path))


	def build_data(self,path_to_animations,dim_tconv=0,dim_conv=64,channel=1,time_step=18,aug_methods=[],background_free=True):

		animations=deque()
		pattern_images=deque()
		labels=deque()
		amount=0

		if len(aug_methods)==0:

			methods=['orig']

		else:

			remove=[]

			all_methods=['orig','rot1','rot2','rot3','rot4','rot5','rot6','shrp','shrn','sclh','sclw','del1','del2']
			options=['rot7','flph','flpv','brih','bril','shrr','sclr']
			for r in range(1,len(options)+1):
				all_methods.extend([''.join(c) for c in itertools.combinations(options,r)])

			for i in all_methods:
				if 'all' not in aug_methods:
					if 'random rotation' not in aug_methods:
						if 'rot' in i:
							remove.append(i)
					if 'horizontal flipping' not in aug_methods:
						if 'flph' in i:
							remove.append(i)
					if 'vertical flipping' not in aug_methods:
						if 'flpv' in i:
							remove.append(i)
					if 'random brightening' not in aug_methods:
						if 'brih' in i:
							remove.append(i)
					if 'random dimming' not in aug_methods:
						if 'bril' in i:
							remove.append(i)
					if 'random shearing' not in aug_methods:
						if 'shr' in i:
							remove.append(i)
					if 'random rescaling' not in aug_methods:
						if 'scl' in i:
							remove.append(i)
					if 'random deletion' not in aug_methods:
						if 'del' in i:
							remove.append(i)

			methods=list(set(all_methods)-set(remove))

		for i in path_to_animations:

			label=os.path.splitext(i)[0].split('_')[-1]
			path_to_pattern_image=os.path.splitext(i)[0]+'.jpg'
			
			random.shuffle(methods)

			for m in methods:

				if 'rot1' in m:
					angle=np.random.uniform(5,45)
				elif 'rot2' in m:
					angle=np.random.uniform(45,85)
				elif 'rot3' in m:
					angle=90.0
				elif 'rot4' in m:
					angle=np.random.uniform(95,135)
				elif 'rot5' in m:
					angle=np.random.uniform(135,175)
				elif 'rot6' in m:
					angle=180.0
				elif 'rot7' in m:
					angle=np.random.uniform(5,175)
				else:
					angle=None

				if 'flph' in m:
					code=1
				elif 'flpv' in m:
					code=0
				else:
					code=None

				if 'brih' in m:
					beta=np.random.uniform(10,50)
				elif 'bril' in m:
					beta=np.random.uniform(-50,10)
				else:
					beta=None

				if 'shrp' in m:
					shear=np.random.uniform(0.15,0.21)
				elif 'shrn' in m:
					shear=np.random.uniform(-0.21,-0.15)
				elif 'shrr' in m:
					shear=np.random.uniform(-0.21,0.21)
				else:
					shear=None

				if 'sclh' in m:
					width=0
					scale=np.random.uniform(0.6,0.9)
				elif 'sclw' in m:
					width=1
					scale=np.random.uniform(0.6,0.9)
				elif 'sclr' in m:
					width=random.randint(0,1)
					scale=np.random.uniform(0.6,0.9)
				else:
					scale=None

				if 'del1' in m:
					to_delete=[i for i in range(random.randint(0,round(time_step/4)))]
				elif 'del2' in m:
					to_delete=[i for i in range(random.randint(int(time_step*3/4),time_step),time_step+1)]
				else:
					to_delete=None

				if dim_tconv!=0:

					capture=cv2.VideoCapture(i)
					animation=deque()
					frames=deque(maxlen=time_step)
					original_frame=None
					n=0

					while True:
						retval,frame=capture.read()
						if original_frame is None:
							original_frame=frame
						if frame is None:
							break
						frames.append(frame)

					frames_length=len(frames)
					if frames_length<time_step:
						for diff in range(time_step-frames_length):
							frames.append(np.zeros_like(original_frame))
						print('Inconsistent duration of animation detected at: '+str(i)+'.')
						print('Zero padding has been used, which may decrease the training accuracy.')

					for blob in frames:

						if to_delete is not None and n in to_delete:

							blob=np.zeros_like(original_frame)

						else:

							if code is not None:
								blob=cv2.flip(blob,code)

							if beta is not None:
								blob=blob.astype('float')
								if background_free:
									blob[blob>30]+=beta
								else:
									blob+=beta
								blob=np.uint8(np.clip(blob,0,255))

							if angle is not None:
								blob=ndimage.rotate(blob,angle,reshape=False,prefilter=False)

							if shear is not None:
								tf=AffineTransform(shear=shear)
								blob=transform.warp(blob,tf,order=1,preserve_range=True,mode='constant')

							if scale is not None:
								blob_black=np.zeros_like(blob)
								if width==0:
									blob_scl=cv2.resize(blob,(blob.shape[1],int(blob.shape[0]*scale)),interpolation=cv2.INTER_AREA)
								else:
									blob_scl=cv2.resize(blob,(int(blob.shape[1]*scale),blob.shape[0]),interpolation=cv2.INTER_AREA)
								blob_scl=img_to_array(blob_scl)
								x=(blob_black.shape[1]-blob_scl.shape[1])//2
								y=(blob_black.shape[0]-blob_scl.shape[0])//2
								blob_black[y:y+blob_scl.shape[0],x:x+blob_scl.shape[1]]=blob_scl
								blob=blob_black

						if channel==1:
							blob=cv2.cvtColor(np.uint8(blob),cv2.COLOR_BGR2GRAY)

						blob=cv2.resize(blob,(dim_tconv,dim_tconv),interpolation=cv2.INTER_AREA)
						blob=img_to_array(blob)
						animation.append(blob)

						n+=1

					capture.release()

					animations.append(np.array(animation))


				pattern_image=cv2.imread(path_to_pattern_image)

				if code is not None:
					pattern_image=cv2.flip(pattern_image,code)

				if angle is not None:
					pattern_image=ndimage.rotate(pattern_image,angle,reshape=False,prefilter=False)

				if shear is not None:
					tf=AffineTransform(shear=shear)
					pattern_image=transform.warp(pattern_image,tf,order=1,preserve_range=True,mode='constant')

				if scale is not None:
					pattern_image_black=np.zeros_like(pattern_image)
					if width==0:
						pattern_image_scl=cv2.resize(pattern_image,(pattern_image.shape[1],int(pattern_image.shape[0]*scale)),interpolation=cv2.INTER_AREA)
					else:
						pattern_image_scl=cv2.resize(pattern_image,(int(pattern_image.shape[1]*scale),pattern_image.shape[0]),interpolation=cv2.INTER_AREA)
					x=(pattern_image_black.shape[1]-pattern_image_scl.shape[1])//2
					y=(pattern_image_black.shape[0]-pattern_image_scl.shape[0])//2
					pattern_image_black[y:y+pattern_image_scl.shape[0],
					x:x+pattern_image_scl.shape[1],:]=pattern_image_scl
					pattern_image=pattern_image_black

				pattern_image=cv2.resize(pattern_image,(dim_conv,dim_conv),interpolation=cv2.INTER_AREA)
				pattern_images.append(img_to_array(pattern_image))

				labels.append(label)

				amount+=1
				if amount%10000==0:
					print('The augmented example amount: '+str(amount))
					print(datetime.datetime.now())

		if dim_tconv!=0:
			animations=np.array(animations,dtype='float32')/255.0
		pattern_images=np.array(pattern_images,dtype='float32')/255.0
		labels=np.array(labels)

		return animations,pattern_images,labels


	def simple_vgg(self,inputs,filters,classes=3,level=2,with_classifier=False):

		if level<2:
			layers=[2]
		elif level<3:
			layers=[2,3]
		elif level<4:
			layers=[2,3,4]
		else:
			layers=[2,3,4,4]

		for i in layers:
			for n in range(i):
				if n==0:
					if layers.index(i)==0:
						x=Conv2D(filters,kernel_size=(3,3),padding='same',activation='relu')(inputs)
						x=BatchNormalization()(x)
					else:
						x=Conv2D(filters,kernel_size=(3,3),padding='same',activation='relu')(x)
						x=BatchNormalization()(x)
				else:
					x=Conv2D(filters,kernel_size=(3,3),padding='same',activation='relu')(x)
					x=BatchNormalization()(x)
			x=MaxPooling2D(pool_size=(2,2))(x)
			filters=int(filters*2)

		x=Flatten()(x)

		if with_classifier is False:

			return x

		else:

			x=Dense(int(filters/2),activation='relu')(x)
			x=BatchNormalization()(x)
			x=Dropout(0.5)(x)
			if classes==2:
				x=Dense(1,activation='sigmoid')(x)
			else:
				x=Dense(classes,activation='softmax')(x)

			model=Model(inputs=inputs,outputs=x)

			return model


	def simple_tvgg(self,inputs,filters,classes=3,level=2,with_classifier=False):

		if level<2:
			layers=[2]
		elif level<3:
			layers=[2,3]
		elif level<4:
			layers=[2,3,4]
		else:
			layers=[2,3,4,4]

		for i in layers:
			for n in range(i):
				if n==0:
					if layers.index(i)==0:
						x=TimeDistributed(Conv2D(filters,kernel_size=(3,3),padding='same',activation='relu'))(inputs)
						x=TimeDistributed(BatchNormalization())(x)
					else:
						x=TimeDistributed(Conv2D(filters,kernel_size=(3,3),padding='same',activation='relu'))(x)
						x=TimeDistributed(BatchNormalization())(x)
				else:
					x=TimeDistributed(Conv2D(filters,kernel_size=(3,3),padding='same',activation='relu'))(x)
					x=TimeDistributed(BatchNormalization())(x)
			x=TimeDistributed(MaxPooling2D(pool_size=(2,2)))(x)
			filters=int(filters*2)

		x=TimeDistributed(Flatten())(x)
		x=LSTM(int(filters/2),return_sequences=False,return_state=False)(x)

		if with_classifier is False:

			return x

		else:

			x=Dense(int(filters/2),activation='relu')(x)
			x=BatchNormalization()(x)
			x=Dropout(0.5)(x)
			if classes==2:
				x=Dense(1,activation='sigmoid')(x)
			else:
				x=Dense(classes,activation='softmax')(x)

			model=Model(inputs=inputs,outputs=x)

			return model


	def res_block(self,x,filters,strides=2,block=False,basic=True):

		shortcut=x

		if basic:

			x=ZeroPadding2D((1,1))(x)
			x=Conv2D(filters,(3,3),strides=(strides,strides))(x)

		else:

			x=Conv2D(filters,(1,1),strides=(strides,strides))(x)

		x=BatchNormalization()(x)
		x=Activation('relu')(x)

		x=ZeroPadding2D((1,1))(x)
		x=Conv2D(filters,(3,3),strides=(1,1))(x)
		x=BatchNormalization()(x)

		if basic:

			if block is False:
				shortcut=Conv2D(filters,(1,1),strides=(strides,strides))(shortcut)
				shortcut=BatchNormalization()(shortcut)

		else:

			x=Activation('relu')(x)
		
			x=Conv2D(int(filters*4),(1,1),strides=(1,1))(x)
			x=BatchNormalization()(x)

			if block is False:
				shortcut=Conv2D(filters*4,(1,1),strides=(strides,strides))(shortcut)
				shortcut=BatchNormalization()(shortcut)

		x=Add()([x,shortcut])
		x=Activation('relu')(x)

		return x


	def tres_block(self,x,filters,strides=2,block=False,basic=True):

		shortcut=x

		if basic:

			x=TimeDistributed(ZeroPadding2D((1,1)))(x)
			x=TimeDistributed(Conv2D(filters,(3,3),strides=(strides,strides)))(x)

		else:

			x=TimeDistributed(Conv2D(filters,(1,1),strides=(strides,strides)))(x)

		x=TimeDistributed(BatchNormalization())(x)
		x=TimeDistributed(Activation('relu'))(x)

		x=TimeDistributed(ZeroPadding2D((1,1)))(x)
		x=TimeDistributed(Conv2D(filters,(3,3),strides=(1,1)))(x)
		x=TimeDistributed(BatchNormalization())(x)

		if basic:

			if block is False:
				shortcut=TimeDistributed(Conv2D(filters,(1,1),strides=(strides,strides)))(shortcut)
				shortcut=TimeDistributed(BatchNormalization())(shortcut)

		else:

			x=TimeDistributed(Activation('relu'))(x)
		
			x=TimeDistributed(Conv2D(int(filters*4),(1,1),strides=(1,1)))(x)
			x=TimeDistributed(BatchNormalization())(x)

			if block is False:
				shortcut=TimeDistributed(Conv2D(int(filters*4),(1,1),strides=(strides,strides)))(shortcut)
				shortcut=TimeDistributed(BatchNormalization())(shortcut)

		x=Add()([x,shortcut])
		x=TimeDistributed(Activation('relu'))(x)

		return x


	def simple_resnet(self,inputs,filters,classes=3,level=5,with_classifier=False):

		x=ZeroPadding2D((3,3))(inputs)
		x=Conv2D(filters,(5,5),strides=(2,2))(x)
		x=BatchNormalization()(x)
		x=Activation('relu')(x)
		x=MaxPooling2D((3,3),strides=(2,2))(x)

		if level<6:
			layers=[2,2,2,2]
			basic=True
		elif level<7:
			layers=[3,4,6,3]
			basic=True
		else:
			layers=[3,4,6,3]
			basic=False

		for i in layers:
			for n in range(i):
				if n==0:
					if layers.index(i)==0:
						x=self.res_block(x,filters,strides=1,block=False,basic=basic)
					else:
						x=self.res_block(x,filters,strides=2,block=False,basic=basic)
				else:
					x=self.res_block(x,filters,strides=1,block=True,basic=basic)
			filters=int(filters*2)

		x=AveragePooling2D((2,2))(x)
		x=Flatten()(x)

		if with_classifier is False:

			return x

		else:

			x=Dropout(0.5)(x)
			if classes==2:
				x=Dense(1,activation='sigmoid')(x)
			else:
				x=Dense(classes,activation='softmax')(x)

			model=Model(inputs=inputs,outputs=x)

			return model


	def simple_tresnet(self,inputs,filters,classes=3,level=5,with_classifier=False):

		x=TimeDistributed(ZeroPadding2D((3,3)))(inputs)
		x=TimeDistributed(Conv2D(filters,(5,5),strides=(2,2)))(x)
		x=TimeDistributed(BatchNormalization())(x)
		x=TimeDistributed(Activation('relu'))(x)
		x=TimeDistributed(MaxPooling2D((3,3),strides=(2,2)))(x)

		if level<6:
			layers=[2,2,2,2]
			basic=True
		elif level<7:
			layers=[3,4,6,3]
			basic=True
		else:
			layers=[3,4,6,3]
			basic=False

		for i in layers:
			for n in range(i):
				if n==0:
					if layers.index(i)==0:
						x=self.tres_block(x,filters,strides=1,block=False,basic=basic)
					else:
						x=self.tres_block(x,filters,strides=2,block=False,basic=basic)
				else:
					x=self.tres_block(x,filters,strides=1,block=True,basic=basic)
			filters=int(filters*2)

		x=TimeDistributed(AveragePooling2D((2,2)))(x)
		x=TimeDistributed(Flatten())(x)

		if level==5:
			x=LSTM(1024,return_sequences=False,return_state=False)(x)
		elif level==6:
			x=LSTM(2048,return_sequences=False,return_state=False)(x)
		else:
			x=LSTM(4096,return_sequences=False,return_state=False)(x)
			
		if with_classifier is False:

			return x

		else:

			if level==5:
				x=Dense(1024,activation='relu')(x)
			elif level==6:
				x=Dense(2048,activation='relu')(x)
			else:
				x=Dense(4096,activation='relu')(x)
				
			x=Dropout(0.5)(x)
			if classes==2:
				x=Dense(1,activation='sigmoid')(x)
			else:
				x=Dense(classes,activation='softmax')(x)

			model=Model(inputs=inputs,outputs=x)

			return model


	def combined_network(self,time_step=15,dim_tconv=32,dim_conv=64,channel=1,classes=9,level_tconv=1,level_conv=2):

		animation_inputs=Input(shape=(time_step,dim_tconv,dim_tconv,channel))
		pattern_image_inputs=Input(shape=(dim_conv,dim_conv,3))

		filters_tconv=8
		filters_conv=8

		for i in range(round(dim_tconv/60)):
			filters_tconv=min(int(filters_tconv*2),64)
		
		for i in range(round(dim_conv/60)):
			filters_conv=min(int(filters_conv*2),64)

		if level_tconv<5:
			animation_feature=self.simple_tvgg(animation_inputs,filters_tconv,level=level_tconv,with_classifier=False)
		else:
			animation_feature=self.simple_tresnet(animation_inputs,filters_tconv,level=level_tconv,with_classifier=False)
		
		if level_conv<5:
			pattern_image_feature=self.simple_vgg(pattern_image_inputs,filters_conv,level=level_conv,with_classifier=False)
		else:
			pattern_image_feature=self.simple_resnet(pattern_image_inputs,filters_conv,level=level_conv,with_classifier=False)

		merged_features=concatenate([animation_feature,pattern_image_feature])

		nodes=32
		for i in range(max(level_tconv,level_conv)):
			nodes=int(nodes*2)
		outputs=Dense(nodes,activation='relu')(merged_features)
		outputs=BatchNormalization()(outputs)
		outputs=Dropout(0.5)(outputs)
		if classes==2:
			predictions=Dense(1,activation='sigmoid')(outputs)
		else:
			predictions=Dense(classes,activation='softmax')(outputs)

		model=Model(inputs=[animation_inputs,pattern_image_inputs],outputs=predictions)

		return model


	def train_pattern_recognizer(self,data_path,model_path,out_path=None,dim=64,channel=3,time_step=18,level=2,aug_methods=[],augvalid=True,background_free=True):

		filters=8

		for i in range(round(dim/60)):
			filters=min(int(filters*2),64)

		inputs=Input(shape=(dim,dim,channel))

		print('Training the Categorizer w/ only Pattern Recognizer using the behavior examples in: '+str(data_path))

		files=[i for i in os.listdir(data_path) if i.endswith(self.extension_image)]

		path_files=[]
		labels=[]

		for i in files:
			path_file=os.path.join(data_path,i)
			path_files.append(path_file)
			labels.append(os.path.splitext(i)[0].split('_')[-1])

		labels=np.array(labels)
		lb=LabelBinarizer()
		labels=lb.fit_transform(labels)
		self.classnames=lb.classes_

		if len(list(self.classnames))<2:

			print('You need at least 2 categories of behaviors!')
			print('Training aborted!')

		else:

			print('Found behavior names: '+str(self.classnames))

			if background_free:
				background_code=0
			else:
				background_code=1

			parameters={'classnames':list(self.classnames),'dim_conv':int(dim),'channel':int(channel),'time_step':int(time_step),'network':0,'level_conv':int(level),'background_free':int(background_code)}
			pd_parameters=pd.DataFrame.from_dict(parameters)
			pd_parameters.to_csv(os.path.join(model_path,'model_parameters.txt'),index=False)

			(train_files,test_files,y1,y2)=train_test_split(path_files,labels,test_size=0.2,stratify=labels)

			print('Perform augmentation for the behavior examples...')
			print('This might take hours or days, depending on the capacity of your computer.')
			print(datetime.datetime.now())

			print('Start to augment training examples...')
			_,trainX,trainY=self.build_data(train_files,dim_tconv=0,dim_conv=dim,channel=channel,time_step=time_step,aug_methods=aug_methods,background_free=background_free)
			trainY=lb.fit_transform(trainY)
			print('Start to augment validation examples...')
			if augvalid:
				_,testX,testY=self.build_data(test_files,dim_tconv=0,dim_conv=dim,channel=channel,time_step=time_step,aug_methods=aug_methods,background_free=background_free)
			else:
				_,testX,testY=self.build_data(test_files,dim_tconv=0,dim_conv=dim,channel=channel,time_step=time_step,aug_methods=[],background_free=background_free)
			testY=lb.fit_transform(testY)

			with tf.device('CPU'):
				trainX=tf.convert_to_tensor(trainX)
				trainY=tf.convert_to_tensor(trainY)
				testX_tensor=tf.convert_to_tensor(testX)
				testY_tensor=tf.convert_to_tensor(testY)

			print('Training example shape : '+str(trainX.shape))
			print('Training label shape : '+str(trainY.shape))
			print('Validation example shape : '+str(testX.shape))
			print('Validation label shape : '+str(testY.shape))
			print(datetime.datetime.now())

			if trainX.shape[0]<5000:
				batch_size=8
			elif trainX.shape[0]<50000:
				batch_size=16
			else:
				batch_size=32

			if level<5:
				model=self.simple_vgg(inputs,filters,classes=len(self.classnames),level=level,with_classifier=True)
			else:
				model=self.simple_resnet(inputs,filters,classes=len(self.classnames),level=level,with_classifier=True)
			if len(self.classnames)==2:
				model.compile(optimizer=SGD(learning_rate=1e-4,momentum=0.9),loss='binary_crossentropy',metrics=['accuracy'])
			else:
				model.compile(optimizer=SGD(learning_rate=1e-4,momentum=0.9),loss='categorical_crossentropy',metrics=['accuracy'])

			cp=ModelCheckpoint(model_path,monitor='val_loss',verbose=1,save_best_only=True,save_weights_only=False,mode='min',save_freq='epoch')
			es=EarlyStopping(monitor='val_loss',min_delta=0.001,mode='min',verbose=1,patience=6,restore_best_weights=True)
			rl=ReduceLROnPlateau(monitor='val_loss',min_delta=0.001,factor=0.2,patience=3,verbose=1,mode='min',min_lr=1e-7)

			H=model.fit(trainX,trainY,batch_size=batch_size,validation_data=(testX_tensor,testY_tensor),epochs=1000000,callbacks=[cp,es,rl])

			model.save(model_path)
			print('Trained Categorizer saved in: '+str(model_path))
			
			predictions=model.predict(testX,batch_size=batch_size)

			if len(self.classnames)==2:
				predictions=[round(i[0]) for i in predictions]
				print(classification_report(testY,predictions,target_names=self.classnames))
				report=classification_report(testY,predictions,target_names=self.classnames,output_dict=True)
			else:
				print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=self.classnames))
				report=classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=self.classnames,output_dict=True)

			pd.DataFrame(report).transpose().to_csv(os.path.join(model_path,'training_metrics.csv'),float_format='%.2f')
			if out_path is not None:
				pd.DataFrame(report).transpose().to_excel(os.path.join(out_path,'training_metrics.xlsx'),float_format='%.2f')
			
			plt.style.use('classic')
			plt.figure()
			plt.plot(H.history['loss'],label='train_loss')
			plt.plot(H.history['val_loss'],label='val_loss')
			plt.plot(H.history['accuracy'],label='train_accuracy')
			plt.plot(H.history['val_accuracy'],label='val_accuracy')
			plt.title('Loss and Accuracy')
			plt.xlabel('Epoch')
			plt.ylabel('Loss/Accuracy')
			plt.legend(loc='center right')
			plt.savefig(os.path.join(model_path,'training_history.png'))
			if out_path is not None:
				plt.savefig(os.path.join(out_path,'training_history.png'))
				print('Training reports saved in: '+str(out_path))
			plt.close('all')


	def train_animation_analyzer(self,data_path,model_path,out_path=None,dim=64,channel=1,time_step=18,level=2,aug_methods=[],augvalid=True,background_free=True):

		filters=8

		for i in range(round(dim/60)):
			filters=min(int(filters*2),64)
		
		inputs=Input(shape=(time_step,dim,dim,channel))

		print('Training the Categorizer w/o Pattern Recognizer using the behavior examples in: '+str(data_path))

		files=[i for i in os.listdir(data_path) if i.endswith(self.extension_video)]

		path_files=[]
		labels=[]

		for i in files:
			path_file=os.path.join(data_path,i)
			path_files.append(path_file)
			labels.append(os.path.splitext(i)[0].split('_')[-1])

		labels=np.array(labels)
		lb=LabelBinarizer()
		labels=lb.fit_transform(labels)
		self.classnames=lb.classes_

		if len(list(self.classnames))<2:

			print('You need at least 2 categories of behaviors!')
			print('Training aborted!')

		else:

			print('Found behavior names: '+str(self.classnames))

			if background_free:
				background_code=0
			else:
				background_code=1

			parameters={'classnames':list(self.classnames),'dim_tconv':int(dim),'channel':int(channel),'time_step':int(time_step),'network':1,'level_tconv':int(level),'background_free':int(background_code)}
			pd_parameters=pd.DataFrame.from_dict(parameters)
			pd_parameters.to_csv(os.path.join(model_path,'model_parameters.txt'),index=False)

			(train_files,test_files,y1,y2)=train_test_split(path_files,labels,test_size=0.2,stratify=labels)

			print('Perform augmentation for the behavior examples...')
			print('This might take hours or days, depending on the capacity of your computer.')
			print(datetime.datetime.now())

			print('Start to augment training examples...')
			trainX,_,trainY=self.build_data(train_files,dim_tconv=dim,dim_conv=dim,channel=channel,time_step=time_step,aug_methods=aug_methods,background_free=background_free)
			trainY=lb.fit_transform(trainY)
			print('Start to augment validation examples...')
			if augvalid:
				testX,_,testY=self.build_data(test_files,dim_tconv=dim,dim_conv=dim,channel=channel,time_step=time_step,aug_methods=aug_methods,background_free=background_free)
			else:
				testX,_,testY=self.build_data(test_files,dim_tconv=dim,dim_conv=dim,channel=channel,time_step=time_step,aug_methods=[],background_free=background_free)
			testY=lb.fit_transform(testY)

			with tf.device('CPU'):
				trainX=tf.convert_to_tensor(trainX)
				trainY=tf.convert_to_tensor(trainY)
				testX_tensor=tf.convert_to_tensor(testX)
				testY_tensor=tf.convert_to_tensor(testY)

			print('Training example shape : '+str(trainX.shape))
			print('Training label shape : '+str(trainY.shape))
			print('Validation example shape : '+str(testX.shape))
			print('Validation label shape : '+str(testY.shape))
			print(datetime.datetime.now())

			if trainX.shape[0]<5000:
				batch_size=8
			elif trainX.shape[0]<50000:
				batch_size=16
			else:
				batch_size=32

			if level<5:
				model=self.simple_tvgg(inputs,filters,classes=len(self.classnames),level=level,with_classifier=True)
			else:
				model=self.simple_tresnet(inputs,filters,classes=len(self.classnames),level=level,with_classifier=True)

			if len(self.classnames)==2:
				model.compile(optimizer=SGD(learning_rate=1e-4,momentum=0.9),loss='binary_crossentropy',metrics=['accuracy'])
			else:
				model.compile(optimizer=SGD(learning_rate=1e-4,momentum=0.9),loss='categorical_crossentropy',metrics=['accuracy'])

			cp=ModelCheckpoint(model_path,monitor='val_loss',verbose=1,save_best_only=True,save_weights_only=False,mode='min',save_freq='epoch')
			es=EarlyStopping(monitor='val_loss',min_delta=0.001,mode='min',verbose=1,patience=6,restore_best_weights=True)
			rl=ReduceLROnPlateau(monitor='val_loss',min_delta=0.001,factor=0.2,patience=3,verbose=1,mode='min',min_lr=1e-7)

			H=model.fit(trainX,trainY,batch_size=batch_size,validation_data=(testX_tensor,testY_tensor),epochs=1000000,callbacks=[cp,es,rl])

			model.save(model_path)
			print('Trained Categorizer saved in: '+str(model_path))
			
			predictions=model.predict(testX,batch_size=batch_size)

			if len(self.classnames)==2:
				predictions=[round(i[0]) for i in predictions]
				print(classification_report(testY,predictions,target_names=self.classnames))
				report=classification_report(testY,predictions,target_names=self.classnames,output_dict=True)
			else:
				print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=self.classnames))
				report=classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=self.classnames,output_dict=True)

			pd.DataFrame(report).transpose().to_csv(os.path.join(model_path,'training_metrics.csv'),float_format='%.2f')
			if out_path is not None:
				pd.DataFrame(report).transpose().to_excel(os.path.join(out_path,'training_metrics.xlsx'),float_format='%.2f')

			plt.style.use('classic')
			plt.figure()
			plt.plot(H.history['loss'],label='train_loss')
			plt.plot(H.history['val_loss'],label='val_loss')
			plt.plot(H.history['accuracy'],label='train_accuracy')
			plt.plot(H.history['val_accuracy'],label='val_accuracy')
			plt.title('Loss and Accuracy')
			plt.xlabel('Epoch')
			plt.ylabel('Loss/Accuracy')
			plt.legend(loc='center right')
			plt.savefig(os.path.join(model_path,'training_history.png'))
			if out_path is not None:
				plt.savefig(os.path.join(out_path,'training_history.png'))
				print('Training reports saved in: '+str(out_path))
			plt.close('all')


	def train_combnet(self,data_path,model_path,out_path=None,dim_tconv=32,dim_conv=64,channel=1,time_step=18,level_tconv=1,level_conv=2,aug_methods=[],augvalid=True,background_free=True):

		print('Training Categorizer with both Animation Analyzer and Pattern Recognizer using the behavior examples in: '+str(data_path))

		files=[i for i in os.listdir(data_path) if i.endswith(self.extension_video)]

		path_files=[]
		labels=[]

		for i in files:
			path_file=os.path.join(data_path,i)
			path_files.append(path_file)
			labels.append(os.path.splitext(i)[0].split('_')[-1])

		labels=np.array(labels)
		lb=LabelBinarizer()
		labels=lb.fit_transform(labels)
		self.classnames=lb.classes_

		if len(list(self.classnames))<2:

			print('You need at least 2 categories of behaviors!')
			print('Training aborted!')

		else:

			print('Found behavior names: '+str(self.classnames))

			if background_free:
				background_code=0
			else:
				background_code=1

			parameters={'classnames':list(self.classnames),'dim_tconv':int(dim_tconv),'dim_conv':int(dim_conv),'channel':int(channel),'time_step':int(time_step),'network':2,'level_tconv':int(level_tconv),'level_conv':int(level_conv),'background_free':int(background_code)}
			pd_parameters=pd.DataFrame.from_dict(parameters)
			pd_parameters.to_csv(os.path.join(model_path,'model_parameters.txt'),index=False)

			(train_files,test_files,y1,y2)=train_test_split(path_files,labels,test_size=0.2,stratify=labels)

			print('Perform augmentation for the behavior examples...')
			print('This might take hours or days, depending on the capacity of your computer.')
			print(datetime.datetime.now())

			print('Start to augment training examples...')
			train_animations,train_pattern_images,trainY=self.build_data(train_files,dim_tconv=dim_tconv,dim_conv=dim_conv,channel=channel,time_step=time_step,aug_methods=aug_methods,background_free=background_free)
			trainY=lb.fit_transform(trainY)
			print('Start to augment validation examples...')
			if augvalid:
				test_animations,test_pattern_images,testY=self.build_data(test_files,dim_tconv=dim_tconv,dim_conv=dim_conv,channel=channel,time_step=time_step,aug_methods=aug_methods,background_free=background_free)
			else:
				test_animations,test_pattern_images,testY=self.build_data(test_files,dim_tconv=dim_tconv,dim_conv=dim_conv,channel=channel,time_step=time_step,aug_methods=[],background_free=background_free)
			testY=lb.fit_transform(testY)

			with tf.device('CPU'):
				train_animations=tf.convert_to_tensor(train_animations)
				train_pattern_images=tf.convert_to_tensor(train_pattern_images)
				trainY=tf.convert_to_tensor(trainY)
				test_animations_tensor=tf.convert_to_tensor(test_animations)
				test_pattern_images_tensor=tf.convert_to_tensor(test_pattern_images)
				testY_tensor=tf.convert_to_tensor(testY)

			print('Training example shape : '+str(train_animations.shape)+', '+str(train_pattern_images.shape))
			print('Training label shape : '+str(trainY.shape))
			print('Validation example shape : '+str(test_animations.shape)+', '+str(test_pattern_images.shape))
			print('Validation label shape : '+str(testY.shape))
			print(datetime.datetime.now())

			if train_animations.shape[0]<5000:
				batch_size=8
			elif train_animations.shape[0]<50000:
				batch_size=16
			else:
				batch_size=32

			model=self.combined_network(time_step=time_step,dim_tconv=dim_tconv,dim_conv=dim_conv,channel=channel,classes=len(self.classnames),level_tconv=level_tconv,level_conv=level_conv)
			if len(self.classnames)==2:
				model.compile(optimizer=SGD(learning_rate=1e-4,momentum=0.9),loss='binary_crossentropy',metrics=['accuracy'])
			else:
				model.compile(optimizer=SGD(learning_rate=1e-4,momentum=0.9),loss='categorical_crossentropy',metrics=['accuracy'])

			cp=ModelCheckpoint(model_path,monitor='val_loss',verbose=1,save_best_only=True,save_weights_only=False,mode='min',save_freq='epoch')
			es=EarlyStopping(monitor='val_loss',min_delta=0.001,mode='min',verbose=1,patience=6,restore_best_weights=True)
			rl=ReduceLROnPlateau(monitor='val_loss',min_delta=0.001,factor=0.2,patience=3,verbose=1,mode='min',min_lr=1e-7)

			H=model.fit([train_animations,train_pattern_images],trainY,batch_size=batch_size,validation_data=([test_animations_tensor,test_pattern_images_tensor],testY_tensor),epochs=1000000,callbacks=[cp,es,rl])

			model.save(model_path)
			print('Trained Categorizer saved in: '+str(model_path))

			predictions=model.predict([test_animations,test_pattern_images],batch_size=batch_size)

			if len(self.classnames)==2:
				predictions=[round(i[0]) for i in predictions]
				print(classification_report(testY,predictions,target_names=self.classnames))
				report=classification_report(testY,predictions,target_names=self.classnames,output_dict=True)
			else:
				print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=self.classnames))
				report=classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=self.classnames,output_dict=True)

			pd.DataFrame(report).transpose().to_csv(os.path.join(model_path,'training_metrics.csv'),float_format='%.2f')
			if out_path is not None:
				pd.DataFrame(report).transpose().to_excel(os.path.join(out_path,'training_metrics.xlsx'),float_format='%.2f')

			plt.style.use('classic')
			plt.figure()
			plt.plot(H.history['loss'],label='train_loss')
			plt.plot(H.history['val_loss'],label='val_loss')
			plt.plot(H.history['accuracy'],label='train_accuracy')
			plt.plot(H.history['val_accuracy'],label='val_accuracy')
			plt.title('Loss and Accuracy')
			plt.xlabel('Epoch')
			plt.ylabel('Loss/Accuracy')
			plt.legend(loc='center right')
			plt.savefig(os.path.join(model_path,'training_history.png'))
			if out_path is not None:
				plt.savefig(os.path.join(out_path,'training_history.png'))
				print('Training reports saved in: '+str(out_path))
			plt.close('all')



