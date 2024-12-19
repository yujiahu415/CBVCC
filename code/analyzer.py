import os
import cv2
import gc
import torch
import json
import datetime
import numpy as np
from scipy.spatial import distance
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pandas as pd
from detector import Detector
from tools import *
import math




class AnalyzecellDetector():

	def __init__(self):

		self.detector=None
		self.cell_mapping=None
		self.path_to_video=None
		self.basename=None
		self.fps=None
		self.framewidth=None
		self.frameheight=None
		self.kernel=3
		self.results_path=None
		self.dim_tconv=8
		self.dim_conv=8
		self.channel=1
		self.animation_analyzer=True
		self.cell_number=None
		self.cell_kinds=None
		self.t=0
		self.duration=5
		self.length=None
		self.background=None
		self.skipped_frames=[]
		self.all_time=[]
		self.total_analysis_framecount=None
		self.to_deregister={}
		self.count_to_deregister=None
		self.register_counts={}
		self.cell_contours={}
		self.cell_centers={}
		self.cell_existingcenters={}
		self.animations={}
		self.pattern_images={}
		self.event_probability={}
		self.all_behavior_parameters={}
		self.final_score=0
		self.temp_frames=None


	def prepare_analysis(self,path_to_detector,path_to_video,results_path,cell_number,cell_kinds,names_and_colors=None,framewidth=None,dim_tconv=8,dim_conv=8,channel=1,animation_analyzer=True,t=0,duration=5,length=15):
		
		print('Preparation started...')
		print(datetime.datetime.now())

		self.detector=Detector()
		self.detector.load(path_to_detector,cell_kinds)
		self.cell_mapping=self.detector.cell_mapping
		self.path_to_video=path_to_video
		self.basename=os.path.basename(self.path_to_video)
		self.framewidth=framewidth
		self.results_path=os.path.join(results_path,os.path.splitext(self.basename)[0])
		self.cell_number=cell_number
		self.cell_kinds=cell_kinds
		self.dim_tconv=dim_tconv
		self.dim_conv=dim_conv
		self.channel=channel
		self.animation_analyzer=animation_analyzer
		self.t=t
		self.duration=duration
		self.length=length
		os.makedirs(self.results_path,exist_ok=True)
		capture=cv2.VideoCapture(self.path_to_video)
		self.fps=round(capture.get(cv2.CAP_PROP_FPS))
		self.count_to_deregister=self.fps*2

		if self.duration<=0:
			self.total_analysis_framecount=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))+1
		else:
			self.total_analysis_framecount=int(self.fps*self.duration)+1

		while True:
			retval,frame=capture.read()
			break
		capture.release()

		print('Video fps: '+str(self.fps))
		print('The original video framesize: '+str(int(frame.shape[0]))+' X '+str(int(frame.shape[1])))

		if self.framewidth is not None:
			self.frameheight=int(frame.shape[0]*self.framewidth/frame.shape[1])
			self.background=cv2.resize(frame,(self.framewidth,self.frameheight),interpolation=cv2.INTER_AREA)
			print('The resized video framesize: '+str(self.frameheight)+' X '+str(self.framewidth))
		else:
			self.background=frame
		self.temp_frames=deque(maxlen=self.length)
		framesize=min(self.background.shape[0],self.background.shape[1])

		total_number=0
		for cell_name in self.cell_kinds:
			total_number+=self.cell_number[cell_name]
			self.event_probability[cell_name]={}
			self.all_behavior_parameters[cell_name]={}
			for behavior_name in names_and_colors:
				self.all_behavior_parameters[cell_name][behavior_name]={}
				self.all_behavior_parameters[cell_name][behavior_name]['color']=names_and_colors[behavior_name]
			self.to_deregister[cell_name]={}
			self.register_counts[cell_name]={}
			self.cell_contours[cell_name]={}
			self.cell_centers[cell_name]={}
			self.cell_existingcenters[cell_name]={}
			self.pattern_images[cell_name]={}
			if self.animation_analyzer:
				self.animations[cell_name]={}
			for i in range(self.cell_number[cell_name]):
				self.to_deregister[cell_name][i]=0
				self.register_counts[cell_name][i]=None
				self.cell_contours[cell_name][i]=[None]*self.total_analysis_framecount
				self.cell_centers[cell_name][i]=[None]*self.total_analysis_framecount
				self.cell_existingcenters[cell_name][i]=(-10000,-10000)
				if self.animation_analyzer:
					self.animations[cell_name][i]=[np.zeros((self.length,self.dim_tconv,self.dim_tconv,self.channel),dtype='uint8')]*self.total_analysis_framecount
				self.pattern_images[cell_name][i]=[np.zeros((self.dim_conv,self.dim_conv,3),dtype='uint8')]*self.total_analysis_framecount

		if framesize/total_number<250:
			self.kernel=3
		elif framesize/total_number<500:
			self.kernel=5
		elif framesize/total_number<1000:
			self.kernel=7
		elif framesize/total_number<1500:
			self.kernel=9
		else:
			self.kernel=11

		print('Preparation completed!')


	def track_cell(self,frame_count_analyze,cell_name,contours,centers):

		unused_existing_indices=list(self.cell_existingcenters[cell_name])
		existing_centers=list(self.cell_existingcenters[cell_name].values())
		unused_new_indices=list(range(len(centers)))
		dt_flattened=distance.cdist(existing_centers,centers).flatten()
		dt_sort_index=dt_flattened.argsort()
		length=len(centers)

		for idx in dt_sort_index:
			index_in_existing=int(idx/length)
			index_in_new=int(idx%length)
			if index_in_existing in unused_existing_indices:
				if index_in_new in unused_new_indices:
					unused_existing_indices.remove(index_in_existing)
					unused_new_indices.remove(index_in_new)
					if self.register_counts[cell_name][index_in_existing] is None:
						self.register_counts[cell_name][index_in_existing]=frame_count_analyze
					self.to_deregister[cell_name][index_in_existing]=0
					self.cell_contours[cell_name][index_in_existing][frame_count_analyze]=contours[index_in_new]
					center=centers[index_in_new]
					self.cell_centers[cell_name][index_in_existing][frame_count_analyze]=center
					self.cell_existingcenters[cell_name][index_in_existing]=center
					pattern_image=generate_patternimage(self.background,self.cell_contours[cell_name][index_in_existing][max(0,(frame_count_analyze-self.length+1)):frame_count_analyze+1])
					pattern_image=cv2.resize(pattern_image,(self.dim_conv,self.dim_conv),interpolation=cv2.INTER_AREA)
					self.pattern_images[cell_name][index_in_existing][frame_count_analyze]=np.array(pattern_image)

		if len(unused_existing_indices)>0:
			for i in unused_existing_indices:
				if self.to_deregister[cell_name][i]<=self.count_to_deregister:
					self.to_deregister[cell_name][i]+=1
				else:
					self.cell_existingcenters[cell_name][i]=(-10000,-10000)


	def detect_track_individuals(self,frames,batch_size,frame_count_analyze,background_free=True,animation=None):

		tensor_frames=[torch.as_tensor(frame.astype('float32').transpose(2,0,1)) for frame in frames]
		inputs=[{'image':tensor_frame} for tensor_frame in tensor_frames]

		outputs=self.detector.inference(inputs)

		for batch_count,output in enumerate(outputs):

			frame=frames[batch_count]
			self.temp_frames.append(frame)
			instances=outputs[batch_count]['instances'].to('cpu')
			masks=instances.pred_masks.numpy().astype(np.uint8)
			classes=instances.pred_classes.numpy()
			scores=instances.scores.numpy()

			if len(masks)==0:

				self.skipped_frames.append(frame_count_analyze+1-batch_size+batch_count)

			else:

				mask_area=np.sum(np.array(masks),axis=(1,2))
				exclusion_mask=np.zeros(len(masks),dtype=bool)
				exclusion_mask[np.where((np.sum(np.logical_and(masks[:,None],masks),axis=(2,3))/mask_area[:,None]>0.8) & (mask_area[:,None]<mask_area[None,:]))[0]]=True
				masks=[m for m,exclude in zip(masks,exclusion_mask) if not exclude]
				classes=[c for c,exclude in zip(classes,exclusion_mask) if not exclude]
				classes=[self.cell_mapping[str(x)] for x in classes]
				scores=[s for s,exclude in zip(scores,exclusion_mask) if not exclude]

				for cell_name in self.cell_kinds:

					contours=[]
					centers=[]
					goodcontours=[]
					goodmasks=[]

					cell_number=int(self.cell_number[cell_name])
					cell_masks=[masks[a] for a,name in enumerate(classes) if name==cell_name]
					cell_scores=[scores[a] for a,name in enumerate(classes) if name==cell_name]

					if len(cell_masks)>0:

						if len(cell_scores)>cell_number*2:
							sorted_scores_indices=np.argsort(cell_scores)[-int(cell_number*2):]
							cell_masks=[cell_masks[x] for x in sorted_scores_indices]
						for mask in cell_masks:
							mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((self.kernel,self.kernel),np.uint8))
							goodmasks.append(mask)
							cnts,_=cv2.findContours((mask*255).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
							goodcontours.append(sorted(cnts,key=cv2.contourArea,reverse=True)[0])
						areas=[cv2.contourArea(ct) for ct in goodcontours]
						sorted_area_indices=np.argsort(np.array(areas))[-cell_number:]

						for x in sorted_area_indices:
							mask=goodmasks[x]
							cnt=goodcontours[x]
							contours.append(cnt)
							centers.append((int(cv2.moments(cnt)['m10']/cv2.moments(cnt)['m00']),int(cv2.moments(cnt)['m01']/cv2.moments(cnt)['m00'])))  
							(_,_),(w,h),_=cv2.minAreaRect(cnt)

						self.track_cell(frame_count_analyze+1-batch_size+batch_count,cell_name,contours,centers)

						if self.animation_analyzer:
							for i in self.cell_centers[cell_name]:
								for n,f in enumerate(self.temp_frames):
									contour=self.cell_contours[cell_name][i][max(0,frame_count_analyze+1-batch_size+batch_count-self.length+1):frame_count_analyze+1-batch_size+batch_count+1][n]
									if contour is None:
										blob=np.zeros((self.dim_tconv,self.dim_tconv,self.channel),dtype='uint8')
									else:
										blob=extract_blob_background(f,self.cell_contours[cell_name][i][max(0,frame_count_analyze+1-batch_size+batch_count-self.length+1):frame_count_analyze+1-batch_size+batch_count+1],contour=contour,channel=self.channel,background_free=background_free)
										blob=cv2.resize(blob,(self.dim_tconv,self.dim_tconv),interpolation=cv2.INTER_AREA)
									animation.append(img_to_array(blob))
								self.animations[cell_name][i][frame_count_analyze+1-batch_size+batch_count]=np.array(animation)


	def acquire_information(self,batch_size=1,background_free=True):

		print('Acquiring information in each frame...')
		print(datetime.datetime.now())

		capture=cv2.VideoCapture(self.path_to_video)
		batch=[]
		batch_count=frame_count=frame_count_analyze=0
		animation=deque([np.zeros((self.dim_tconv,self.dim_tconv,self.channel),dtype='uint8')],maxlen=self.length)*self.length

		start_t=round((self.t-self.length/self.fps),2)
		if start_t<0:
			start_t=0.00
		if self.duration==0:
			end_t=float('inf')
		else:
			end_t=start_t+self.duration

		while True:

			retval,frame=capture.read()
			time=round((frame_count+1)/self.fps,2)

			if time>end_t or frame is None:
				break

			if time>=start_t:

				self.all_time.append(round((time-start_t),2))
				
				if (frame_count_analyze+1)%1000==0:
					print(str(frame_count_analyze+1)+' frames processed...')
					print(datetime.datetime.now())

				if self.framewidth is not None:
					frame=cv2.resize(frame,(self.framewidth,self.frameheight),interpolation=cv2.INTER_AREA)

				batch.append(frame)
				batch_count+=1

				if batch_count==batch_size:
					batch_count=0
					self.detect_track_individuals(batch,batch_size,frame_count_analyze,background_free=background_free,animation=animation)
					batch=[]

				frame_count_analyze+=1

			frame_count+=1

		capture.release()

		print('Information acquisition completed!')


	def craft_data(self,filter_center_cell=False):

		print('Crafting data...')
		print(datetime.datetime.now())

		for cell_name in self.cell_kinds:

			length=len(self.all_time)
			IDs=list(self.cell_centers[cell_name].keys())
			ID_tokeep=None

			if filter_center_cell:

				if len(IDs)>1:

					distances={}

					for i in IDs:
						dts=[100000]
						for c in self.cell_centers[cell_name][i]:
							if c is not None:
								dts.append(math.dist(c,(50,50)))
						distances[i]=min(dts)

					dt_min=100000
					for i in distances:
						if distances[i]<dt_min:
							ID_tokeep=i
							dt_min=distances[i]
					
					for i in IDs:
						if ID_tokeep is not None:
							if i!=ID_tokeep:
								del self.to_deregister[cell_name][i]
								del self.register_counts[cell_name][i]
								del self.cell_centers[cell_name][i]
								del self.cell_existingcenters[cell_name][i]
								del self.cell_contours[cell_name][i]
								if self.animation_analyzer:
									del self.animations[cell_name][i]
								del self.pattern_images[cell_name][i]

			for i in self.cell_centers[cell_name]:
				self.cell_centers[cell_name][i]=self.cell_centers[cell_name][i][:length]
				self.cell_contours[cell_name][i]=self.cell_contours[cell_name][i][:length]
				if self.animation_analyzer:
					self.animations[cell_name][i]=self.animations[cell_name][i][:length]
				self.pattern_images[cell_name][i]=self.pattern_images[cell_name][i][:length]

		print('Data crafting completed!')


	def categorize_behaviors(self,path_to_categorizer,uncertain=0):

		print('Categorizing behaviors...')
		print(datetime.datetime.now())

		categorizer=load_model(path_to_categorizer)

		for cell_name in self.cell_kinds:

			IDs=list(self.pattern_images[cell_name].keys())

			if len(IDs)>0:

				if self.animation_analyzer:
					animations=self.animations[cell_name][IDs[0]]
				pattern_images=self.pattern_images[cell_name][IDs[0]]

				if len(self.pattern_images[cell_name])>1:
					for n in IDs[1:]:
						if self.animation_analyzer:
							animations+=self.animations[cell_name][n]
						pattern_images+=self.pattern_images[cell_name][n]

				if self.animation_analyzer:
					del self.animations[cell_name]
				del self.pattern_images[cell_name]
				gc.collect()

				with tf.device('CPU'):
					if self.animation_analyzer:
						animations=tf.convert_to_tensor(np.array(animations,dtype='float32')/255.0)
					pattern_images=tf.convert_to_tensor(np.array(pattern_images,dtype='float32')/255.0)

				if self.animation_analyzer:
					inputs=[animations,pattern_images]
				else:
					inputs=pattern_images

				predictions=categorizer.predict(inputs,batch_size=32)

				for i in IDs:
					self.event_probability[cell_name][i]=[['NA',-1]]*len(self.all_time)

				idx=0
				for n in IDs:
					i=int(self.length/2)
					idx+=i
					while i<len(self.cell_centers[cell_name][n]):
						prediction=predictions[idx]
						behavior_names=list(self.all_behavior_parameters[cell_name].keys())
						if len(behavior_names)==2:
							if prediction[0]>0.5:
								if prediction[0]-(1-prediction[0])>uncertain:
									self.event_probability[cell_name][n][i]=[behavior_names[1],prediction[0]]
							if prediction[0]<0.5:
								if (1-prediction[0])-prediction[0]>uncertain:
									self.event_probability[cell_name][n][i]=[behavior_names[0],1-prediction[0]]
						else:
							if sorted(prediction)[-1]-sorted(prediction)[-2]>uncertain:
								self.event_probability[cell_name][n][i]=[behavior_names[np.argmax(prediction)],max(prediction)]
						idx+=1
						i+=1

				del predictions
				gc.collect()

		print('Behavioral categorization completed!')


	def annotate_video(self,cell_to_include,behavior_to_include,show_legend=False):

		print('Annotating video...')
		print(datetime.datetime.now())

		text_scl=max(0.5,round((self.background.shape[0]+self.background.shape[1])/1080,1))
		text_tk=max(1,round((self.background.shape[0]+self.background.shape[1])/540))
		background=np.zeros_like(self.background)
		if self.framewidth is not None:
			background=cv2.resize(background,(self.framewidth,self.frameheight),interpolation=cv2.INTER_AREA)

		colors={}
		for behavior_name in self.all_behavior_parameters[self.cell_kinds[0]]:
			if self.all_behavior_parameters[self.cell_kinds[0]][behavior_name]['color'][1][0]!='#':
				colors[behavior_name]=(255,255,255)
			else:
				hex_color=self.all_behavior_parameters[self.cell_kinds[0]][behavior_name]['color'][1].lstrip('#')
				color=tuple(int(hex_color[i:i+2],16) for i in (0,2,4))
				colors[behavior_name]=color[::-1]
		
		if len(behavior_to_include)!=len(self.all_behavior_parameters[self.cell_kinds[0]]):
			for behavior_name in self.all_behavior_parameters[self.cell_kinds[0]]:
				if behavior_name not in behavior_to_include:
					del colors[behavior_name]
		
		if show_legend:	
			scl=self.background.shape[0]/1024
			if 25*(len(colors)+1)<self.background.shape[0]:
				intvl=25
			else:
				intvl=int(self.background.shape[0]/(len(colors)+1))

		capture=cv2.VideoCapture(self.path_to_video)
		writer=None
		frame_count=frame_count_analyze=0

		total_cell_number=0
		for cell_name in self.cell_kinds:
			df=pd.DataFrame(self.cell_centers[cell_name],index=self.all_time)
			df.to_excel(os.path.join(self.results_path,cell_name+'_'+'all_centers.xlsx'),index_label='time/ID')
			for i in self.cell_centers[cell_name]:
				total_cell_number+=1
		if total_cell_number<=0:
			total_cell_number=1
		color_diff=int(510/total_cell_number)

		start_t=round((self.t-self.length/self.fps),2)
		if start_t<0:
			start_t=0.00
		if self.duration==0:
			end_t=float('inf')
		else:
			end_t=start_t+self.duration

		while True:

			retval,frame=capture.read()
			time=round((frame_count+1)/self.fps,2)

			if time>=end_t or frame is None:
				break

			if time>=start_t:

				if self.framewidth is not None:
					frame=cv2.resize(frame,(self.framewidth,self.frameheight),interpolation=cv2.INTER_AREA)

				if show_legend:
					n=1
					for i in colors:
						cv2.putText(frame,i,(10,intvl*n),cv2.FONT_HERSHEY_SIMPLEX,scl,colors[i],text_tk)
						n+=1

				current_cell_number=0

				if frame_count_analyze not in self.skipped_frames:

					for cell_name in cell_to_include:

						for i in self.cell_contours[cell_name]:

							if frame_count_analyze<len(self.cell_contours[cell_name][i]):

								if self.cell_contours[cell_name][i][frame_count_analyze] is not None:

									cx=self.cell_centers[cell_name][i][frame_count_analyze][0]
									cy=self.cell_centers[cell_name][i][frame_count_analyze][1]

									if self.cell_centers[cell_name][i][max(frame_count_analyze-1,0)] is not None:
										cxp=self.cell_centers[cell_name][i][max(frame_count_analyze-1,0)][0]
										cyp=self.cell_centers[cell_name][i][max(frame_count_analyze-1,0)][1]
										cv2.line(self.background,(cx,cy),(cxp,cyp),(abs(int(color_diff*(total_cell_number-current_cell_number)-255)),int(color_diff*current_cell_number/2),int(color_diff*(total_cell_number-current_cell_number)/2)),int(text_tk))
										cv2.line(background,(cx,cy),(cxp,cyp),(abs(int(color_diff*(total_cell_number-current_cell_number)-255)),int(color_diff*current_cell_number/2),int(color_diff*(total_cell_number-current_cell_number)/2)),int(text_tk))
									else:
										cv2.circle(self.background,(cx,cy),int(text_tk),(abs(int(color_diff*(total_cell_number-current_cell_number)-255)),int(color_diff*current_cell_number/2),int(color_diff*(total_cell_number-current_cell_number)/2)),-1)
										cv2.circle(background,(cx,cy),int(text_tk),(abs(int(color_diff*(total_cell_number-current_cell_number)-255)),int(color_diff*current_cell_number/2),int(color_diff*(total_cell_number-current_cell_number)/2)),-1)

									cv2.circle(frame,(cx,cy),int(text_tk*3),(255,0,0),-1)
									cv2.putText(frame,cell_name+' '+str(i),(cx-10,cy-25),cv2.FONT_HERSHEY_SIMPLEX,text_scl,(255,255,255),text_tk)

									if self.event_probability[cell_name][i][frame_count_analyze][0]=='NA':
										cv2.drawContours(frame,[self.cell_contours[cell_name][i][frame_count_analyze]],0,(255,255,255),1)
										cv2.putText(frame,'NA',(cx-10,cy-10),cv2.FONT_HERSHEY_SIMPLEX,text_scl,(255,255,255),text_tk)
									else:
										name=self.event_probability[cell_name][i][frame_count_analyze][0]
										probability=str(round(self.event_probability[cell_name][i][frame_count_analyze][1]*100))+'%'
										if name in colors:
											color=colors[self.event_probability[cell_name][i][frame_count_analyze][0]]
											cv2.drawContours(frame,[self.cell_contours[cell_name][i][frame_count_analyze]],0,color,1)
											cv2.putText(frame,name+' '+probability,(cx-10,cy-10),cv2.FONT_HERSHEY_SIMPLEX,text_scl,color,text_tk)
										else:
											cv2.drawContours(frame,[self.cell_contours[cell_name][i][frame_count_analyze]],0,(255,255,255),1)
											cv2.putText(frame,'NA',(cx-10,cy-10),cv2.FONT_HERSHEY_SIMPLEX,text_scl,(255,255,255),text_tk)

							current_cell_number+=1

				if writer is None:
					(h,w)=frame.shape[:2]
					writer=cv2.VideoWriter(os.path.join(self.results_path,'Annotated video.avi'),cv2.VideoWriter_fourcc(*'MJPG'),self.fps,(w,h),True)

				writer.write(frame)

				frame_count_analyze+=1

			frame_count+=1

		capture.release()
		writer.release()

		cv2.imwrite(os.path.join(self.results_path,'Trajectory_background.jpg'),self.background)
		cv2.imwrite(os.path.join(self.results_path,'Trajectory_black.jpg'),background)

		print('Video annotation completed!')


	def analyze_parameters(self):

		print('Quantifying scores...')
		print(datetime.datetime.now())

		for cell_name in self.cell_kinds:

			events_df=pd.DataFrame(self.event_probability[cell_name],index=self.all_time)
			events_df.to_excel(os.path.join(self.results_path,cell_name+'_all_event_probability.xlsx'),float_format='%.2f',index_label='time/ID')

			for i in self.cell_centers[cell_name]:

				n=0
				events=[]

				while n<len(self.cell_centers[cell_name][i]):
					events.append(self.event_probability[cell_name][i][n][0])
					n+=1

				effective_events=events[14:17]
				delta=0.5/len(effective_events)
				score=0.5

				for behavior_name in events:
					if behavior_name=='linear':
						score-=delta
					elif behavior_name=='inplace':
						score-=2*delta
					elif behavior_name=='orient':
						score+=4*delta

				if score<0:
					score=0
				if score>1:
					score=1
				if score==0.5:
					score=0.49

				self.final_score=score

		print('Scores quantification Completed!')
		print(datetime.datetime.now())


	def generate_data(self,background_free=True,skip_redundant=1):
		
		print('Generating behavior examples...')
		print(datetime.datetime.now())

		capture=cv2.VideoCapture(self.path_to_video)
		frame_count=frame_count_analyze=0
		animation=deque(maxlen=self.length)
		for cell_name in self.cell_kinds:
			for i in range(self.cell_number[cell_name]):
				os.makedirs(os.path.join(self.results_path,str(cell_name)+'_'+str(i)),exist_ok=True)

		start_t=round((self.t-self.length/self.fps),2)
		if start_t<0:
			start_t=0.00
		if self.duration==0:
			end_t=float('inf')
		else:
			end_t=start_t+self.duration

		while True:

			retval,frame=capture.read()
			time=round((frame_count+1)/self.fps,2)

			if time>=end_t or frame is None:
				break

			if time>=start_t:

				if self.framewidth is not None:
					frame=cv2.resize(frame,(self.framewidth,self.frameheight),interpolation=cv2.INTER_AREA)

				self.detect_track_individuals([frame],1,frame_count_analyze,background_free=background_free,animation=animation)

				for cell_name in self.cell_kinds:
						
					if frame_count_analyze>=self.length and frame_count_analyze%skip_redundant==0:

						for n in self.cell_centers[cell_name]:

							h=w=0

							for i,f in enumerate(self.temp_frames):
								contour=self.cell_contours[cell_name][n][frame_count_analyze-self.length+1:frame_count_analyze+1][i]
								if contour is None:
									blob=np.zeros_like(self.background)
								else:
									blob=extract_blob_background(f,self.cell_contours[cell_name][n][frame_count_analyze-self.length+1:frame_count_analyze+1],contour=contour,channel=3,background_free=background_free)
									h,w=blob.shape[:2]
								animation.append(blob)

							if h>0:

								animation_name=os.path.splitext(self.basename)[0]+'_'+cell_name+'_'+str(n)+'_'+str(frame_count_analyze)+'_len'+str(self.length)+'.avi'
								pattern_image_name=os.path.splitext(self.basename)[0]+'_'+cell_name+'_'+str(n)+'_'+str(frame_count_analyze)+'_len'+str(self.length)+'.jpg'
								pattern_image=generate_patternimage(self.background,self.cell_contours[cell_name][n][frame_count_analyze-self.length+1:frame_count_analyze+1])

								path_animation=os.path.join(self.results_path,str(cell_name)+'_'+str(n),animation_name)
								path_pattern_image=os.path.join(self.results_path,str(cell_name)+'_'+str(n),pattern_image_name)

								writer=cv2.VideoWriter(path_animation,cv2.VideoWriter_fourcc(*'MJPG'),self.fps/5,(w,h),True)
								for blob in animation:
									writer.write(cv2.resize(blob,(w,h),interpolation=cv2.INTER_AREA))
								writer.release()

								cv2.imwrite(path_pattern_image,pattern_image)

				frame_count_analyze+=1

			frame_count+=1

		capture.release()

		print('Behavior example generation completed!')



