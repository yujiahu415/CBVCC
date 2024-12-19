import shutil
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from skimage import exposure




def rename_process(input_path,output_path):

	for folder in os.listdir(input_path):
		if os.path.isdir(os.path.join(input_path,folder)):
			for video in os.listdir(os.path.join(input_path,folder)):
				if video.endswith('.avi'):
					path_to_video=os.path.join(input_path,folder,video)
					capture=cv2.VideoCapture(path_to_video)
					writer=cv2.VideoWriter(os.path.join(output_path,video),cv2.VideoWriter_fourcc(*'MJPG'),6,(100,100),True)
					while True:
						ret,frame=capture.read()
						if frame is None:
							break
						frame=cv2.resize(frame,(100,100),interpolation=cv2.INTER_AREA)
						writer.write(frame)
					writer.release()
					capture.release()


def batch_move(input_path,output_path):

	for folder in os.listdir(input_path):
		if os.path.isdir(os.path.join(input_path,folder)):
			for subfolder in os.listdir(os.path.join(input_path,folder)):
				if os.path.isdir(os.path.join(input_path,folder,subfolder)):
					for example in os.listdir(os.path.join(input_path,folder,subfolder)):
						if example.endswith('.avi'):
							classname=example.split('_')[2]
							if not os.path.isdir(os.path.join(output_path,classname)):
								os.makedirs(os.path.join(output_path,classname),exist_ok=True) 
							animation=os.path.join(output_path,classname,example)
							pattern_image=os.path.join(output_path,classname,example.split('.')[0]+'.jpg')
							shutil.move(os.path.join(input_path,folder,subfolder,example),animation)
							shutil.move(os.path.join(input_path,folder,subfolder,example.split('.')[0]+'.jpg'),pattern_image)


def crop_frame(frame,contours):

	lfbt=np.array([contours[i].min(0) for i in range(len(contours)) if contours[i] is not None]).min(0)[0]
	x_lf=lfbt[0]
	y_bt=lfbt[1]
	rttp=np.array([contours[i].max(0) for i in range(len(contours)) if contours[i] is not None]).max(0)[0]
	x_rt=rttp[0]
	y_tp=rttp[1]

	w=x_rt-x_lf+1
	h=y_tp-y_bt+1

	difference=int(abs(w-h)/2)+1

	if w>h:
		y_bt=max(y_bt-difference-1,0)
		y_tp=min(y_tp+difference+1,frame.shape[0])
		x_lf=max(x_lf-1,0)
		x_rt=min(x_rt+1,frame.shape[1])
	if w<h:
		y_bt=max(y_bt-1,0)
		y_tp=min(y_tp+1,frame.shape[0])
		x_lf=max(x_lf-difference-1,0)
		x_rt=min(x_rt+difference+1,frame.shape[1])

	return (y_bt,y_tp,x_lf,x_rt)


def extract_blob_background(frame,contours,contour=None,channel=1,background_free=False):

	(y_bt,y_tp,x_lf,x_rt)=crop_frame(frame,contours)
	if background_free:
		mask=np.zeros_like(frame)
		cv2.drawContours(mask,[contour],0,(255,255,255),-1)
		masked_frame=frame*(mask/255.0)
	else:
		masked_frame=frame
	blob=masked_frame[y_bt:y_tp,x_lf:x_rt]
	blob=np.uint8(exposure.rescale_intensity(blob,out_range=(0,255)))

	if channel==1:
		blob=cv2.cvtColor(blob,cv2.COLOR_BGR2GRAY)
		blob=img_to_array(blob)

	return blob


def generate_patternimage(frame,outlines):

	background_outlines=np.zeros_like(frame)

	(y_bt,y_tp,x_lf,x_rt)=crop_frame(frame,outlines)

	length=len(outlines)
	p_size=int(max(abs(y_bt-y_tp),abs(x_lf-x_rt))/150+1)

	for n,outline in enumerate(outlines):

		if outline is not None:

			if n<length/4:
				d=n*int((255*4/length))
				cv2.drawContours(background_outlines,[outline],0,(255,d,0),p_size)
			elif n<length/2:
				d=int((n-length/4)*(255*4/length))
				cv2.drawContours(background_outlines,[outline],0,(255,255,d),p_size)
			elif n<3*length/4:
				d=int((n-length/2)*(255*4/length))
				cv2.drawContours(background_outlines,[outline],0,(255,255-d,255),p_size)
			else:
				d=int((n-3*length/4)*(255*4/length))
				cv2.drawContours(background_outlines,[outline],0,(255-d,0,255),p_size)

	pattern_image=background_outlines[y_bt:y_tp,x_lf:x_rt]

	return pattern_image


def extract_frames(path_to_videos,out_path,framewidth=None,start_t=0,duration=0,skip_redundant=1000):

	for video in os.listdir(path_to_videos):

		if video.endswith('.avi'):

			capture=cv2.VideoCapture(os.path.join(path_to_videos,video))
			fps=round(capture.get(cv2.CAP_PROP_FPS))
			full_duration=capture.get(cv2.CAP_PROP_FRAME_COUNT)/fps
			video_name=os.path.splitext(video)[0]

			if start_t>=full_duration:
				print('The beginning time is later than the end of the video!')
				print('Will use the beginning of the video as the beginning time!')
				start_t=0
			if duration<=0:
				duration=full_duration
			end_t=start_t+duration
				
			frame_count=1
			frame_count_generate=0

			while True:

				retval,frame=capture.read()
				t=(frame_count)/fps

				if frame is None:
					break

				if t>=end_t:
					break

				if t>=start_t:
					
					if frame_count_generate%skip_redundant==0:

						if framewidth is not None:
							frameheight=int(frame.shape[0]*framewidth/frame.shape[1])
							frame=cv2.resize(frame,(framewidth,frameheight),interpolation=cv2.INTER_AREA)

						cv2.imwrite(os.path.join(out_path,video_name+'_'+str(frame_count_generate)+'.jpg'),frame)

					frame_count_generate+=1

				frame_count+=1

			capture.release()

			print('The image examples stored in: '+out_path)


