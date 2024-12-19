import json
import pandas as pd
from detector import Detector
from categorizer import Categorizers
from analyzer import AnalyzecellDetector
from tools import *




# move all videos to the same folder, and resize frame to 100 X 100, fps to 6

input_path='path_to_testing_videos'
output_path='path_to_processed_testing_videos'

rename_process(input_path,output_path)




# extract frames from training videos for annotating cells in the images (using Roboflow: https://roboflow.com/)

path_to_videos='path_to_training_videos'
out_path='path_to_extracted_images'

extract_frames(path_to_videos,out_path,framewidth=None,start_t=0,duration=0,skip_redundant=2)




# after annotation of cells in images, train a Detector to detect cells in images

path_to_annotation='path_to_the_annotations.coco.json'
path_to_trainingimages='path_to_training_images'
path_to_detector='path_to_Detector'
iteration_num=30000
inference_size=100

DT=Detector()
DT.train(path_to_annotation,path_to_trainingimages,path_to_detector,iteration_num,inference_size)




# use a trained Detector to generate unsorted behavior examples from training videos

path_to_detector='path_to_Detector'
path_to_training_videos='path_to_training_videos'
result_path='path_to_generated_behavior_examples'

with open(os.path.join(path_to_detector,'model_parameters.txt')) as f:
	model_parameters=f.read()
cell_kinds=json.loads(model_parameters)['cell_names']
detector_batch=1

cell_number={}
for cell_name in cell_kinds:
	cell_number[cell_name]=5

framewidth=100
t=0.0
duration=0
channel=3
length=10
background_free=True


for i in os.listdir(path_to_training_videos):

	if i.endswith('.avi'):

		path_to_video=os.path.join(path_to_training_videos,i)
		basename=os.path.splitext(os.path.basename(i))[0]

		ACD=AnalyzecellDetector()
		ACD.prepare_analysis(path_to_detector,path_to_video,result_path,cell_number,cell_kinds,names_and_colors={},framewidth=framewidth,channel=channel,animation_analyzer=False,t=t,duration=duration,length=length)
		ACD.acquire_information(batch_size=detector_batch,background_free=background_free)
		ACD.generate_data(background_free=background_free,skip_redundant=1)




# prepare the sorted behavior examples into a folder (path_to_prepared_examples) to store all the training examples

path_to_sorted_examples='path_to_sorted_examples'
path_to_prepared_examples='path_to_prepared_examples'

CA=Categorizers()
CA.rename_label(path_to_sorted_examples,path_to_prepared_examples,resize=None)




# use the prepared training examples to train a Categorizer for identifying user-defined cell behaviors

path_to_prepared_examples='path_to_prepared_examples'
path_to_categorizer='path_to_categorizer'

CA=Categorizers()
CA.train_combnet(path_to_prepared_examples,path,out_path=None,dim_tconv=32,dim_conv=64,channel=1,time_step=10,level_tconv=2,level_conv=4,aug_methods=['all'],augvalid=True,background_free=True)




# output the predictions on the testing videos using the trained Detector and Categorizer

path_to_testing_videos='path_to_testing_videos'
result_path='path_to_the_analysis_outputs'

path_to_detector='path_to_detector'
with open(os.path.join(path_to_detector,'model_parameters.txt')) as f:
	model_parameters=f.read()
cell_kinds=json.loads(model_parameters)['cell_names']
detector_batch=1

cell_number={}
for cell_name in cell_kinds:
	cell_number[cell_name]=5

path_to_categorizer='path_to_categorizer'
parameters=pd.read_csv(os.path.join(path_to_categorizer,'model_parameters.txt'))
if 'dim_conv' in parameters:
	dim_conv=int(parameters['dim_conv'][0])
if 'dim_tconv' in parameters:
	dim_tconv=int(parameters['dim_tconv'][0])
else:
	dim_tconv=8
channel=int(parameters['channel'][0])
length=int(parameters['time_step'][0])
categorizer_type=int(parameters['network'][0])
if categorizer_type==2:
	animation_analyzer=True
else:
	animation_analyzer=False
if int(parameters['background_free'][0])==0:
	background_free=True
else:
	background_free=False

behaviornames_and_colors={'inplace':['#ffffff','#ffff00'],'linear':['#ffffff','#ffcc00'],'orient':['#ffffff','#ff00ff']}
framewidth=100
t=0.0
duration=0
uncertain=0
filter_center_cell=True

all_scores={}

for i in os.listdir(path_to_testing_videos):

	if i.endswith('.avi'):

		path_to_video=os.path.join(path_to_testing_videos,i)
		basename=os.path.basename(i)

		ACD=AnalyzecellDetector()
		ACD.prepare_analysis(path_to_detector,path_to_video,result_path,cell_number,cell_kinds,names_and_colors=behaviornames_and_colors,framewidth=framewidth,dim_tconv=dim_tconv,dim_conv=dim_conv,channel=channel,animation_analyzer=animation_analyzer,t=t,duration=duration,length=length)
		ACD.acquire_information(batch_size=detector_batch,background_free=background_free)
		ACD.craft_data(filter_center_cell=filter_center_cell)
		ACD.categorize_behaviors(path_to_categorizer,uncertain=uncertain)
		ACD.annotate_video(cell_kinds,list(behaviornames_and_colors.keys()))
		ACD.analyze_parameters()

		all_scores[basename]=ACD.final_score

all_scores_df=pd.DataFrame.from_dict(all_scores,orient='index',columns=['Probability'])
all_scores_df.to_csv(os.path.join(result_path,'All_probability.csv'),index_label='ID',float_format='%.2f')


