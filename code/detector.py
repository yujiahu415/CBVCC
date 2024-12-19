import os
import json
import torch
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog,DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer,DefaultPredictor
from detectron2.modeling import build_model



class Detector():

	def __init__(self):

		self.device='cuda' if torch.cuda.is_available() else 'cpu'
		self.cell_mapping=None
		self.current_detector=None


	def train(self,path_to_annotation,path_to_trainingimages,path_to_detector,iteration_num,inference_size):

		if str('Cellan_detector_train') in DatasetCatalog.list():
			DatasetCatalog.remove('Cellan_detector_train')
			MetadataCatalog.remove('Cellan_detector_train')

		register_coco_instances('Cellan_detector_train',{},path_to_annotation,path_to_trainingimages)

		datasetcat=DatasetCatalog.get('Cellan_detector_train')
		metadatacat=MetadataCatalog.get('Cellan_detector_train')

		classnames=metadatacat.thing_classes

		model_parameters_dict={}
		model_parameters_dict['cell_names']=[]

		annotation_data=json.load(open(path_to_annotation))

		for i in annotation_data['categories']:
			if i['id']>0:
				model_parameters_dict['cell_names'].append(i['name'])

		print('Cell names in annotation file: '+str(model_parameters_dict['cell_names']))

		cfg=get_cfg()
		cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
		cfg.OUTPUT_DIR=path_to_detector
		cfg.DATASETS.TRAIN=('Cellan_detector_train',)
		cfg.DATASETS.TEST=()
		cfg.DATALOADER.NUM_WORKERS=0
		cfg.MODEL.WEIGHTS=model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
		cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE=128
		cfg.MODEL.ROI_HEADS.NUM_CLASSES=int(len(classnames))
		cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.5
		cfg.SOLVER.MAX_ITER=int(iteration_num)
		cfg.SOLVER.BASE_LR=0.001
		cfg.SOLVER.WARMUP_ITERS=int(iteration_num*0.1)
		cfg.SOLVER.STEPS=(int(iteration_num*0.4),int(iteration_num*0.8))
		cfg.SOLVER.GAMMA=0.5
		cfg.SOLVER.IMS_PER_BATCH=4
		cfg.MODEL.DEVICE=self.device
		cfg.SOLVER.CHECKPOINT_PERIOD=100000000000000000
		cfg.INPUT.MIN_SIZE_TEST=int(inference_size)
		cfg.INPUT.MAX_SIZE_TEST=int(inference_size)
		cfg.INPUT.MIN_SIZE_TRAIN=(int(inference_size),)
		cfg.INPUT.MAX_SIZE_TRAIN=int(inference_size)
		os.makedirs(cfg.OUTPUT_DIR,exist_ok=True)

		trainer=DefaultTrainer(cfg)
		trainer.resume_or_load(False)
		trainer.train()

		model_parameters=os.path.join(cfg.OUTPUT_DIR,'model_parameters.txt')
		
		model_parameters_dict['cell_mapping']={}
		model_parameters_dict['inferencing_framesize']=int(inference_size)

		for i in range(len(classnames)):
			model_parameters_dict['cell_mapping'][i]=classnames[i]

		with open(model_parameters,'w') as f:
			f.write(json.dumps(model_parameters_dict))

		predictor=DefaultPredictor(cfg)
		model=predictor.model

		DetectionCheckpointer(model).resume_or_load(os.path.join(cfg.OUTPUT_DIR,'model_final.pth'))
		model.eval()

		config=os.path.join(cfg.OUTPUT_DIR,'config.yaml')

		with open(config,'w') as f:
			f.write(cfg.dump())

		print('Detector training completed!')


	def load(self,path_to_detector,cell_kinds):

		config=os.path.join(path_to_detector,'config.yaml')
		detector_model=os.path.join(path_to_detector,'model_final.pth')
		cellmapping=os.path.join(path_to_detector,'model_parameters.txt')
		with open(cellmapping) as f:
			model_parameters=f.read()
		self.cell_mapping=json.loads(model_parameters)['cell_mapping']
		cell_names=json.loads(model_parameters)['cell_names']
		dt_infersize=int(json.loads(model_parameters)['inferencing_framesize'])

		print('The total categories of cells in this Detector: '+str(cell_names))
		print('The cells of interest in this Detector: '+str(cell_kinds))
		print('The inferencing framesize of this Detector: '+str(dt_infersize))

		cfg=get_cfg()
		cfg.merge_from_file(config)
		cfg.MODEL.DEVICE=self.device
		self.current_detector=build_model(cfg)
		DetectionCheckpointer(self.current_detector).load(detector_model)
		self.current_detector.eval()


	def inference(self,inputs):

		with torch.no_grad():
			outputs=self.current_detector(inputs)

		return outputs


