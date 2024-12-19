# CBVCC
Code for Cell Behavior Video Classification Challenge (CBVCC) 2024

Summary

This pipeline (referred as ‘CB pipeline’ below) was used for the [2024 Cell Behavior Video Classification Challenge (CBVCC)](https://www.immunemap.org/index.php/challenges-menu/cbvcc).

The CB pipeline was modified from [LabGym](https://github.com/umyelab/LabGym) ([ref1](https://www.sciencedirect.com/science/article/pii/S2667237523000267), [ref2](https://www.biorxiv.org/content/10.1101/2024.07.07.602350v1)).

LabGym was developed for analyzing user-defined behaviors, including cell behaviors. My goal here is to use this opportunity to develop a general approach that can be used to track individual cells in videos and classify the user-defined behaviors of these cells, not just restricted to the CBVCC.

The goal of the CBVCC is to classify videos into two categories: 0 (the cell that approximately passes through the center of the frame performs a linear movement or no movement) or 1 (the cell that approximately passes through the center of the frame performs a sudden direction change in movement).

However, there might be multiple cells performing different behaviors in one video. Besides, the cell at different time point (frames) might perform different behaviors. The CB pipeline addresses these issues by tracking all the cells in a video and classifying behaviors for every individual cell at each frame.

To participate the competition, the CB pipeline uses a filter to filter out cells that do not pass through the center of the frame and only focus on one cell per video, after it tracks all the cells and classifies their behaviors. It uses the predictions on the behavior categories at 14th, 15th, and 16th frames to determine the ‘final’ behavior category the ‘center’ cell performs, although it can categorize the behavior in a frame-wise manner.

Methods

1.	Process all the training videos to 100 X 100, 6fps.
2.	Extract frames from training videos, select 475 frames and annotate cells in them with [Roboflow](https://roboflow.com/), and augment the images using: horizontal and vertical flipping, hue between +-15%, 90° rotate (clockwise, counter-clockwise, upside down), brightness between +-25%, and exposure between +-10%.
3.	Train a ‘Detector’ that can detect and segment cells in images.
4.	Use the trained Detector to generate behavior examples from the training videos.
5.	Select some behavior examples and sort them into three categories: inplace, linear, and orient.
6.	Use the sorted behavior examples to train a ‘Categorizer’ that can distinguish the three cell behaviors.
7.	Use the trained Detector and Categorizer to analyze cell behaviors in testing videos.
8.	For each testing video, the analysis output a video copy in which the behaviors are annotated in frame-wise manner (light yellow represents’ inplace’, dark yellow represents ‘linear’, magenta represents ‘orient’), the frame-wise behavior categories and probability, the frame-wise centers of the tracked cells, and the trajectories of the tracked cells.
	


