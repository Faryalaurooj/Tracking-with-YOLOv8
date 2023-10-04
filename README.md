# Tracking-with-YOLO
In this repo, we will perform object tracking for vulnerable road users (VRUs) like pedestrians bicycle and tricycles.
Object tracking in video analytics is a critical task that not only identifies the location and class of objects within the frame but also maintains a unique ID for each detected object as the video progresses. The applications are limitless—ranging from surveillance and security to real-time vulnerable road users safety in case of traditional vehicles and self driving vehicles.

## Why Choose Ultralytics YOLO for Object Tracking?

The output from Ultralytics trackers is consistent with standard object detection but has the added value of object IDs. This makes it easy to track objects in video streams and perform subsequent analytics. Here's why i have considered using Ultralytics YOLO for my VRU tracking project:

   (1) Efficiency: Process video streams in real-time without compromising accuracy.
   (2) Flexibility: Supports multiple tracking algorithms and configurations.
   (3) Ease of Use: Simple Python API and CLI options for quick integration and deployment.
   (4) Customizability: Easy to use with custom trained YOLO models, allowing integration into domain-specific applications.

## What i am going to do
I have perfromed this task with YOLOv5 and YOLOv8 models. In my other repo, i have provided complete details on development of a custom dataset for VRUs , training of YOLOv5 , YOLOv7 and YOLOv8 models on custom dataset. THe detection results with YOLOv8 and YOLOv5 were good but with YOLOv7 were not good. In this repository i will do two things :

(1) Perfrom tracking with custom trained YOLOv5 and YOLOv7 on my VRU dataset

(2) Perfrom tracking with transfer learning on COCO trained YOLOv5 and YOLOv7 models , then trained further on my custom VRU dataset

(3) Provide comparison of results
