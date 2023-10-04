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

## Tracking with custom trained YOLO on VRU dataset

### (1) Download repo for YOLOv8
Clone repo and install requirements.txt in a Python>=3.8.0 environment, including PyTorch>=1.7. Models and datasets download automatically from the latest YOLOv5 release.

```
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```
### (2) Tracking
YOLOv5s and YOLOv5x are already trained on VRU dataset. Whichever model you want to use , simply copy the weights best.pt file from yolov5/runs/train/yolov5x or yolov5s subfolder into main yolov5 directory and then run this command
```

  python examples/track.py --source 1.mp4 --yolo-model yolov8s.pt --save --classes 0 1 # COCO yolov8 model for persons and bicycles detection only. 
  python examples/track.py --source 0                               # webcam
                                    img.jpg                         # image
                                    vid.mp4                         # video
                                    path/                           # directory
                                    path/*.jpg                      # glob
                                    'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                    'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

By default the tracker tracks all MS COCO classes. In our application we need to track only Vulnerable road users (persons, bicycles , tricycles) therefore we will add their corresponding index after the classes flag. Resultant video with tracking can be saved to your experiment folder runs/track/exp*/ by --save and it can be seved as a text file by --save-mot

I ran it for 1.mp4 video and got this result



https://github.com/Faryalaurooj/Tracking-with-YOLOv8/assets/138756263/8fba4585-aae9-4de6-9e1d-3feccf576dbf



Speed: 0.7ms preprocess, 6.1ms inference, 1.3ms postprocess, 21.9ms tracking per image at shape (1, 3, 384, 640)
Results saved to /home/caic/anaconda3/envs/yolo_ds1/lib/python3.9/site-packages/runs/track/exp5
