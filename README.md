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



### Experiments

(1) I ran it for 1.mp4 video and got this result:

https://github.com/Faryalaurooj/Tracking-with-YOLOv8/assets/138756263/e1f6582d-b423-4eb5-baae-3e0b0254b29b

Speed: 0.7ms preprocess, 6.1ms inference, 1.3ms postprocess, 21.9ms tracking per image at shape (1, 3, 384, 640)
Results saved to /home/caic/anaconda3/envs/yolo_ds1/lib/python3.9/site-packages/runs/track/exp5

(2) Then i did tracking on another video examp.mp4 and got this result:


video 1/1 (27002/27002) /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov8/yolo_tracking/examp.mp4: 384x640 4 persons, 5.8ms
Speed: 1.0ms preprocess, 5.7ms inference, 0.9ms postprocess, 19.3ms tracking per image at shape (1, 3, 384, 640)
Results saved to /home/caic/anaconda3/envs/yolo_ds1/lib/python3.9/site-packages/runs/track/exp8


Run it by
(3) Then i did it through this command 

```
python examples/track.py --yolo-model yolov8n --source 'https://youtu.be/_zIKxCB3jcI' --save  --classes 0 1 
```
and i got this result

(4) By default, deeposcort tracker is used. If we want to do tracking with botsort , strongsort, ocsort, bytetrack then we can do this in this manner:
    Now with ocsort:
```
 python examples/track.py --source 1.mp4 --yolo-model yolov8s.pt --save --classes 0 1 ----tracking-method ocsort
```
Following results achieved
Speed: 0.7ms preprocess, 6.3ms inference, 1.5ms postprocess, 1.1ms tracking per image at shape (1, 3, 384, 640)
Results saved to /home/caic/anaconda3/envs/yolo_ds1/lib/python3.9/site-packages/runs/track/exp12-oscort


(5) with botsort

```
 python examples/track.py --source 1.mp4 --yolo-model yolov8s.pt --save --classes 0 1 ----tracking-method botsort
```
results are:
Speed: 0.7ms preprocess, 6.4ms inference, 1.5ms postprocess, 21.4ms tracking per image at shape (1, 3, 384, 640)
Results saved to /home/caic/anaconda3/envs/yolo_ds1/lib/python3.9/site-packages/runs/track/exp13-botsort

(6) with strongsort:

Speed: 0.7ms preprocess, 6.3ms inference, 1.5ms postprocess, 23.0ms tracking per image at shape (1, 3, 384, 640)
Results saved to /home/caic/anaconda3/envs/yolo_ds1/lib/python3.9/site-packages/runs/track/exp14-strongsort

(7) with bytetrack:

Speed: 0.7ms preprocess, 6.1ms inference, 1.5ms postprocess, 1.2ms tracking per image at shape (1, 3, 384, 640)
Results saved to /home/caic/anaconda3/envs/yolo_ds1/lib/python3.9/site-packages/runs/track/exp15-bytetrack


### Results

We are using a fast multi-object tracking genetic algorithm for tracker hyperparameter tuning. By default the objectives are:

(1) HOTA (Higher Order Tracking Accuracy) : is a novel metric which balances the effect of performing accurate detection, association and localization into a single unified metric for comparing trackers.

(2) MOTA (Multiple Object Tracking Accuracy) : MOTA performs both matching and association scoring at a local detection level but pronounces detection accuracy more.

(3) IDF1 : IDF1 performs at a trajectory level by emphasizing the effect of association. It is the ratio of correctly identified detections over the average number of ground-truth and computed detections. The basic idea of IDF1 is to com- bine IDP and IDR to a single number.

(4) Tracking speed

By comparing results from above experiments benchmarked on same video '1.mp4' it was observed that tracking speed per image is :

oscort   = 1.1ms
deepsort =  21.9ms 
botsort  = 21.4ms
strongsort = 23.0ms
bytetrack = 1.2ms

so oscort is FASTEST and bytetrack is second FAST tracker found so far.

