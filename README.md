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

   (1)  Perfrom tracking with custom trained YOLOv5 and YOLOv7 on my VRU dataset

   (2)  Perfrom tracking with transfer learning on COCO trained YOLOv5 and YOLOv7 models , then trained further on my custom VRU dataset

   (3)  Provide comparison of results

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

(1) By default, deeposcort tracker is used. I have benchmarked trackers one by one by with '1.mp4' video. For deepsort tracker, got this result:

https://github.com/Faryalaurooj/Tracking-with-YOLOv8/assets/138756263/e1f6582d-b423-4eb5-baae-3e0b0254b29b

Speed: 0.7ms preprocess, 6.1ms inference, 1.3ms postprocess, 21.9ms tracking per image at shape (1, 3, 384, 640)
Results saved to /home/caic/anaconda3/envs/yolo_ds1/lib/python3.9/site-packages/runs/track/exp5


(2)  If we want to do tracking with botsort , strongsort, ocsort, bytetrack then we can do this in this manner. First with ocsort:

```
 python examples/track.py --source 1.mp4 --yolo-model yolov8s.pt --save --classes 0 1 ----tracking-method ocsort
```

Following results achieved
Speed: 0.7ms preprocess, 6.3ms inference, 1.5ms postprocess, 1.1ms tracking per image at shape (1, 3, 384, 640)
Results saved to /home/caic/anaconda3/envs/yolo_ds1/lib/python3.9/site-packages/runs/track/exp12-oscort

(3) with botsort

```
 python examples/track.py --source 1.mp4 --yolo-model yolov8s.pt --save --classes 0 1 ----tracking-method botsort
```
results are:
Speed: 0.7ms preprocess, 6.4ms inference, 1.5ms postprocess, 21.4ms tracking per image at shape (1, 3, 384, 640)
Results saved to /home/caic/anaconda3/envs/yolo_ds1/lib/python3.9/site-packages/runs/track/exp13-botsort

(4) with strongsort:

Speed: 0.7ms preprocess, 6.3ms inference, 1.5ms postprocess, 23.0ms tracking per image at shape (1, 3, 384, 640)
Results saved to /home/caic/anaconda3/envs/yolo_ds1/lib/python3.9/site-packages/runs/track/exp14-strongsort

(5) with bytetrack:

Speed: 0.7ms preprocess, 6.1ms inference, 1.5ms postprocess, 1.2ms tracking per image at shape (1, 3, 384, 640)
Results saved to /home/caic/anaconda3/envs/yolo_ds1/lib/python3.9/site-packages/runs/track/exp15-bytetrack

(2) Then i did tracking on another video examp.mp4 and got this result:


video 1/1 (27002/27002) /home/caic/Downloads/yolo_series_deepsort_pytorch/yolov8/yolo_tracking/examp.mp4: 384x640 4 persons, 5.8ms
Speed: 1.0ms preprocess, 5.7ms inference, 0.9ms postprocess, 19.3ms tracking per image at shape (1, 3, 384, 640)
Results saved to /home/caic/anaconda3/envs/yolo_ds1/lib/python3.9/site-packages/runs/track/exp8


(6) Then i did it through this command which is for a video url instead of source video in directory

```
python examples/track.py --yolo-model yolov8n --source 'https://youtu.be/_zIKxCB3jcI' --save  --classes 0 1 
```

### Results

We are using a fast multi-object tracking genetic algorithm for tracker hyperparameter tuning. By default the objectives are:

(1) HOTA (Higher Order Tracking Accuracy) : is a novel metric which balances the effect of performing accurate detection, association and localization into a single unified metric for comparing trackers.

(2) MOTA (Multiple Object Tracking Accuracy) : MOTA performs both matching and association scoring at a local detection level but pronounces detection accuracy more.

(3) IDF1 : IDF1 performs at a trajectory level by emphasizing the effect of association. It is the ratio of correctly identified detections over the average number of ground-truth and computed detections. The basic idea of IDF1 is to com- bine IDP and IDR to a single number.

(4) Tracking speed

By comparing results from above experiments (1) till (5) benchmarked on same video '1.mp4' it was observed that tracking speed per image is:

deepsort =   21.9ms 
ocsort   =   1.1ms
botsort  =   21.4ms
strongsort = 23.0ms
bytetrack =  1.2ms

So oscort is FASTEST and bytetrack is second FAST tracker found so far.

## Tracking with custom trained YOLOv8:

After completing tracking with COCO dataset weights, i have to perform tracking with my own custom data VRU_dataset. For that i copied the weights best.pt from yolov8/runs/train/yolov8s folder into main directory yolo_tracking and rename them as yolov8s_custom then perform tracking with this command:

```
cd yolo_tracking
python examples/track.py --source 1.mp4 --yolo-model yolov8s_custom.pt --save   # default tracking method is deeposcort
```

After running above command i get this resulting video with tracking results:

Speed: 0.6ms preprocess, 5.6ms inference, 0.4ms postprocess, 3.4ms tracking per image at shape (1, 3, 384, 640)
Results saved to /home/caic/anaconda3/envs/yolo_ds1/lib/python3.9/site-packages/runs/track/exp13

so we observed tracking time is reduced from 21.9ms to 3.4ms on same '1.mp4' video but there are missed detections. It only detects small objects , not big persons which needs improvement.


https://github.com/Faryalaurooj/Tracking-with-YOLOv8/assets/138756263/bd94c06e-1a99-4ece-a655-3ad7f74e00d6



## Transfer Learning 

In above case where i trained YOLOv8s on my custom VRU_dataset, i observed that detection performance parameters were credibaly good (NOT AS GOOD AS ON coco DATASET) BUT WHEN I PERFORMED TRACKING, RESULTS WERE NOT GOOD. MAny frames were missed in the video 1.mp4 without tracking. THerefore, i have performed 'transfer learning' on YOLOv8s by freezing 12 layers, and training remaining model onto my custom VRU_dataset to see how does tracking performance improves.
For this i ran this command:
```
python train_transfer_learning.py --freeze 12 --data VRU.yaml --epochs 300 --img 640  --batch 4 --cfg  ./models/yolov5s.yaml --weights ''  --workers 8 --name yolov5

```
train_transfer_learning: weights=, cfg=./models/yolov5s.yaml, data=VRU.yaml, hyp=data/hyps/hyp.scratch-low.yaml, epochs=300, batch_size=4, imgsz=640, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, bucket=, cache=None, image_weights=False, device=, multi_scale=False, single_cls=False, optimizer=SGD, sync_bn=False, workers=8, project=runs/train, name=yolov5, exist_ok=False, quad=False, cos_lr=False, label_smoothing=0.0, patience=100, freeze=[12], save_period=-1, seed=0, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest
Command 'git fetch origin' timed out after 5 seconds
YOLOv5 🚀 v7.0-217-g8c45e51 Python-3.9.17 torch-1.13.1+cu117 CUDA:0 (NVIDIA GeForce RTX 2080 SUPER, 7982MiB)

hyperparameters: lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
Comet: run 'pip install comet_ml' to automatically track and visualize YOLOv5 🚀 runs in Comet
TensorBoard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/
Overriding model.yaml nc=80 with nc=3
YOLOv5s summary: 214 layers, 7027720 parameters, 7027720 gradients, 16.0 GFLOPs
AMP: checks passed ✅
freezing model.0.conv.weight
freezing model.0.bn.weight
freezing model.0.bn.bias

......

300 epochs completed in 14.708 hours.
Optimizer stripped from runs/train/yolov53/weights/last.pt, 14.3MB
Optimizer stripped from runs/train/yolov53/weights/best.pt, 14.3MB

Validating runs/train/yolov53/weights/best.pt...
Fusing layers... 
YOLOv5s summary: 157 layers, 7018216 parameters, 0 gradients, 15.8 GFLOPs
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 69/69 [00:04<00:00, 14.10it/s]
                   all        548      16833      0.134     0.0617     0.0449     0.0134
                people        548      13969      0.168      0.154     0.0993     0.0295
              tricycle        548       1577      0.153     0.0178     0.0258    0.00821
               bicycle        548       1287     0.0795     0.0136    0.00953    0.00266
Results saved to runs/train/yolov53

When i run detect command with this trained model with :

```
cd yolov5
python detect.py --weights best.pt --source ./VRU_Dataset/images/test

```
i get this result:
Speed: 0.2ms pre-process, 5.4ms inference, 0.2ms NMS per image at shape (1, 3, 640, 640)
Results saved to runs/detect/exp6
                             YOLOv5s     YOLOv5s (TF_12_layers_freeze)
Precision                      0.455           0.134
Recall                         0.292           0.0617
mAP                            0.292           0.0449
Inference time (ms)            5.7              5.4
FPS                            175              185
Training time (hrs)            16.6             14.7
SO we observed that the precision recall mAP values have dicreased with transfer learning as compared to training from scratch , however inference time and FPS (speed) and training time has improved slightly.

Now lets try tracking:


