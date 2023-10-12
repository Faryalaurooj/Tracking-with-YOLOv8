# Tracking-with-YOLO

In this repo, we will perform object tracking for vulnerable road users (VRUs) like pedestrians bicycle and tricycles.
Object tracking in video analytics is a critical task that not only identifies the location and class of objects within the frame but also maintains a unique ID for each detected object as the video progresses. The applications are limitless—ranging from surveillance and security to real-time vulnerable road users safety in case of traditional vehicles and self driving vehicles.

# What we expect from tracker

Perfromace Objectives of a good tracker are :

(1) Detection Accuracy : Detection measures the alignment between the set of all predicted detections and the set of all ground-truth detections.

<img width="687" alt="3" src="https://github.com/Faryalaurooj/Tracking-with-YOLOv8/assets/138756263/88c9baa9-17f1-476f-9a15-34441e75dcbe">


(2) Association Accuracy : Association measures how well a tracker links detections over time into the same identities (IDs), given the ground-truth set of identity links in the ground-truth tracks.

<img width="688" alt="5" src="https://github.com/Faryalaurooj/Tracking-with-YOLOv8/assets/138756263/33d8ff3a-6627-46cb-a5b0-b69057e599fa">


(3) Localization Accuracy : Localization measures the spatial alignment between one predicted detection and one ground-truth detection. 

<img width="687" alt="1" src="https://github.com/Faryalaurooj/Tracking-with-YOLOv8/assets/138756263/1d55ce37-c345-4ea5-97b6-194ac78f579b">


There are standard metrices to measure these three parameters. 


(1) MOTA (Multiple Object Tracking Accuracy) : MOTA performs both matching and association scoring at a local detection level but pronounces detection accuracy more.


(2) IDF1 : IDF1 performs at a trajectory level by emphasizing the effect of Association. The basic idea of IDF1 is to combine IDP (precision) and IDR (recall) to a single number.


(3) HOTA (Higher Order Tracking Accuracy) : is a novel metric which balances the effect of performing accurate Detection, Association and Localization into a single unified metric for comparing trackers.


<img width="685" alt="6" src="https://github.com/Faryalaurooj/Tracking-with-YOLOv8/assets/138756263/2351ef58-8482-4eda-8962-70da0a08e490">

There exists no single universal super tracker that is good in everything. But HOTA still provides a balanced matrix for all three perfromance requirements.  In this repo, we will evaluate the trackers on HOTA performance matrix. As an example, let’s look at the results of the top 20 methods on the KITTI tracking leaderboard for pedestrians:

<img width="707" alt="7" src="https://github.com/Faryalaurooj/Tracking-with-YOLOv8/assets/138756263/4ad77c92-e047-409c-b56b-3429118c33f8">

The rank (red number) gives the order of the trackers, ordered by the overall HOTA score. The top 3 ranked trackers have a very similar overall HOTA scores (46.3%, 45.9% and 45.7%), but we can see from this plot that there is a clear difference in where one is better than the other. Tracker 1 is the best at association, while Tracker 2 is better at detection and Tracker 3 is in between the two for both. If you wanted to select a tracker for a particular application, you could now decide whether association or detection was more important for your application and pick the most appropriate tracker accordingly. 

We can go further than just comparing detection and association. We can compare on basis of Recall and Precision. Detection recall (DetRe) measures how well a tracker finds all the ground-truth detections, whereas detection precision (DetPr) measures how well a tracker manages to not predict extra detections that aren’t there. Precision and recall are commonly used for evaluating detection, but now with the HOTA metrics we can extend these concepts to also measure association.

<img width="708" alt="8" src="https://github.com/Faryalaurooj/Tracking-with-YOLOv8/assets/138756263/ab7d3820-bb49-4a09-b28f-4c215d187860">

The above plots have tracker numbers ordered by the overall HOTA score still, so the same number refers to the same tracker as above. From the first plot above, we can see that Tracker 1 and Tracker 3 have a similar detection accuracy overall, but Tracker 3 generally finds more of the ground-truth objects (higher recall), but also predicts more detections that are wrong (lower precision). Association recall (AssRe) measures how well trackers can avoid splitting the same object into multiple shorter tracks. In contrast, association precision (AssPr) measures how well tracks can avoid merging multiple objects together into a single track. E.g. Tracker 15 is more likely to split tracks into multiple smaller ones than tracker 20, but it is better at not merging tracks together. Like detection precision and recall, there is a natural trade-off between association precision and recall when designing trackers.


The HOTA metrics allows meaningful analysis and comparison between trackers over all four of these dimensions (missing detections, extra detections, splitting tracks and merging tracks) while also combining all of these scores meaningfully into an overall score for ranking trackers.

Finally, HOTA also allows the analysis of localization accuracy:


<img width="690" alt="9" src="https://github.com/Faryalaurooj/Tracking-with-YOLOv8/assets/138756263/09f379e6-1436-49b3-83b2-5c0d40dfaf60">


In the first plot above we compare HOTA(0) (HOTA at the single lowest alpha threshold, so to not include the influence of localization accuracy, in this case at alpha=0.05) against the localization accuracy LocA(0) (LocA at the same threshold). We can see that Tracker 3 performs slightly better than Tracker 1 at HOTA(0), e.g. when allowing detections to match even if they have only a little overlap, Tracker 3 has overall better detection + association, however the localization of these matched detections is worse, such that when we calculate the final HOTA score by calculating over the range of localization thresholds, Tracker 1 has a higher score. This shows how HOTA is able to decompose and combine tracker behavior not only for detection and association but also for localization.

In the second plot, we compare the top 5 trackers’ HOTA scores across the range of different alpha thresholds. All trackers get worse when increasing the alpha threshold, but the rate at which they get worse is interesting and is useful for comparing behavior between trackers.


## Why Choose Ultralytics YOLO for Object Tracking ?

The output from Ultralytics trackers is consistent with standard object detection but has the added value of object IDs. This makes it easy to track objects in video streams and perform subsequent analytics. Here's why i have considered using Ultralytics YOLO for my VRU tracking project:

   (1) Efficiency: Process video streams in real-time without compromising accuracy.
   
   
   (2) Flexibility: Supports multiple tracking algorithms and configurations.
   
   
   (3) Ease of Use: Simple Python API and CLI options for quick integration and deployment.
   
   
   (4) Customizability: Easy to use with custom trained YOLO models, allowing integration into domain-specific applications.
   

## What i am going to do

I have perfromed this task with YOLOv5 and YOLOv8 models. In my other repo, i have provided complete details on development of a custom dataset for VRUs , training of YOLOv5 , YOLOv7 and YOLOv8 models on custom dataset. The detection results with YOLOv8 and YOLOv5 were good but with YOLOv7 were not good. In this repository i will do two things :

   (1)  Perform tracking with already trained YOLOv8 model with different trackers bytetrack, strongsort, ocsort etc and benchmarking performance HOTA on MOT17 dataset
   
   (2)  Perfrom tracking with custom trained YOLOv5 and YOLOv8 on my VRU dataset

   (3)  Perfrom tracking with transfer learning on COCO trained YOLOv5 and YOLOv8 models , then trained further on my custom VRU dataset

   (4)  Provide comparison of results


## Tracking with custom trained YOLO on VRU dataset

### (1) Download repo for YOLOv8
Clone repo and install requirements.txt in a Python>=3.8.0 environment, including PyTorch>=1.7. Models and datasets download automatically from the latest YOLOv5 release.

```
git clone https://github.com/yolo_tracking  # clone
pip install -r requirements.txt  # install
```
### (2) Tracking
YOLOv5s and YOLOv5x are already trained on VRU dataset. Whichever model you want to use , simply copy the weights best.pt file from yolov5/runs/train/yolov5x or yolov5s subfolder into main yolov5 directory and then run this command
```

  python examples/track.py --source 1.mp4 --yolo-model yolov8s.pt --save --classes 0 1 # COCO yolov8 model for persons and bicycles detection only. 

```


#### Tracking Methods
By default, deeposcort tracker is used. I have benchmarked trackers one by one by with '1.mp4' video using following command:

```
$ python examples/track.py --tracking-method deepocsort
                                             strongsort
                                             ocsort
                                             bytetrack
                                             botsort
```
#### Tracking Sources
```
$ python examples/track.py --source 0                               # webcam
                                    img.jpg                         # image
                                    vid.mp4                         # video
                                    path/                           # directory
                                    path/*.jpg                      # glob
                                    'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                    'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```
By default the tracker tracks all MS COCO classes. In our application we need to track only Vulnerable road users (persons, bicycles , tricycles) therefore we will add their corresponding index after the classes flag. 

#### MOT compliant results

Resultant video with tracking can be saved to your experiment folder runs/track/exp*/ by --save and it can be seved as a text file by --save-mot
```
python examples/track.py --source ... --save-mot
```

#### Filter tracked classes
By default the tracker tracks all MS COCO classes.

If you want to track a subset of the classes that you model predicts, add their corresponding index after the classes flag,
```
python examples/track.py --source 0 --yolo-model yolov8s.pt --classes 16 17  # COCO yolov8 model. Track cats and dogs, only
```
### Experiments

(1)  For deepsort tracker, got this result:

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
### Evaluation

Evaluate a combination of detector, tracking method on standard MOT 17 dataset or our custom VRU_dataset :
```
$ python3 examples/val.py --yolo-model yolo_nas_s.pt --reid-model osnetx1_0_dukemtcereid.pt --tracking-method deepocsort --benchmark MOT16
                          --yolo-model yolox_n.pt    --reid-model osnet_ain_x1_0_msmt17.pt  --tracking-method ocsort     --benchmark MOT17
                          --yolo-model yolov8s.pt    --reid-model lmbn_n_market.pt          --tracking-method strongsort --benchmark <your-custom-dataset>
```

### Evolution

We are using a fast multi-object tracking genetic algorithm for tracker hyperparameter tuning. 

By comparing results from above experiments (1) till (5) benchmarked on same video '1.mp4' it was observed that tracking speed per image is:

deepsort =   21.9ms 
ocsort   =   1.1ms
botsort  =   21.4ms
strongsort = 23.0ms
bytetrack =  1.2ms

So oscort is FASTEST and bytetrack is second FAST tracker found so far.

Run it by this command:

```
$ python examples/evolve.py --tracking-method strongsort --benchmark MOT17 --n-trials 100  # tune strongsort for MOT17
                            --tracking-method ocsort     --benchmark <your-custom-dataset> --objective HOTA # tune ocsort for maximizing HOTA on your custom tracking dataset
```

with strongsort following results achieved :

![pedestrian_plot](https://github.com/Faryalaurooj/Tracking-with-YOLOv8/assets/138756263/1c8136a3-95be-4e32-b867-ec1ef6030d5a)


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









