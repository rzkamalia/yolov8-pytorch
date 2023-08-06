# Inference YOLOv8 for detection and segmentation.

### This repository is "mini" version inference of repository [**YOLOv8 ultralytics**](https://github.com/ultralytics/ultralytics).

Before running the inference, check some variable in file **config.py**:
1. Set mode that you want, **MODE = 0** for detection and **MODE = 1** for segmentation.
1. This repository using weight version **YOLOv8s** both for detection and segmentation. If you want to use other version, replace variable **weight** with your weight path.
1. Make sure variable **source_input** is correct.
1. You can change variable **classes_list** according to classes that will be shown in inference. For example if you training with COCO dataset and you only want class person (which is index = 0 in COCO dataset) shown, so **classes_list = [0]**. Set **classes_list = None** if all classes will be shown.

After that you can running the inference with the following command:
```
python3 inference.py
```
Note: you can also change **confident** and **iou** **threshold** in function **non_max_suppression** inside file **predictor.py**. Default conf_thres = 0.25 and iou_thres = 0.45.

For training using custom dataset, you can see file **training.py**. Dataset that used in this repository:
1. [Aquarium dataset for detection.](https://public.roboflow.com/object-detection/aquarium/2)
1. [Fan v5 dataset for segmentation.](https://universe.roboflow.com/robocup-z5pzj/fan-fgb9n/dataset/5)