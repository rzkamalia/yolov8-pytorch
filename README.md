# Inference YOLOv8 for detection and segementation.

### This repository is "mini" version inference of repository [**YOLOv8 ultralytics**](https://github.com/ultralytics/ultralytics).

Before running the inference, check some variable in file **config.py**:
1. Set mode that you want, **MODE = 0** for detection and **MODE = 1** for segmentation.
1. This repository using weight version **YOLOv8s** both for detection and segmentation. If you want to use other version, replace variable **weight** with your weight path.
1. Make sure variable **source_input** is correct.
1. You can change classes in the variable **classes_dict** according to the classes you are using.

After that you can running the inference with the following command:
```
python3 inference.py
```
Note: you can also change **confident** and **iou** **threshold** in function **non_max_suppression** inside file **utils.py**. Default conf_thres = 0.25 and iou_thres = 0.45.