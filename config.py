MODE = 0 # 0 = detector, 1 = segmentation
if MODE == 0:
    weight = 'best-yolov8s-aquarium.pt'
elif MODE == 1:
    weight = 'best-yolov8s-seg-fan.pt'
else:
    raise ValueError("Invalid value for MODE. MODE should be either 0 (detector) or 1 (segmentation).")

source_input = 'aquarium.mp4'

classes_list = None # A list of class indices to consider. If None, all classes will be considered.