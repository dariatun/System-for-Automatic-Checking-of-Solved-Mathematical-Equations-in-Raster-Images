# Recognition of mathematical expressions using neural networks
yolo/ folder:\
|_ obj.data\
|_ obj.names\
|_ yolov3-obj.cfg\
|_ yolov3-obj_best.weights\

implementation/ folder:\
|__ generating_dataset.py\
|__ divide_to_train_and_test.py\
|__ handwritten_recogniser.py\
|__ utils.py\
|__ bg_images/\
|__ cnn_weights.hdf5\
|__ fonts/\

1. Generating dataset\
    For the dataset generation some libraries and github directories should be installed:
    * https://github.com/Paperspace/DataAugmentationForObjectDetection.git - the path to the directory should be written on the 13 line in utils.py file.
    * PIL (Python Image Library)
    * NumPy
    * cv2
    * loremipsum
    * matplotlib
    * sys
    * https://github.com/anujdutt9/Handwritten-Digit-Recognition-using-Deep-Learning.git - the path to the directory should be written on the 15 line in generating_dataset.py file.
    
    The generation can be started with the command: python generating_dataset.py
    
2. Divide to train and test dataset\
    Starts with the command: python divide_to_train_and_test.py
    
3. Handwritten digits recogniser\
    For this te following libraries and github directories should be installed:
    * json
    * sys
    * NumPy
    * PIL
    * matplotlib
    * keras
    * https://github.com/anujdutt9/Handwritten-Digit-Recognition-using-Deep-Learning.git - the path to the directory should be written on the 10 line in handwritten_recogniser.py file.
    
    Starts with the command: python handwritten_recogniser.py

4. YOLO test\
    Install YOLO.
    Command to test: ./darknet detector test obj.data yolov3-obj.cfg yolov3-obj_best.weights