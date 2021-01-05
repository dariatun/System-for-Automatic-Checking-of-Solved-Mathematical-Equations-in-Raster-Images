# System for Automatic Checking of Solved Mathematical Equations in Raster Images

## Instalation
```bash
pip install -r requirements.txt
```
For this project https://github.com/anujdutt9/Handwritten-Digit-Recognition-using-Deep-Learning.git 
is needed. The path to the directory should be added to training_dataset_generation.constants file and
application.changeables.\

For the training dataset generation is also needed
https://github.com/Paperspace/DataAugmentationForObjectDetection.git.
The path to it should be added to training_dataset_generation.constants file.

## Usage
To start the training dataset generation:
```bash
python3 start_generating_dataset.py
```

To start the application:
```bash
python3 start_application.py
```

It also can be started with argument -p < path >, where path is 
a path with the images that are desired to process.

### Example:
```bash
python3 start_application.py -p 'images/program_images'
```
Argument -e is used for starting an evaluation. It will take 
images from evaluation_data/test_images folder and their annotations from
evaluation_data/test_annotations.

### Example:
```bash
python3 start_application.py -e
```

## File management description

1. model/ \
|_ obj.data\
|_ obj.names\
|_ yolov3-obj.cfg\
|_ yolov3-obj_best.weights


2. training_dataset_generation/ \
|__ generating_dataset.py\
|__ divide_to_train_and_test.py\
|__ data_augmentation.py\
|__ utils.py\
|__ constants.py
   

3. cnn_weights.hdf5


4. application/ \
|__ answer_correctness_checker.py\
|__ application.py\
|__ changeables.py\
|__ constants.py\
|__ detect_equation.py\
|__ detect_handwritten_digit.py\
|__ evaluation_mode.py\
|__ image_manipulation.py\
|__ logger.py\
|__ object_detection.py\
|__ object_localisation.py\
|__ prediction_matrix_creation.py\
|__ scanner.py\
|__ utils.py


5. fonts/ 


6. images/ \
   |__ background/ \
   |__ corner/ \
   |__ emotions/ \
   |__ program_images/ 
   

7. requirements.txt
   

8. output/ 


9. temporary/ 


10. evaluation_data/ \
   |__ test_images/ \
   |__ test_annotations/ 
    

11. start_application.py


12. start_generating_dataset.py
    nd to test: ./darknet detector test obj.data yolov3-obj.cfg yolov3-obj_best.weights