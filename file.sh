#cat ~/.bashrc
#echo "$PATH"
#exec "$BASH"
echo "$PATH"
cd new/darknet/
touch test.txt
echo "/home.stud/tunindar/new/darknet/data/capture.jpg" > test.txt
./darknet detector test data/obj-full1.data cfg/yolov3-full1.cfg backup-eq1/yolov3-full1_best.weights -dont_show -ext_output -out result.json < test.txt > result.txt
