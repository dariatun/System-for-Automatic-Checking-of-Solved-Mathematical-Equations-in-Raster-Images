#!/bin/bash
SCRIPT="cat ~/.bashrc; pwd; cd new/darknet/;  pwd; echo \"$PATH\";  pwd; touch test.txt;  pwd; echo \"/home.stud/tunindar/new/darknet/data/capture.jpg\" > test.txt;  pwd; ./darknet detector test data/obj-full1.data cfg/yolov3-full1.cfg backup-eq1/yolov3-full1_best.weights -dont_show -ext_output -out result.json < test.txt > result.txt;  pwd;"
HOSTS="boruvka.felk.cvut.cz"
USERNAMES="tunindar"
PASSWORDS="123123"
#SCR=${SCRIPT/PASSWORD/${PASSWORDS[i]}}
sshpass -p ${PASSWORDS[i]} scp  ~/mach-lerinig/mLStuff/capture.jpg ${USERNAMES[i]}"@"${HOSTS[i]}":/home.stud/tunindar/new/darknet/data/"
cat file.sh | sshpass -p ${PASSWORDS} ssh  -t -oStrictHostKeyChecking=no "tunindar@boruvka.felk.cvut.cz"
#ssh -T ${USERNAMES}"@"${HOSTS} 'cd new/darknet/'
#ssh -T ${USERNAMES}"@"${HOSTS} './darknet detector test data/obj-full1.data cfg/yolov3-full1.cfg backup-eq1/yolov3-full1_best.weights -dont_show -ext_output -out result.json < test.txt > result.txt'

sshpass -p ${PASSWORDS[i]} scp ${USERNAMES[i]}"@"${HOSTS[i]}":/home.stud/tunindar/new/darknet/result.json" "/Users/dariatunina/mach-lerinig/mLStuff/"

