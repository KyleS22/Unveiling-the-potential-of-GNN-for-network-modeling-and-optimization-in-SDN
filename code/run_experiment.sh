# Usage:
# $1 train or train_multiple
# $2 hyperparam string
# $3 train data path
# $4 test data path
# $5 out path
# $6 train2 data path

echo $1
echo $2
echo $3
echo $4
echo $5
echo $6


if [ "$1" = "train" ]; then
    ls "$3/tfrecords/train/*.tfrecords" "$6/tfrecords/train/*.tfrecords"
    python3 routenet_with_link_cap.py train --hparams="$2" --train "$3/tfrecords/train/*.tfrecords" --train_steps 50000 --eval_ "$4/tfrecords/evaluate/*.tfrecords" --model_dir "$5"
    rc=$?; if [ $rc != 0 ]; then exit $rc; fi
fi

if [ "$1" = "train_multiple" ]; then
    ls "$3/tfrecords/train/*.tfrecords" "$6/tfrecords/train/*.tfrecords" 
    python3 routenet_with_link_cap.py train --hparams="$2" --train "$3/tfrecords/train/*.tfrecords" "$6/tfrecords/train/*.tfrecords" --train_steps 50000 --eval_ "$4/tfrecords/evaluate/*.tfrecords" --model_dir "$5"
    rc=$?; if [ $rc != 0 ]; then exit $rc; fi
fi

#if [[ "$1" = "train" ]]; then
#
#    python3 routenet_with_link_cap.py train --hparams="l2=0.1,dropout_rate=0.5,link_state_dim=32,path_state_dim=32,readout_units=256,learning_rate=0.001,T=8"  --train  $PATH_TO_DATA/tfrecords/train/*.tfrecords --train_steps $3 --eval_ $PATH_TO_DATA/tfrecords/evaluate/*.tfrecords --model_dir ./CheckPoints/$2
#
#fi
#
#if [[ "$1" = "train_multiple" ]]; then
#
#    python3 routenet_with_link_cap.py train --hparams="l2=0.1,dropout_rate=0.5,link_state_dim=32,path_state_dim=32,readout_units=256,learning_rate=0.001,T=8"  --train  /home/datasets/SIGCOMM/$2/tfrecords/train/*.tfrecords /home/datasets/SIGCOMM/$3/tfrecords/train/*.tfrecords --train_steps $5 --eval_ /home/datasets/SIGCOMM/geant2bw/tfrecords/evaluate/*.tfrecords /home/datasets/SIGCOMM/geant2bw/tfrecords/train/*.tfrecords --shuffle_buf 30000 --model_dir ./CheckPoints/$4
#fi
#
#
