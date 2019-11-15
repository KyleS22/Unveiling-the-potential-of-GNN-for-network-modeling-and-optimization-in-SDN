# Usage:
# $1 train or train_multiple
# $2 hyperparam string
# $3 train data path
# $4 test data path
# $5 out path
# $6 train2 data path

TRAIN_1="$3/tfrecords/train/*.tfrecords"
TRAIN_2="$6/tfrecords/train/*.tfrecords"
TEST="$4/tfrecords/evaluate/*.tfrecords"


if [ "$1" = "train" ]; then
    python3 routenet_with_link_cap.py train --hparams="$2" --train $TRAIN_1 --train_steps 50000 --eval_ $TEST --model_dir $5 

    rc=$?; if [ $rc != 0 ]; then exit $rc; fi
fi

if [ "$1" = "train_multiple" ]; then
    python3 routenet_with_link_cap.py train --hparams="$2" --train $TRAIN_1 $TRAIN2 --train_steps 50000 --eval_ $TEST --model_dir $5 
    rc=$?; if [ $rc != 0 ]; then exit $rc; fi
fi

