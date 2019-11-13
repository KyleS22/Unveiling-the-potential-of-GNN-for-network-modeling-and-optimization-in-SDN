# Usage:
# $1 hyperparam string
# $2 train data path
# $3 test data path
# $4 out path


#TODO FIX THIS TO WORK WIT PYTHON SCRIPT

if [[ "$1" = "train" ]]; then

    python3 routenet_with_link_cap.py train --hparams="l2=0.1,dropout_rate=0.5,link_state_dim=32,path_state_dim=32,readout_units=256,learning_rate=0.001,T=8"  --train  $PATH_TO_DATA/tfrecords/train/*.tfrecords --train_steps $3 --eval_ $PATH_TO_DATA/tfrecords/evaluate/*.tfrecords --model_dir ./CheckPoints/$2

fi

if [[ "$1" = "train_multiple" ]]; then

    python3 routenet_with_link_cap.py train --hparams="l2=0.1,dropout_rate=0.5,link_state_dim=32,path_state_dim=32,readout_units=256,learning_rate=0.001,T=8"  --train  /home/datasets/SIGCOMM/$2/tfrecords/train/*.tfrecords /home/datasets/SIGCOMM/$3/tfrecords/train/*.tfrecords --train_steps $5 --eval_ /home/datasets/SIGCOMM/geant2bw/tfrecords/evaluate/*.tfrecords /home/datasets/SIGCOMM/geant2bw/tfrecords/train/*.tfrecords --shuffle_buf 30000 --model_dir ./CheckPoints/$4
fi


