
Create tf records

python ./create_tf_record.py \
    --data_dir=/home/arpit/training \
    --output_dir=/home/arpit/training/data \
    --label_map_path=/home/arpit/training/data/label_map.pbtxt

python train.py \
    --logtostderr \
    --pipeline_config_path=/home/arpit/training/models/model/ssd_mobilenet_v1_coco.config \
    --train_dir=/home/arpit/training/models/model/train  

python eval.py \
    --logtostderr \
    --pipeline_config_path=/home/arpit/training/models/model/ssd_mobilenet_v1_coco.config \
    --checkpoint_dir=/home/arpit/training/models/model/train \
    --eval_dir=/home/arpit/training/models/model/eval

python train.py \
    --logtostderr \
    --pipeline_config_path=/home/arpit/training/models/model/faster_rcnn_resnet101_coco.config \
    --train_dir=/home/arpit/training/models/model/train  

python eval.py \
    --logtostderr \
    --pipeline_config_path=/home/arpit/training/models/model/faster_rcnn_resnet101_coco.config \
    --checkpoint_dir=/home/arpit/training/models/model/train \
    --eval_dir=/home/arpit/training/models/model/eval


import tensorflow as tf 
>>> g = tf.GraphDef()
>>> g.ParseFromString(open(“path/to/mymodel.pb”, “rb”).read())
>>> [n for n in g.node if n.name.find(“input”) != -1] # same for     