
Create tf records

python preparedata.py \
    --data_dir=. \
    --output_dir=data \
    --label_map_path=data/label_map.pbtxt

python train.py \
    --logtostderr \
    --pipeline_config_path=models/model/ssd_mobilenet_v1_coco.config \
    --train_dir=models/model/train  

python eval.py \
    --logtostderr \
    --pipeline_config_path=models/model/ssd_mobilenet_v1_coco.config \
    --checkpoint_dir=models/model/train \
    --eval_dir=models/model/eval
python export.py \
    --input_type image_tensor \
    --pipeline_config_path models/model/ssd_mobilenet_v1_coco.config \
    --trained_checkpoint_prefix models/model/train/model.ckpt-{checkpintdigit} \
    --output_directory rawmodels/sdd/
   