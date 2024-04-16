

## Environment
```shell
conda create -n JSALT python=3.11.5

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Encoder pre-training
```shell
python encoder_pretraining.py \
  --config_file spoter2/configs/basic_config_how2sign.yaml \
  --wandb_api_key <your_wandb_api_key> \
  --tags pre_training \
  --train_file <path/to>/H2S_train.h5 \
  --val_file <path/to>/How2Sign_val.h5 \
  --checkpoint_folder checkpoints
```

## Classification training
```shell
python classification_training.py \
  --config_file spoter2/configs/basic_config_wlasl.yaml \
  --wandb_api_key <your_wandb_api_key> \
  --tags classification_training \
  --train_file <path/to>/WLASL100_train_25fps.csv \
  --val_file <path/to>/WLASL100_val_25fps.csv \
  --num_classes 100 \
  --checkpoint <path/to/checkpoint.pth>
```