
LR=1e-1

for LR in 1e-3
do

    CUDA_VISIBLE_DEVICES=1 \
    python main.py \
    --video_path data/UCF101/jpg \
    --annotation_path data/UCF101/annotation/ucf101_01.json \
    --result_path results/ucf101/finetune/default/lr${LR} \
    --dataset ucf101 \
    --n_classes 101 \
    --n_pretrain_classes 700 \
    --pretrain_path pretrained/r3d18_K_200ep.pth \
    --ft_begin_module fc \
    --model resnet \
    --model_depth 18 \
    --batch_size 128 \
    --n_threads 4 \
    --checkpoint 5 \
    --n_epochs 30 \

done