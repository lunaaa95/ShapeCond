dataset='har'
model='Transformer'
DIR="logs/$model/$dataset"
if [ ! -d "$DIR" ]; then
  mkdir -p "$DIR"
  echo "文件夹已创建：$DIR"
fi
num_shapelets=20


# model reverse
scond=0
# best setting: lr_lr=0.01, lr_stu=1e-3
lr_lr=0.01
lr_stu=1e-3
for ipc in 1 5 10; do
  python main.py --teacher_pretrain 1 --imbalance_ratio 1.0 --dataset $dataset --model $model --ipc $ipc --num_shapelets $num_shapelets --transform_steps 1 --scond $scond --re_epochs 2000 --lr_r $lr_lr --norm 1 --device $device --inputaug raw --ttrain_epochs 200 --pre_shapelet_discovery 1 --lr_stu $lr_stu --strain_epochs 200 --lr_teacher 1e-4 >> logs/$model/$dataset/$dataset\_scond=$scond\_lr_lr=$lr_lr\_lr_stu=$lr_stu\_ipc=$ipc.log 2>&1 &
  sleep 1
done

# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------

# shapecond
scond=1
num_shapelets=20
for ipc in 1 5 10; do
    python main.py --imbalance_ratio 1.0 --dataset $dataset --model $model --ipc $ipc --num_shapelets $num_shapelets --transform_steps 1 --scond $scond --re_epochs 2000 --lr_r 0.2 --norm 1 --device $device --inputaug raw --ttrain_epochs 100 --pre_shapelet_discovery 1 --lr_stu 1e-3 --strain_epochs 200 --lr_teacher 1e-4 > logs/$model/$dataset/$dataset\_scond=$scond\_ipc=$ipc.log
done

