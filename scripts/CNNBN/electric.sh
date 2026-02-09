dataset='electric'
model='CNNBN'
DIR="logs/$model/$dataset"
# 检查文件夹是否存在
if [ ! -d "$DIR" ]; then
  # 如果文件夹不存在，则创建它
  mkdir -p "$DIR"
  echo "文件夹已创建：$DIR"
fi

num_shapelets=10
aug='raw'

# # rand teacher
# scond=-1
# for ipc in 1 5 10; do
  # python main.py --imbalance_ratio 1.0 --dataset $dataset --model $model --ipc $ipc --num_shapelets $num_shapelets --scond $scond --re_epochs 3000 --lr_r 0.25 --norm 1 --device $device --inputaug $aug --ttrain_epochs 400 --pre_shapelet_discovery 1 --lr_stu 1e-4 --strain_epochs 300 --lr_teacher 1e-4 > $DIR/$dataset\_scond=$scond\_ipc=$ipc.log
# done
# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------

# model reverse
scond=0
for ipc in 1 5 10; do
  python main.py --imbalance_ratio 1.0 --fast 0 --fit_steps 1 --transform_steps 1 --obs_window_size 20 --dataset $dataset --model $model --ipc $ipc --num_shapelets $num_shapelets --scond $scond --re_epochs 2000 --lr_r 0.35 --norm 1 --device $device --inputaug $aug --ttrain_epochs 100 --pre_shapelet_discovery 1 --lr_stu 5e-4 --strain_epochs 100 --lr_teacher 1e-4 > $DIR/$dataset\_scond=$scond\_ipc=$ipc.log
done

# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------

# shapecond
scond=1
for ipc in 1 5 10; do
  python main.py --imbalance_ratio 1.0 --fast 0 --fit_steps 1 --transform_steps 1 --obs_window_size 20 --dataset $dataset --model $model --ipc $ipc --num_shapelets $num_shapelets --scond $scond --re_epochs 2000 --lr_r 0.35 --norm 1 --device $device --inputaug $aug --ttrain_epochs 100 --pre_shapelet_discovery 1 --lr_stu 5e-4 --strain_epochs 100 --lr_teacher 1e-4 > $DIR/$dataset\_scond=$scond\_ipc=$ipc.log
done

# # shapecond + aug to improve performance. (optional)
# ipc=1
# aug=raw_slstrong
  # python main.py --imbalance_ratio 1.0 --fast 0 --fit_steps 1 --transform_steps 1 --obs_window_size 20 --dataset $dataset --model $model --ipc $ipc --num_shapelets $num_shapelets --scond $scond --re_epochs 2000 --lr_r 0.35 --norm 1 --device $device --inputaug $aug --ttrain_epochs 100 --pre_shapelet_discovery 1 --lr_stu 5e-4 --strain_epochs 100 --lr_teacher 1e-4 > $DIR/$dataset\_scond=$scond\_ipc=$ipc.log
