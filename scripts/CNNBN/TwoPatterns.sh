dataset='TwoPatterns'
model='CNNBN'
DIR="logs/$model/$dataset"
if [ ! -d "$DIR" ]; then
  mkdir -p "$DIR"
  echo "文件夹已创建：$DIR"
fi

num_shapelets=20
device=3

# model reverse
scond=0
for ipc in 1 10; do
  python main.py --imbalance_ratio 1.0 --obs_window_size 20 --fast 0 --dataset $dataset --model $model --ipc $ipc --num_shapelets $num_shapelets --scond $scond --re_epochs 2000 --lr_r 0.2 --norm 1 --device $device --inputaug raw --ttrain_epochs 400 --pre_shapelet_discovery 1 --lr_stu 1e-3 --strain_epochs 300 --lr_teacher 1e-3 > logs/$model/$dataset/$dataset\_scond=$scond\_ipc=$ipc.log
done

# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------

# shapecond
scond=1
num_shapelets=20
for ipc in 1 10; do
    python main.py --imbalance_ratio 1.0 --obs_window_size 20 --fast 0 --dataset $dataset --model $model --ipc $ipc --num_shapelets $num_shapelets --scond $scond --re_epochs 2000 --lr_r 0.2 --norm 1 --device $device --inputaug raw --ttrain_epochs 400 --pre_shapelet_discovery 1 --lr_stu 1e-3 --strain_epochs 10 --lr_teacher 1e-3 > logs/$model/$dataset/$dataset\_scond=$scond\_ipc=$ipc.log
done
