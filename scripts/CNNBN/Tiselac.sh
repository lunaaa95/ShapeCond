dataset='Tiselac'
model='CNNBN'
DIR="logs/$model/$dataset"
if [ ! -d "$DIR" ]; then
  mkdir -p "$DIR"
  echo "文件夹已创建：$DIR"
fi

device=1
num_shapelets=20

# # rand teacher
# scond=-1
# for ipc in 1 10 20 50; do
  # python main.py --dataset $dataset --model $model --ipc $ipc --transform_steps 1 --num_shapelets $num_shapelets --scond $scond --re_epochs 2000 --lr_r 0.2 --norm 1 --device $device --inputaug raw --ttrain_epochs 100 --pre_shapelet_discovery 1 --lr_stu 1e-3 --strain_epochs 200 --lr_teacher 1e-4 > logs/$model/$dataset/$dataset\_scond=$scond\_ipc=$ipc.log
# done

# # -------------------------------------------------------------------------------------------------------
# # -------------------------------------------------------------------------------------------------------------

# model reverse
scond=0
for ipc in 1 10 20 50; do
  python main.py --imbalance_ratio 1.0 --dataset $dataset --model $model --ipc $ipc --num_shapelets $num_shapelets --transform_steps 1 --scond $scond --re_epochs 2000 --lr_r 0.2 --norm 1 --device $device --inputaug raw --ttrain_epochs 100 --pre_shapelet_discovery 1 --lr_stu 1e-3 --strain_epochs 100 --lr_teacher 1e-4 > 
done

# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------

# shapecond

## discovery
scond=1
ipc=1
num_shapelets=20
python main.py --num_processes 64 --dataset $dataset --model $model --ipc $ipc --fast 1 --obs_window_size 3 --mipc 1500 --num_shapelets $num_shapelets --scond $scond --re_epochs 1000 --lr_r 0.2 --norm 1 --device $device --inputaug raw --ttrain_epochs 100 --pre_shapelet_discovery 1 --lr_stu 1e-2 --strain_epochs 100 --lr_teacher 1e-5 > logs/$model/$dataset/$dataset\_discovery.log

##
for ipc in 1 10 20 50
do
  python main.py --imbalance_ratio 1.0 --dataset $dataset --model $model --ipc $ipc --num_shapelets $num_shapelets --transform_steps 1 --scond $scond --re_epochs 2000 --lr_r 0.2 --norm 1 --device $device --inputaug raw --ttrain_epochs 100 --pre_shapelet_discovery 1 --lr_stu 1e-3 --strain_epochs 100 --lr_teacher 1e-5 > logs/$model/$dataset/$dataset\_ipc=$ipc.log
done





