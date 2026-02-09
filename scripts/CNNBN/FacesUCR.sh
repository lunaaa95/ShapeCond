dataset='FacesUCR'
model='CNNBN'
DIR="logs/$model/$dataset"
# 检查文件夹是否存在
if [ ! -d "$DIR" ]; then
  # 如果文件夹不存在，则创建它
  mkdir -p "$DIR"
  echo "文件夹已创建：$DIR"
fi

num_shapelets=15

# # rand teacher
# scond=-1
# for ipc in 1 5 10; do
  # python main.py --dataset $dataset --model $model --ipc $ipc --transform_steps 1 --num_shapelets $num_shapelets --scond $scond --re_epochs 1000 --lr_r 0.2 --norm 1 --device $device --inputaug raw --ttrain_epochs 100 --pre_shapelet_discovery 1 --lr_stu 1e-3 --strain_epochs 100 --lr_teacher 1e-3 > logs/$model/$dataset/$dataset\_scond=$scond\_ipc=$ipc.log
# done

# # -------------------------------------------------------------------------------------------------------
# # -------------------------------------------------------------------------------------------------------------

# model reverse
scond=0
for ipc in 1 5 10; do
  python main.py --imbalance_ratio 1.0 --dataset $dataset --model $model --ipc $ipc --num_shapelets $num_shapelets --transform_steps 1 --scond $scond --re_epochs 2000 --lr_r 0.2 --norm 1 --device $device --inputaug raw --ttrain_epochs 100 --pre_shapelet_discovery 1 --lr_stu 1e-3 --strain_epochs 100 --lr_teacher 1e-3 > logs/$model/$dataset/$dataset\_scond=$scond\_ipc=$ipc.log
done

# # # ----------------------------------------------------------------------------------------------------------
# # # ----------------------------------------------------------------------------------------------------------

# shapecond
scond=1
num_shapelets=15
ipc=1
python main.py --imbalance_ratio 1.0 --dataset $dataset --model $model --ipc $ipc --num_shapelets $num_shapelets --transform_steps 1 --scond $scond --re_epochs 2000 --lr_r 0.2 --norm 1 --device $device --inputaug raw --ttrain_epochs 100 --pre_shapelet_discovery 1 --teacher_pretrain 0 --lr_stu 1e-3 --strain_epochs 100 --lr_teacher 1e-3 > logs/$model/$dataset/$dataset\_scond=$scond\_ipc=$ipc.log

num_shapelets=5
ipc=5
python main.py --imbalance_ratio 1.0 --dataset $dataset --model $model --ipc $ipc --num_shapelets $num_shapelets --transform_steps 1 --scond $scond --re_epochs 2000 --lr_r 0.2 --norm 1 --device $device --inputaug raw --ttrain_epochs 100 --pre_shapelet_discovery 1 --teacher_pretrain 0 --lr_stu 1e-3 --strain_epochs 100 --lr_teacher 1e-3 > logs/$model/$dataset/$dataset\_scond=$scond\_ipc=$ipc.log

num_shapelets=5
ipc=10
python main.py --imbalance_ratio 1.0 --dataset $dataset --model $model --ipc $ipc --num_shapelets $num_shapelets --transform_steps 1 --scond $scond --re_epochs 2000 --lr_r 0.2 --norm 1 --device $device --inputaug raw --ttrain_epochs 100 --pre_shapelet_discovery 1 --teacher_pretrain 1 --lr_stu 1e-3 --strain_epochs 100 --lr_teacher 1e-3 > logs/$model/$dataset/$dataset\_scond=$scond\_ipc=$ipc.log
