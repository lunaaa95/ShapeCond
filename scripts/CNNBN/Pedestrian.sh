dataset='Pedestrian'
model='CNNBN'
DIR="logs/$model/$dataset"
if [ ! -d "$DIR" ]; then
  mkdir -p "$DIR"
  echo "文件夹已创建：$DIR"
fi



# rand teacher
scond=-1
for ipc in 1; do
  python main.py --dataset $dataset --model $model --ipc $ipc --transform_steps 1 --num_shapelets $num_shapelets --scond $scond --re_epochs 2000 --lr_r 0.2 --norm 1 --device $device --inputaug raw --ttrain_epochs 100 --pre_shapelet_discovery 1 --lr_stu 1e-3 --strain_epochs 200 --lr_teacher 1e-4 #> logs/$model/$dataset/$dataset\_scond=$scond\_ipc=$ipc.log
done

 # # -------------------------------------------------------------------------------------------------------
 # # -------------------------------------------------------------------------------------------------------------

# model reverse
scond=0
ipc=10
lr_stu=1e-2
for lr_stu in 1; do
  python main.py --train_bsz 256 --imbalance_ratio 1.0 --dataset $dataset --model $model --ipc $ipc --num_shapelets $num_shapelets --transform_steps 1 --scond $scond --re_epochs 2000 --lr_r 0.2 --norm 1 --device $device --inputaug raw --ttrain_epochs 50 --pre_shapelet_discovery 1 --lr_stu $lr_stu --strain_epochs 50 --lr_teacher 1e-2 > logs/$model/$dataset/$dataset\_scond=$scond\_ipc=$ipc\_lr_stu=$lr_stu.log &
done

# # ----------------------------------------------------------------------------------------------------------
# # ----------------------------------------------------------------------------------------------------------

# # shapecond

# # discovery
# # spl search para: [3 7 1 1]
# scond=1
# ipc=10
# num_shapelets=20
# python main.py --imbalance_ratio 0.0 --num_processes 64 --dataset $dataset --model $model --ipc $ipc --fast 1 --obs_window_size 2 --mipc 1000 --num_shapelets $num_shapelets --scond $scond --re_epochs 1000 --lr_r 0.2 --norm 1 --device $device --inputaug raw --ttrain_epochs 50 --pre_shapelet_discovery 1 --lr_stu 1e-2 --strain_epochs 50 --lr_teacher 1e-3 > logs/$model/$dataset/$dataset\_discovery.log


device=0
scond=1
num_shapelets=15
teacher_pretrain=0
for ipc in 1 5 10 20; do
  CUDA_VISIBLE_DEVICES=2 python main.py --teacher_pretrain $teacher_pretrain --imbalance_ratio 1.0 --dataset $dataset --model $model --ipc $ipc --num_shapelets $num_shapelets --transform_steps 1 --scond 1 --re_epochs 1500 --lr_r 0.2 --norm 1 --device $device --inputaug raw --ttrain_epochs 100 --pre_shapelet_discovery 1 --lr_stu 1e-3 --strain_epochs 100 --lr_teacher 1e-5 > logs/$model/$dataset/$dataset\_scond=$scond\_ipc=$ipc\_sl=$num_shapelets.log
  teacher_pretrain=1
done

