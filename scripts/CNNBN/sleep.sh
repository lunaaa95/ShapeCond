dataset='sleep'
model='CNNBN'
DIR="logs/$model/$dataset"
# 检查文件夹是否存在
if [ ! -d "$DIR" ]; then
  # 如果文件夹不存在，则创建它
  mkdir -p "$DIR"
  echo "文件夹已创建：$DIR"
fi

num_shapelets=20
aug="raw"



for scond in -1 0; do
  teacher_pretrain=0
  for ipc in 10 50; do
    python main.py --teacher_pretrain $teacher_pretrain --imbalance_ratio 1.0 --dataset $dataset --model $model --ipc $ipc --num_shapelets $num_shapelets --scond $scond --re_epochs 2000 --lr_r 0.15 --norm 1 --device $device --inputaug $aug --ttrain_epochs 150 --pre_shapelet_discovery 1 --lr_stu 5e-5 --strain_epochs 100 --lr_teacher 1e-5 > logs/$model/$dataset/$dataset\_scond=$scond\_ipc=$ipc\_$aug.log
    teacher_pretrain=1
  done
done

# shapecond
## discovery
    python main.py --num_processes 64 --mipc 500 --fast 1 --obs_window_size 50 --fit_steps 5 --transform_steps 2 --teacher_pretrain 1 --imbalance_ratio 1.0 --dataset $dataset --model $model --ipc $ipc --num_shapelets $num_shapelets --scond $scond --re_epochs 2000 --lr_r 0.15 --norm 1 --device $device --inputaug $aug --ttrain_epochs 150 --pre_shapelet_discovery 1 --lr_stu 5e-5 --strain_epochs 100 --lr_teacher 1e-5 > logs/$model/$dataset/$dataset\_scond=$scond\_ipc=$ipc\_$aug.log
## shapecond

ipc=10
lr_stu=2e-4
num_shapelets=20
python main.py --num_processes 64 --fast 1 --obs_window_size 50 --fit_steps 5 --transform_steps 2 --teacher_pretrain 1 --imbalance_ratio 1.0 --dataset $dataset --model $model --ipc $ipc --num_shapelets $num_shapelets --scond $scond --re_epochs 2000 --lr_r 0.15 --norm 1 --device $device --inputaug $aug --ttrain_epochs 150 --pre_shapelet_discovery 1 --lr_stu $lr_stu --strain_epochs 100 --lr_teacher 1e-5 > logs/$model/$dataset/aug/$dataset\_scond=$scond\_ipc=$ipc\_aug=$aug.log

ipc=50
lr_stu=2e-4
num_shapelets=5
python main.py --num_processes 64 --fast 1 --obs_window_size 50 --fit_steps 5 --transform_steps 2 --teacher_pretrain 0 --imbalance_ratio 1.0 --dataset $dataset --model $model --ipc $ipc --num_shapelets $num_shapelets --scond $scond --re_epochs 2000 --lr_r 0.15 --norm 1 --device $device --inputaug $aug --ttrain_epochs 150 --pre_shapelet_discovery 1 --lr_stu $lr_stu --strain_epochs 100 --lr_teacher 1e-5 > logs/$model/$dataset/aug/$dataset\_scond=$scond\_ipc=$ipc\_aug=$aug.log
