#OUTPUT_PATH="experiments/results"
OUTPUT_PATH="/media/hdd/xiaohao/experiments_ma_7scene/results_noloop"

#DATASET_PATH="dataset/Replica"
DATASET_PATH="/media/hdd/xiaohao/data/7Scene_ICP/"

str_pad() {

  local pad_length="$1" pad_string="$2" pad_type="$3"
  local pad length llength offset rlength

  pad="$(eval "printf '%0.${#pad_string}s' '${pad_string}'{1..$pad_length}")"
  pad="${pad:0:$pad_length}"

  if [[ "$pad_type" == "left" ]]; then

    while read line; do
      line="${line:0:$pad_length}"
      length="$(( pad_length - ${#line} ))"
      echo -n "${pad:0:$length}$line"
    done

  elif [[ "$pad_type" == "both" ]]; then

    while read line; do
      line="${line:0:$pad_length}"
      length="$(( pad_length - ${#line} ))"
      llength="$(( length / 2 ))"
      offset="$(( llength + ${#line} ))"
      rlength="$(( llength + (length % 2) ))"
      echo -n "${pad:0:$llength}$line${pad:$offset:$rlength}"
    done

  else

    while read line; do
      line="${line:0:$pad_length}"
      length="$(( pad_length - ${#line} ))"
      echo -n "$line${pad:${#line}:$length}"
    done

  fi
}

run_()
{
    local datasets=$1
    local config=$2
    local result_txt=$3
    local keyframe_th=$4
    local knn_maxd=$5
    local overlapped_th=$6
    local max_correspondence_distance=$7
    local trackable_opacity_th=$8
    local overlapped_th2=$9
    local downsample_rate=${10}
    local cuda_device=${11}  # Added parameter for CUDA device
    local group=${12}
    echo "run datasets: $datasets on CUDA device $cuda_device"
     python -W ignore mac_ego.py --dataset_path $datasets\
                                    --config $config\
                                    --output_path "$OUTPUT_PATH$group"\
                                    --keyframe_th $keyframe_th\
                                    --knn_maxd $knn_maxd\
                                    --overlapped_th $overlapped_th\
                                    --max_correspondence_distance $max_correspondence_distance\
                                    --trackable_opacity_th $trackable_opacity_th\
                                    --overlapped_th2 $overlapped_th2\
                                    --downsample_rate $downsample_rate\
                                    --cuda $cuda_device\
                                    --mu 1\
                                    --noise 0\
                                    --lc_freq 60\
                                    --post_opt 0\
				    --save_results #&
    # Use & to run the process in the background
}


run_cpslam()
{
    local result_txt=$1
    local keyframe_th=$2
    local knn_maxd=$3
    local overlapped_th=$4
    local max_correspondence_distance=$5
    local trackable_opacity_th=$6
    local overlapped_th2=$7
    local downsample_rate=$8
    local device=0

    # Pass datasets without extra single quotes to prevent path errors
    #chess 1~3
    datasets="$DATASET_PATH/chess/seq-03;$DATASET_PATH/chess/seq-01;$DATASET_PATH/chess/seq-02"
    run_ "$datasets" "configs/7Scene/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate $device "/chess_1"
    # chess 4~6
    datasets="$DATASET_PATH/chess/seq-05;$DATASET_PATH/chess/seq-04;$DATASET_PATH/chess/seq-06"
    run_ "$datasets" "configs/7Scene/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate $device "/chess_2"
    # chess 5,3,1
    datasets="$DATASET_PATH/chess/seq-05;$DATASET_PATH/chess/seq-03;$DATASET_PATH/chess/seq-01"
    run_ "$datasets" "configs/7Scene/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate $device "/chess_3"

    #fire
    datasets="$DATASET_PATH/fire/seq-01;$DATASET_PATH/fire/seq-02;$DATASET_PATH/fire/seq-03;$DATASET_PATH/fire/seq-04"
    run_ "$datasets" "configs/7Scene/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate $device "/fire"

    #heads
    datasets="$DATASET_PATH/heads/seq-02;$DATASET_PATH/heads/seq-01"
    run_ "$datasets" "configs/7Scene/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate $device "/heads"

    #office 1~3
    datasets="$DATASET_PATH/office/seq-01;$DATASET_PATH/office/seq-02;$DATASET_PATH/office/seq-03"
    run_ "$datasets" "configs/7Scene/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate $device "/office_1"
    #office 4~6
    datasets="$DATASET_PATH/office/seq-04;$DATASET_PATH/office/seq-06;$DATASET_PATH/office/seq-05"
    run_ "$datasets" "configs/7Scene/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate $device "/office_2"
    #office 7~10
    datasets="$DATASET_PATH/office/seq-08;$DATASET_PATH/office/seq-10;$DATASET_PATH/office/seq-09;$DATASET_PATH/office/seq-07"
    run_ "$datasets" "configs/7Scene/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate $device "/office_3"
    #office 1,2,4
    datasets="$DATASET_PATH/office/seq-04;$DATASET_PATH/office/seq-01;$DATASET_PATH/office/seq-02"
    run_ "$datasets" "configs/7Scene/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate $device "/office_4"

    #stairs 1~3
    datasets="$DATASET_PATH/stairs/seq-01;$DATASET_PATH/stairs/seq-02;$DATASET_PATH/stairs/seq-03"
    run_ "$datasets" "configs/7Scene/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate $device "/stairs_1"
    stairs 4~6
    datasets="$DATASET_PATH/stairs/seq-04;$DATASET_PATH/stairs/seq-05;$DATASET_PATH/stairs/seq-06"
    run_ "$datasets" "configs/7Scene/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate $device "/stairs_2"
    #stairs 2,4,1,5
    datasets="$DATASET_PATH/stairs/seq-02;$DATASET_PATH/stairs/seq-04;$DATASET_PATH/stairs/seq-01;$DATASET_PATH/stairs/seq-05"
    run_ "$datasets" "configs/7Scene/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate $device "/stairs_3"

    #pumpkin 1~3
    datasets="$DATASET_PATH/pumpkin/seq-03;$DATASET_PATH/pumpkin/seq-01;$DATASET_PATH/pumpkin/seq-02"
    run_ "$datasets" "configs/7Scene/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate $device "/pumpkin_1"
    pumpkin 6~8
    datasets="$DATASET_PATH/pumpkin/seq-06;$DATASET_PATH/pumpkin/seq-08;$DATASET_PATH/pumpkin/seq-07"
    run_ "$datasets" "configs/7Scene/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate $device "/pumpkin_2"
    #pumpkin 3,6,1
    datasets="$DATASET_PATH/pumpkin/seq-06;$DATASET_PATH/pumpkin/seq-03;$DATASET_PATH/pumpkin/seq-01"
    run_ "$datasets" "configs/7Scene/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate $device "/pumpkin_3"

    #redkitchen 1~4
    datasets="$DATASET_PATH/redkitchen/seq-01;$DATASET_PATH/redkitchen/seq-03;$DATASET_PATH/redkitchen/seq-04;$DATASET_PATH/redkitchen/seq-02"
    run_ "$datasets" "configs/7Scene/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate $device "/redkitchen_1"
    #redkitchen 5~8
    datasets="$DATASET_PATH/redkitchen/seq-07;$DATASET_PATH/redkitchen/seq-08;$DATASET_PATH/redkitchen/seq-05;$DATASET_PATH/redkitchen/seq-06"
    run_ "$datasets" "configs/7Scene/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate $device "/redkitchen_2"
    #redkitchen 11~14
    datasets="$DATASET_PATH/redkitchen/seq-13;$DATASET_PATH/redkitchen/seq-12;$DATASET_PATH/redkitchen/seq-11;$DATASET_PATH/redkitchen/seq-14"
    run_ "$datasets" "configs/7Scene/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate $device "/redkitchen_3"
    #redkitchen 1,4,3
    datasets="$DATASET_PATH/redkitchen/seq-01;$DATASET_PATH/redkitchen/seq-03;$DATASET_PATH/redkitchen/seq-04"
    run_ "$datasets" "configs/7Scene/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate $device "/redkitchen_t"

    # Wait for all background processes to finish
    wait
}
# txt_file="re_init_ablation/default_3DGS.txt"
txt_file="dummy.txt"
str_pad 20 " " left <<< "FPS" > $txt_file
str_pad 15 " " left <<< "RMSE" >> $txt_file
str_pad 15 " " left <<< "train iter" >> $txt_file
str_pad 15 " " left <<< "kframes" >> $txt_file
str_pad 15 " " left <<< "gaussians_num" >> $txt_file
# str_pad 32 " " left <<< "Depth L1" >> $txt_file
str_pad 30 " " left <<< "PSNR" >> $txt_file
str_pad 15 " " left <<< "SSIM" >> $txt_file
str_pad 15 " " left <<< "LPIPS" >> $txt_file
echo "" >> $txt_file

overlapped_th=5e-4
# overlapped_th=1e-3 #for larger noise
max_correspondence_distance=0.02
knn_maxd=99999.0

trackable_opacity_th=0.05
overlapped_th2=5e-5
downsample_rate=5
# keyframe_th=0.7
keyframe_th=0.7


run_cpslam $txt_file $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance \
$trackable_opacity_th $overlapped_th2 $downsample_rate
