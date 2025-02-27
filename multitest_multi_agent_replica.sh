#OUTPUT_PATH="experiments/results"
OUTPUT_PATH="/media/hdd/xiaohao/experiments_ma_ulim/test"

#DATASET_PATH="dataset/Replica"
DATASET_PATH="/media/hdd/xiaohao/data/cpslam_data_icp/cp-slam"
# DATASET_PATH="/media/hdd/xiaohao/data/cpslam_14_1_0"


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
                                    --output_path $OUTPUT_PATH$group\
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
                                    --lc_freq 150\
                                    --post_opt 1\
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
    local device=2
    # Pass datasets without extra single quotes to prevent path errors
    datasets="$DATASET_PATH/Apart-0/apart_0_part1;$DATASET_PATH/Apart-0/apart_0_part2"
    run_ "$datasets" "configs/Replica/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate $device "/Apart-0"

    datasets="$DATASET_PATH/Apart-1/apart_1_part1;$DATASET_PATH/Apart-1/apart_1_part2"
    run_ "$datasets" "configs/Replica/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate $device "/Apart-1"

    datasets="$DATASET_PATH/Apart-2/apart_2_part2;$DATASET_PATH/Apart-2/apart_2_part1"
    run_ "$datasets" "configs/Replica/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate $device "/Apart-2"

    datasets="$DATASET_PATH/Office-0/office_0_part1;$DATASET_PATH/Office-0/office_0_part2"
    run_ "$datasets" "configs/Replica/caminfo.txt" $result_txt $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate $device "/Office-0"


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
# overlapped_th=1e-3 #For higher noise to improve efficiency
max_correspondence_distance=0.02
knn_maxd=99999.0

trackable_opacity_th=0.05
overlapped_th2=5e-5
downsample_rate=10
keyframe_th=0.7


run_cpslam $txt_file $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance \
$trackable_opacity_th $overlapped_th2 $downsample_rate
