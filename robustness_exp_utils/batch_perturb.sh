DATASET_PATH="/media/hdd/xiaohao/data/cpslam_data_icp/cp-slam"

pt=14 # perturbation type
pd=0 #  perturb_dynamic

run_()
{
    local perturb_type=$1
    local perturb_severity=$2
    local frame_downsample=$3
    local perturb_dynamic=$4
    
    python -W ignore data_perturb.py --dataset_path $DATASET_PATH\
                                    --perturb_type $perturb_type\
                                    --perturb_severity $perturb_severity\
                                    --frame_downsample $frame_downsample\
                                    --perturb_dynamic $perturb_dynamic
    wait
}

run_disturb()
{
      ds=1
      for sev in 1 3 5; do
        EXPNAME="${pt}_${sev}_${pd}"
        print $EXPNAME
        run_ $pt $sev $ds $pd
        echo "${EXPNAME} done!"
      done

}

run_disturb