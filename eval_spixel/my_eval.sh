SUPERPIXELS=("96" "216" "384" "600" "864" "1176" "1536" "1944")  
#img and seg and gt's names must be same
IMG_PATH=/home/BSR/dump_path/test
GT_PATH=/home/BSR/dump_path/test/map_csv
SEG_PATH=/home/ELGANet/output/Ours

for SUPERPIXEL in "${SUPERPIXELS[@]}"
do
   #--vis create restruction_map
   echo $SUPERPIXEL
       ../bin/eval_summary_cli $SEG_PATH/test_multiscale_enforce_connect/ELGANet_nSpixel_${SUPERPIXEL}/map_csv  $IMG_PATH $GT_PATH  
done
