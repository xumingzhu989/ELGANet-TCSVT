# Superpixel Segmentation With Edge Guided Local-Global Attention Network

This is a PyTorch implementation of the superpixel segmentation network introduced in our TCSVT paper.



## Prerequisites

During test, we make use of the component connection method in [SSN](https://github.com/NVlabs/ssn_superpixels) to enforce the connectivity 
in superpixels. The code has been included in ```/third_paty/cython```. To compile it:

 ```
cd third_party/cython/
python setup.py install --user
cd ../..
 ```



## Data preparation 

To generate training and test dataset, please first download the data from the original [BSDS500 dataset](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_full.tgz), 
and extract it to  ```<BSDS_DIR>```. Then, run 

```
cd data_preprocessing
python pre_process_bsd500.py --dataset=<BSDS_DIR> --dump_root=<DUMP_DIR>
python pre_process_bsd500_ori_sz.py --dataset=<BSDS_DIR> --dump_root=<DUMP_DIR>
cd ..
```

The code will generate three folders under the ```<DUMP_DIR>```, named as ```/train```, ```/val```, and ```/test```, and three ```.txt``` files 
record the absolute path of the images, named as ```train.txt```, ```val.txt```, and ```test.txt```.



## Training

Once the data is prepared, we should be able to train the model by running the following command

```
python main.py --data=<DUMP_DIR> --savepath=<CKPT_LOG_DIR>
```

if we wish to continue a training process or fine-tune from a pre-trained model, we can run 

```
python main.py --data=<DUMP_DIR> --savepath=<CKPT_LOG_DIR> --pretrained=<PATH_TO_THE_CKPT> 
```

The code will start from the recorded status, which includes the optimizer status and epoch number. 



## Evaluation

Following [FCN](https://github.com/fuy34/superpixel_fcn), we use the code from [superpixel benchmark](https://github.com/davidstutz/superpixel-benchmark) for superpixel evaluation. 
A detailed  [instruction](https://github.com/davidstutz/superpixel-benchmark/blob/master/docs/BUILDING.md) is available in the repository, please

(1) download the code and build it accordingly;

(2) edit the variables ```$SUPERPIXELS```, ```IMG_PATH``` , ```GT_PATH``` and ```SEG_PATH``` (output path) in ```/eval_spixel/my_eval.sh```,
example:

```
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

```

(3)run 

```
cp /eval_spixel/my_eval.sh <path/to/the/benchmark>/examples/bash/
cd  <path/to/the/benchmark>/examples/
bash my_eval.sh
```

(4) set the ```our1l_res_path``` in ```plot_benchmark_curve.m```

example:

```
our1l_res_path ='/home/ELGANet/output/Ours/test_multiscale_enforce_connect/ELGANet_nSpixel_'
n_set = length(num_list);
Ours = zeros(n_set,5);
for i=1:n_set
    load_path = [our1l_res_path  num2str(num_list(i)) '/map_csv/results.csv']; 
    Ours(i,:) = loadcsv(load_path);
end
```

(5) run the ```plot_benchmark_curve.m```, the ```ASA Score```, ```CO Score```, and ```BR-BP curve```  of our method should be shown on the screen.



## Demo

Specify the image path and use the pre-trained model to generate superpixels for images. 

```
python run_demo.py --data_dir=PATH_TO_IMAGE_DIR --output=./demo 
```

The results will be generate in a new folder under ```/demo``` called ```spixel_viz```.



## Citation

If it helps your research, please use the information below to cite our work, thank you.

```
@ARTICLE{ELGANet,
  author={Xu, Mingzhu and Sun, Zhengyu and Hu, Yijun and Tang, Haoyu and Hu, Yupeng and Song, Xuemeng and Nie, Liqiang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Superpixel Segmentation With Edge Guided Local-Global Attention Network}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Image edge detection;Feature extraction;Convolution;Training;Semantics;Object detection;Circuits and systems;Visualization;Data mining;Iterative methods;Superpixel segmentation;Edge enhancement;Local-Global context},
  doi={10.1109/TCSVT.2025.3587485}}

```

