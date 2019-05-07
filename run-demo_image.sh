# dir=/home/ydwu/datasets/00-pictures/bank_datasets
dir=/home/ydwu/datasets/00-pictures/car

for file in $dir/*;
do
    echo $file
    CUDA_VISIBLE_DEVICES="" python demo_image.py --test_datasets=$file
done
