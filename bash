
python download-scannet.py -o /home/shared/datasets/ScanNet 
python ScanNet/download-scannet.py --rob_task_data -o /home/shared/datasets/ScanNet
python download-scannet.py --rob_task_data -o /home/shared/datasets/ScanNet
cd /home/shared/datasets/ScanNet/



cd data_splits/



python downsample_color_imgs.py 

cd raw_data_KITTI/

cd KITTI/

gedit gather_raw_images.py

cd ScanNet/

python assign_scenes_to_split.py /home/shared/datasets/ScanNet/scenes_all

python assign_scenes_to_split.py /home/shared/datasets/ScanNet






python assign_scenes_to_split.py 


cd rob_devkit/depth/ScanNet/





python download-scannet.py -o /home/shared/datasets/ScanNet  --rob_task_data
python download-scannet.py python ScanNet/download-scannet.py --rob_task_data -o /home/shared/datasets/ScanNet/
python download-scannet.py --rob_task_data -o /home/shared/datasets/ScanNet/
bash unzip-and-remove-all-zipped-scenes.sh
python ScanNet/downsample_color_imgs.py
python downsample_color_imgs.py
python assign_scenes_to_split.py 
python compare_dataset.py 
python assign_scenes_to_split.py 


cd monodepth


python monodepth_main.py --mode train --model_name my_model --data_path /home/shared/datasets/kitti_raw/raw_data/ --filenames_file ~/Desktop/monodepth/utils/filenames/kitti_train_files_png.txt --log_directory log/ --output_directory output/ --num_epochs 2







cd rob_devkit/

cd depth/

cd KITTI/

python create_file_train_monodepth.py /home/shared/datasets/depth_kitti/depth/depth_single_img/
python create_file_train_monodepth.py -d /home/shared/datasets/depth_kitti/depth/depth_single_img/train/ -r /home/shared/datasets/kitti_raw/raw_data/ -t txt -o
python create_file_train_monodepth.py -d /home/shared/datasets/depth_kitti/depth/depth_single_img/train/ -r /home/shared/datasets/kitti_raw/raw_data/ -t txt -f





python monodepth_main.py --mode train --model_name my_model --data_path /home/shared/datasets/kitti_raw/raw_data/ --filenames_file ~/Desktop/monodepth/utils/filenames/kitti_train_files_png.txt --log_directory log/ --output_directory output/ --num_epochs 2 

python monodepth_main.py --mode train --model_name my_model --data_path /home/shared/datasets/kitti_raw/raw_data/ --filenames_file ~/Desktop/monodepth/utils/filenames/kitti_train_files_png.txt --log_directory log/ --output_directory output/ --num_threads 8




cd monodepth_wrong/

python monodepth_main.py --mode test --data_path /home/shared/datasets/kitti_raw/raw_data/ --filenames_file ~/Desktop/monodepth/utils/filenames/kitti_test_files_png.txt --checkpoint_path 
log/monodepth/model-181250

python monodepth_main.py --mode test --data_path /home/shared/datasets/kitti_raw/raw_data/ --filenames_file ~/Desktop/monodepth/utils/filenames/kitti_test_files_png.txt --checkpoint_path 
log/my_model/model-181250



cd monodepth test 

python monodepth_main.py --mode test --data_path /home/shared/datasets/kitti_raw/raw_data/ 
--filenames_file ~/Desktop/monodepth/utils/filenames/kitti_stereo_2015_test_files.txt --checkpoint_path log/my_model/model-181250

python monodepth_main.py --mode test --data_path /home/shared/datasets/kitti2015/ 
--filenames_file ~/Desktop/monodepth/utils/filenames/kitti_stereo_2015_test_files.txt 
--checkpoint_path log/my_model/model-181250

python monodepth_main.py --mode test --data_path /home/shared/datasets/kitti2015/ 
--filenames_file ~/Desktop/monodepth/utils/filenames/kitti_stereo_2015_test_files_png.txt 
--checkpoint_path log/my_model/model-181250

python monodepth_main.py --mode test --data_path /home/shared/datasets/kitti2015/stereo/data_scene_flow/ 
--filenames_file ~/Desktop/monodepth/utils/filenames/kitti_stereo_2015_testing_files_png.txt 
--checkpoint_path log/my_model/model-181250 --output_directory output








python gather_raw_images.py -d /home/shared/datasets/depth_kitti/depth/depth_single_img/train/ -r /home/shared/datasets/kitti_raw/raw_data/ -t txt -f
python create_file_train_monodepth.py -d /home/shared/datasets/depth_kitti/depth/depth_single_img/train/ -r /home/shared/datasets/kitti_raw/raw_data/ -t txt -f




