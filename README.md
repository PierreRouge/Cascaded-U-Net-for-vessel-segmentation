# Cascaded-U-Net-for-vessel-segmentation
Code associated with the paper Cascaded Multitask U-Net using topological loss for vessel segmentation and centerline extraction.



### Installation



### Usage

1. Clone the repository

   ```shell
   git clone https://github.com/PierreRouge/Cascaded-U-Net-for-vessel-segmentation.git
   cd Cascaded-U-Net-for-vessel-segmentation
   ```

2.  Put the MRA Images in data/Images and the GTs in data/GT

3. Use one of the training function 

   1. To pretrain segmentation network

 	```shell
 	  cd code
 	  python train_segmentation.py
 	  ```

   2. To pretrain skeletonization network

 	```shell
 	  cd code
 	  python train_skeletonization.py
 	  ```
 	3. To train cascaded U-Net
```shell
cd code
python train_cascaded_unet.py
```

​    



