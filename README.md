# Cascaded U-Net for vessel segmentation
by Pierre Rougé, Nicolas Passat, Odyssée Merveille

Code associated with the paper Cascaded Multitask U-Net using topological loss for vessel segmentation and centerline extraction.

<p align="center"><img src="https://github.com/PierreRouge/Cascaded-U-Net-for-vessel-segmentation/blob/main/assets/architecture.png" alt="drawing" width="500"/>
</p>



### Installation

Install envrionment with environment.yml

```shell
conda env create -f environment.yml
```

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
 	  cd train
 	  python train_segmentation.py
 	  ```

   2. To pretrain skeletonization network

 	```shell
 	  cd train
 	  python train_skeletonization.py
 	  ```
 	3. To train cascaded U-Net
 	```shell
 	cd train
 	python train_cascaded_unet.py
 	```

### Citation

### Contact

Pierre Rougé : pierre.rouge@creatis.insa-lyon.fr

