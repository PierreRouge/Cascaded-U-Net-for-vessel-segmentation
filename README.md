# Cascaded U-Net for vessel segmentation
by [Pierre Rougé](https://github.com/PierreRouge), Nicolas Passat, [Odyssée Merveille](http://www.odyssee-merveille.com/)

Code associated with the paper [Cascaded Multitask U-Net using topological loss for vessel segmentation and centerline extraction](https://arxiv.org/abs/2307.11603).

<p align="center"><img src="https://github.com/PierreRouge/Cascaded-U-Net-for-vessel-segmentation/blob/main/assets/architecture.png" alt="drawing" width="500"/>
</p>

### Installation

Install environment with environment.yml

```shell
conda env create -f environment.yml
```

### Usage

1. Clone the repository

   ```shell
   git clone https://github.com/PierreRouge/Cascaded-U-Net-for-vessel-segmentation.git
   cd Cascaded-U-Net-for-vessel-segmentation
   ```

2.  Put the MRA Images in data/Images,  the segmentations GTs in data/GT and the skeletons GTs in data/Skeletons

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
 	3. To train cascaded U-Net : put the pretrained weights in respectively pretrained_weights/segmentation and pretrained_weights/skeletonization and run :
 	
 	```shell
 	cd train
 	python train_cascaded_unet.py
 	```
### Visual results

<p align="center"><img src="https://github.com/PierreRouge/Cascaded-U-Net-for-vessel-segmentation/blob/main/assets/results.png" alt="drawing" width="500"/>
</p>

### Future implementations

We are currently working to implement several state-of-the-art methods for vascular segmentation !

### Citation

If you use this repository please consider citing :

```shell
@article{rouge2023cascaded,
      title={Cascaded multitask U-Net using topological loss for vessel segmentation and centerline extraction},
      author={Roug{\'e}, Pierre and Passat, Nicolas and Merveille, Odyss{\'e}e},
      journal={arXiv preprint arXiv:2307.11603},
      year={2023}
   }
```

### Contact

Pierre Rougé : pierre.rouge@creatis.insa-lyon.fr

