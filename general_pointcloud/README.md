# General Point Cloud processing
This floder contains the execution results for pointnet++ & pointnet, with code from [https://github.com/yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch). Based on this code, options to start from a checkpoint and to generate accuracy variation curves during the training process have been added. Test runs for classification (ModelNet10/40), Part Segmentation (ShapeNet), and Semantic Segmentation (S3DIS) have been conducted to obtain accuracy and accuracy variation curves. For the training and testing of classification, ‘classification.ipynb’ was created and exeData Preparation



# Data of Pointnet&point++/train_classification.ipynb
Based on the original source code, added the option to continue from a checkpoint and to output images showing changes in precision. After converting to a Jupyter file, more comments were added, allowing for a clearer understanding of the program's logic.

Download alignment ModelNet [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in Pointnet&point++/data/modelnet40_normal_resampled/.