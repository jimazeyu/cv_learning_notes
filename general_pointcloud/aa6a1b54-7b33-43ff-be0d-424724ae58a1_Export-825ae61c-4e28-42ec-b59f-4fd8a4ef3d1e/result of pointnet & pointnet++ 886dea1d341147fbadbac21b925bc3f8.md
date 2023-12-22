# result of pointnet & pointnet++

This document contains the execution results for pointnet++ & pointnet, with code from [https://github.com/yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch). Based on this code, options to start from a checkpoint and to generate accuracy variation curves during the training process have been added. Test runs for classification (ModelNet10/40), Part Segmentation (ShapeNet), and Semantic Segmentation (S3DIS) have been conducted to obtain accuracy and accuracy variation curves. For the training and testing of classification, ‘classification.ipynb’ was created and executed.

# Classification

The classification algorithm can use either ModelNet10 or ModelNet40. It allows the choice of whether to use the normal features of points (—use_normals) and the sampling method, such as uniform sampling (—use_uniform_sample). I have chosen several options for training.

| ModelNet | Method | Test Instance Accuracy | Class Accuracy |
| --- | --- | --- | --- |
| 40 | basic | 0.924279 | 0.891787 |
| 40 | with normal features | 0.891787 | 0.891787 |
| 40 | with uniform sampling | 0.922276 | 0.889164 |
| 10 | basic | 0.947198 | 0.945434 |

## Accuracy variation curves(According to the order in the table above)

![pointnet2_ssg without normal features.png](result%20of%20pointnet%20&%20pointnet++%20886dea1d341147fbadbac21b925bc3f8/pointnet2_ssg_without_normal_features.png)

![pointnet2_ssg with normal features.png](result%20of%20pointnet%20&%20pointnet++%20886dea1d341147fbadbac21b925bc3f8/pointnet2_ssg_with_normal_features.png)

![pointnet2_ssg with uniform sampling.png](result%20of%20pointnet%20&%20pointnet++%20886dea1d341147fbadbac21b925bc3f8/pointnet2_ssg_with_uniform_sampling.png)

![ModelNet10 pointnet2_ssg without normal features .png](result%20of%20pointnet%20&%20pointnet++%20886dea1d341147fbadbac21b925bc3f8/ModelNet10_pointnet2_ssg_without_normal_features_.png)

# **Part Segmentation (ShapeNet)**

eval mIoU of Airplane       0.822105
eval mIoU of Bag            0.773953
eval mIoU of Cap            0.823337
eval mIoU of Car            0.788797
eval mIoU of Chair          0.908672
eval mIoU of Earphone       0.774496
eval mIoU of Guitar         0.911501
eval mIoU of Knife          0.876008
eval mIoU of Lamp           0.838147
eval mIoU of Laptop         0.955814
eval mIoU of Motorbike      0.718066
eval mIoU of Mug            0.950532
eval mIoU of Pistol         0.809336
eval mIoU of Rocket         0.600681
eval mIoU of Skateboard     0.766923
eval mIoU of Table          0.824887
Accuracy is: 0.94289
Class avg accuracy is: 0.86904
Class avg mIOU is: 0.82145
Inctance avg mIOU is: 0.85188

![Part Segmentation (ShapeNet).png](result%20of%20pointnet%20&%20pointnet++%20886dea1d341147fbadbac21b925bc3f8/Part_Segmentation_(ShapeNet).png)

# **Semantic Segmentation (S3DIS)**

class ceiling , IoU: 0.900
class floor , IoU: 0.979
class wall , IoU: 0.734
class beam , IoU: 0.000
class column , IoU: 0.085
class window , IoU: 0.590
class door , IoU: 0.088
class table , IoU: 0.690
class chair , IoU: 0.77
class sofa , IoU: 0.482
class bookcase , IoU: 0.619
class board , IoU: 0.579
class clutter , IoU: 0.438

eval point avg class IoU: 0.535538
eval whole scene point avg class acc: 0.619653
eval whole scene point accuracy: 0.826357

**batch size=16**

![Semantic Segmentation batchsize=16.png](result%20of%20pointnet%20&%20pointnet++%20886dea1d341147fbadbac21b925bc3f8/Semantic_Segmentation_batchsize16.png)

**batch size=32**

![Semantic Segmentation batchsize=32.png](result%20of%20pointnet%20&%20pointnet++%20886dea1d341147fbadbac21b925bc3f8/Semantic_Segmentation_batchsize32.png)