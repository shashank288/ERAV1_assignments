# Session 8

## Introduction

This assignment compares different normalization techniques: **Batch Norm, Layer Norm** and **Group Norm**.

We are presented with a multiclass classification problem on the CIFAR10 dataset.

### Target
1. Accuracy > 70%
2. Number of Parameters < 50k
3. Epochs <= 20

Use of Residual Connection is also advised.

## Implementation

![Accuracy](../S8/Images/Architecture.png)

The above structure with two residual connections is used.

## Normalization Technique Comparison
_Note: We use GN with num_groups = 4_

### Metrics
|    | Train Acc | Test Acc | Train Loss | Test Loss |
|----|-----------|----------|------------|-----------|
| BN | 80.27     | 79.39    | 0.57       | 0.60      |
| GN | 76.18     | 74.84    | 0.68       | 0.72      |
| LN | 74.17     | 72.79    | 0.73       | 0.76      |

## Performance Curves
![Performance](../S8/Images/Performance_curves.png)

We see that the graphs portray BN > GN (4 groups) > LN consistently in all the training continues. We explore the reason for this in the next sections.

## Confusion Matrices

**Batch Norm | Group Norm | Layer Norm**
<div>
    <img src="https://github.com/Madhur-1/ERA-v1/assets/64495917/6cc20003-e120-4d4d-afbf-398512635fb6" width="325px" alt="image 1">
    <img src="https://github.com/Madhur-1/ERA-v1/assets/64495917/53d8861d-8b44-4e02-9788-d277cad72833" width="325px" alt="image 2">
    <img src="https://github.com/Madhur-1/ERA-v1/assets/64495917/615a69f9-35c3-4e3d-83bc-11e14dae36d1" width="325px" alt="image 3">
</div>


## Misclassified Images
**Batch Norm**

Total Incorrect Preds = 2061

![Batch_norm](../S8/Images/Batch_norm_missclassified.png)


**Group Norm**

Total Incorrect Preds = 2516

![GN](../S8/Images/Group_norm_missclassified.png)


**Layer Norm**

Total Incorrect Preds = 3139

![LN](../S8/Images/Layer_norm_missclassified.png)

We see that the misclassified images in all three models have classes very close to each other as misclassified. These misclassified images would be hard for a human to classify correctly too!

<br>

# Analysis
## Batch normalization
- Highest training and testing accuracy as it normalizes across the whole mini-batch per channel.

## Group normalization
- It performed worse than Batch Normalization and better than Layer Normalization.
- As the number of groups increases, it provides better results.

## Layer normalization
- It performed the worst as it normalizes each image across all channels.
- This is not suitable for image classifiers.

<br>