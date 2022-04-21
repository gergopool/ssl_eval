# Evaluation for Self-Supervised Image Classification in PyTorch

![Python](https://img.shields.io/badge/python-3.8-blue)
![PyTorch](https://img.shields.io/badge/framework-pytorch-orange)

This modul is a handful tool to evaluate your self-supervised algorithm quickly with either linear evaluation or knn. The linear evaluation won't give you the official accuracies that your network can achieve, however, it gives a very good lower bound in a couple of minutes, instead of hours.

To give you an example, the linear evaluation of SimSiam's network achieves a 68% accuracy in 5 hours, while this code achieves 67% in 10 minutes with the same setup.

## :question: How

For accurate offline evaluation researchers use random crops of images. In contrast, this repository takes only a few crops of images, saves the generated embeddings to RAM and quickly iterates that with a large batch size and apex's LARC optimizer.

## :scientist: Target audience

This modul is generally made for researchers working with Imagenet, therefore, the evalautor was designed for a multi-gpu setup with a large amount of RAM provided (200GB+). It's because the evaluator saves all training embeddings to RAM for quick access.

## :electric_plug: Usage

### Define instance

First, build your encoder model in either a single-gpu or a multi-gpu setup. Then, 
create an evalautor instance by
```python
evaluator = Evaluator(model, dataset="imagenet", root='/data/imagenet/', n_views=2, batch_size=256)
```

| Arg | Description |
| --- | ----------- |
| model | The encoder model that maps the input image to a cnn_dim representation. The model doesn't need to be freezed or be in eval mode. |
| dataset | Name of the dataset. Choose from `'imagenet', 'tiny_imagenet', 'cifar10', 'cifar100'` . *Note: The tiny imagenet needs to be structured as imagenet and the evaluation uses the validation folder. Preprocessing is also identical to imagenet's.*|
| root | Path to your dataset |
| n_views | Optional. Number of augmentations, number of views you desire to get from each image example. Default is 1. |
| batch_size | Optional. The batch size used for iterating over images when generating images, per gpu. Default is 256. |
| verbose | Optional. Verbosity. Default is True. |

 ### Generate embeddings

```python
train_z, train_y, val_z, val_y = evaluator.generate_embeddings()
embs = (train_z, train_y, val_z, val_y)
```

| Return value | Description |
| --- | ----------- |
| train_z | NxDxV tensor, where N is the number of samples, D is the cnn_dim and V is the number of views. Note that these are half precision embeddings. |
| train_y | Tensor of labels with length of N |
| val_z | Same as train_z, but with validation set. |
| val_y | Same as train_y, but with validation set. |


### Run linear evaluation
```python
top1_acc = evaluator.linear_eval(batch_size=256)
```
Runs a linear evalaution on the generated embeddings. It uses decreases the learning rate when platues and stop with early stopping if necessary.

| Arg | Description |
| --- | ----------- |
| embs | Optional. Tuple of (z,y) tensors described above. If None, it will use the ones generated the last time. |
| epochs | Optional. Maximum number of epochs to train (it can still stop with early stopping). Default is 100. |
| batch_size | Optional. Batch size used for iterating over the embeddings. Default is 256. |
| lr | Optional. Learning rate. 0.1 by default. |
| warm_start | Optional. If True, it loads the weights from the last training. Default is False. |

| Return value | Description |
| --- | ----------- |
| top1_acc | Top1 accuracy achieved on the validation set. |

*Note: Nvidia's apex Larc optimizer used.*

### KNN
```python
top1_accs = evaluator.knn([1,5,20])
```
| Arg | Description |
| --- | ----------- |
| embs | Optional. Tuple of (z,y) tensors described above. If None, it will use the ones generated the last time. |
| ks | Optional. The K values we desire to run the KNN with. Can be either integer or list of integers. 1 by default. |

| Return value | Description |
| --- | ----------- |
| top1_accs | Top1 accuracies to the K values given, respectively. |


## Contact

For any inquiries please contact me at
gergopool[at]gmail.com