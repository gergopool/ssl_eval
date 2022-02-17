# Evaluation for Self-Supervised Image Classification in PyTorch

![Python](https://img.shields.io/badge/python-3.8-blue)
![PyTorch](https://img.shields.io/badge/framework-pytorch-orange)
![version](https://img.shields.io/badge/version-beta-yellowgreen)

This modul is a handful tool to evaluate your self-supervised algorithm quickly with either linear evaluation or knn. The linear evaluation won't give you the official accuracies that your network can achieve, however, it gives a very good lower bound in a couple of minutes, instead of hours.

To give you an example, the linear evaluation of SimSiam's network achieves a 68% accuracy in 5 hours, while this code achieves 66% in 5 minutes with the same setup.

## :question: How

For accurate offline evaluation researchers use random crops of images. In contrast, this repository takes only a few crops of images, saves the generated embeddings to RAM and quickly iterates that with a large batch size and apex's LARC optimizer.

## :scientist: Target audience

This modul is generally made for researchers working with Imagenet, therefore, the evalautor was designed for a multi-gpu setup with a large amount of RAM provided (200GB+). It's because the evaluator saves all training embeddings to RAM for quick access.

## :electric_plug: Usage

### Define instance

First, build your encoder model in either a single-gpu or a multi-gpu setup. Then, 
create an evalautor instance by
```python
evaluator = Evaluator(model, cnn_dim=2048, dataset="imagenet", root='/data/imagenet/', batch_size=256)
```

| Arg | Description |
| --- | ----------- |
| model | The encoder model that maps the input image to a cnn_dim representation. The model doesn't need to be freezed or be in eval mode. |
| cnn_dim | The dimension of the representation. |
| dataset | Name of the dataset. Choose from `'imagenet', 'cifar10', 'cifar100'` |
| root | Path to your dataset. |
| batch_size | The batch size used for iterating over images. |

 ### Generate embeddings

```python
z, y = evaluator.generate_embeddings(n_views=2)
```
| Arg | Description |
| --- | ----------- |
| n_views | number of augmentations, number of views you desire to get from each image example |

| Return value | Description |
| --- | ----------- |
| z | NxDxV tensor, where N is the number of samples, D is the cnn_dim and V is the number of views. |
| y | Tensor of labels with length of N |


### Run linear evaluation
```python
top1_acc = evaluator.linear_eval(z, y, epochs=5, batch_size=4096, lr=1.6, verbose=True)
```
This is a recommended setup for Imagenet.

| Arg | Description |
| --- | ----------- |
| z | NxDxV tensor, where N is the number of samples, D is the cnn_dim and V is the number of views. |
| y | Tensor of labels with length of N |
| epochs | Number of epochs to train. 10 by default. |
| batch_size | Batch sized used for iterating over the embeddings. 256 by default. |
| lr | Learning rate. 1 by default. |
| verbose | Allowing rank0 process to print results to the standard ouput. Default is True. |

| Return value | Description |
| --- | ----------- |
| top1_acc | Top1 accuracy achieved on the validation set. |

*Note: Nvidia's apex Larc optimizer used.*

### KNN
```python
top1_accs = evaluator.knn(z, y, [1,5,20])
```
| Arg | Description |
| --- | ----------- |
| z | NxDxV tensor, where N is the number of samples, D is the cnn_dim and V is the number of views. |
| y | Tensor of labels with length of N |
| ks | The K values we desire to run the KNN with. |

| Return value | Description |
| --- | ----------- |
| top1_accs | Top1 accuracies to the K values given, respectively. |


## Contact

For any inquiries please contact me at
gergopool[at]gmail.com
