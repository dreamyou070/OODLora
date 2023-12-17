import torchvision

imagenet_instance = torchvision.datasets.ImageNet(root = 'ImageNet',
                                                  split = 'train')