import torchvision

imagenet_instance = torchvision.datasets.ImageNet(root = '/data7/sooyeon/MyData/imagenet',
                                                  split = 'train')