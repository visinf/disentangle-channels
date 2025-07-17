import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_number_of_images_per_class(train_dir):
    imagenet_dir = train_dir
    number_of_images_per_class = []
    for subdir, dirs, files in os.walk(imagenet_dir):
        for dir in sorted(dirs):
            dir_for_class = os.path.join(imagenet_dir, dir)
            number_of_images_per_class.append(len([name for name in os.listdir(dir_for_class)]))
    return number_of_images_per_class

def get_subdataset_for_classes(dir, class_idx_list):
    indices_to_keep = []
    number_of_images_per_class = get_number_of_images_per_class(dir)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    # the train transform is on purpose as for eval --> we don't use them for training but measuring stuff
    train_dataset = datasets.ImageFolder(dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    for class_idx in class_idx_list:
        start_idx_for_class = sum(number_of_images_per_class[:class_idx])
        end_idx_for_class = sum(number_of_images_per_class[:class_idx+1])
        for x in list(range(start_idx_for_class, end_idx_for_class)):
            indices_to_keep.append(x)

    return torch.utils.data.Subset(train_dataset, indices_to_keep)