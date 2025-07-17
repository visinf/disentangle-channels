import os
import argparse
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models.resnet import resnet50

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_dir', metavar='DIR', default='/datasets/imagenet/',
                    help='path to dataset')
parser.add_argument('--store_dir', metavar='DIR', default='/results/',
                    help='path to store results')
parser.add_argument('--model', required=True,
                    choices=['resnet50'],
                    help='model architecture')
parser.add_argument('--batch_size', default=64, type=int,
                    help='batch size')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU to use')

input_activation = {}
def get_input(name):
    def hook(model, input, output):
        input_activation[name] = input[0].detach()
    return hook

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

gradient = {}
def get_gradient(name):
    def hook(model, input, output):
        #print(len(output))
        gradient[name] = output[0].detach()
    return hook

args = parser.parse_args()

device = 'cuda:0'
if args.model == 'resnet50':
    model = resnet50(pretrained=True)
else:
    print('MODEL NOT IMPLEMENTED!')
    assert False

model = model.to(device)
model.eval()

traindir = os.path.join(args.data_dir, 'train')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# the train transform is on purpose the same as for eval
train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True)


store_path = os.path.join(args.store_dir, args.model + '_channel_importances/')
if not os.path.exists(store_path):
    os.makedirs(store_path)
    print('CREATING STORE DIR:')
    print(store_path)

else:
    print('STORE DIR ALREADY EXISTS:')
    print(store_path)

if args.model == 'resnet50':
    model.layer4[2].conv3.register_forward_hook(get_input('layer_of_interest'))
    model.layer4[2].conv3.register_forward_hook(get_activation('layer_of_interest'))
    model.layer4[2].conv3.register_backward_hook(get_gradient('layer_of_interest'))
else:
    print('MODEL NOT IMPLEMENTED!')
    assert False

nr_iterations = 0
for images, targets in tqdm(train_loader):
    images = images.to(device)
    images.requires_grad = True
    targets = targets.to(device)
    outputs = model(images)


    # running this script can take a while and might break
    # this snippet below makes sure that existing files are not computed again but that the script continues where it stopped
    all_already_exist = False
    for b in range(images.shape[0]):
        file_name = 'img' + str(nr_iterations*args.batch_size + b)
        if not os.path.isfile(store_path + file_name + '_prediction.pt'):
            break
        if b == images.shape[0]-1: #only reached if all samples in batch exist
            all_already_exist = True
    if all_already_exist:
        nr_iterations += 1
        continue

    
    target_outputs = torch.gather(outputs, 1, targets.unsqueeze(-1))
    gradients = torch.autograd.grad(torch.unbind(target_outputs), images, create_graph=False)[0]
    
    attribution = activation['layer_of_interest'] * gradient['layer_of_interest']
    channel_attribution_layer = attribution.sum(dim=(2,3)) # B, C
    channel_attribution_layer = channel_attribution_layer.to(torch.float16)
    
    for b in range(images.shape[0]):
        file_name = 'img' + str(nr_iterations*args.batch_size + b)
        if os.path.isfile(store_path + file_name + '_prediction.pt'):
            continue

        #print('NEW:', file_name)
        # store results
        torch.save(channel_attribution_layer[b].detach(), store_path + file_name + '_channel_importance_layer_wrt_class.pt')
        torch.save(targets[b].detach(), store_path + file_name + '_target.pt')
        torch.save(outputs[b].argmax().detach(), store_path + file_name + '_prediction.pt')
        
    
    nr_iterations += 1
