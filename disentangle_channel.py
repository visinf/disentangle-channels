import os
import copy
import argparse
import pickle

import torch
import torch.nn as nn

from models.resnet import resnet50

from utils.visualization import get_subdataset_for_classes

parser = argparse.ArgumentParser(description='Disentangle Channels')
parser.add_argument('--data_dir', metavar='DIR', default='/datasets/imagenet/',
                    help='path to dataset')
parser.add_argument('--store_dir', metavar='DIR', default='/results/',
                    help='path to store results')
parser.add_argument('--model', required=True,
                    choices=['resnet50'],
                    help='model architecture')
parser.add_argument('--batch_size', default=64, type=int,
                    help='batch size')
parser.add_argument('--nr_channels_in_layer_of_interest', type=int, required=True,
                    help='batch size')
parser.add_argument('--channel_disentangle', type=int, required=True,
                    help='channel to disentangle')
parser.add_argument('--classes_for_concept1', required=True, nargs="+", type=int)
parser.add_argument('--classes_for_concept2', required=True, nargs="+", type=int)

args = parser.parse_args()

store_path = os.path.join(args.store_dir, 'models_disentangled')
if not os.path.exists(store_path):
    os.makedirs(store_path)
    print('CREATING STORE DIR:')
    print(store_path)

else:
    print('STORE DIR ALREADY EXISTS:')
    print(store_path)

# fs correspond to different rho in the paper
fs = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95, 1.0]

model_disentangled_relevant_edges_for_fs = {}

device = 'cuda:0'

if args.model == 'resnet50':
    model = resnet50(pretrained=True)
else:
    print('MODEL NOT IMPLEMENTED!')
    assert False

model = model.to(device)
model.eval()

for f in fs:

    model_disentangled = copy.deepcopy(model)
    model_disentangled = model_disentangled.to(device)
    model_disentangled.eval()

    # step1: add new layer that combines disentangled neurons again, init with only ones for each respective channel and 0 for all other
    if args.model == 'resnet50':
        model_disentangled.layer4[2].conv3 = nn.Sequential(
                    nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False),
                    nn.Conv2d(2048, 2048, kernel_size=1, stride=1, bias=False)
                ).to(device)
        model_disentangled.layer4[2].conv3[0].weight = nn.Parameter(model.layer4[2].conv3.weight.clone())

        identity = torch.zeros((2048, 2048, 1, 1)).to(device)
        for i in range(2048):
            identity[i,i,0,0] = 1.

        model_disentangled.layer4[2].conv3[1].weight = nn.Parameter(identity)
    else:
        print('MODEL NOT IMPLEMENTED!')
        assert False


    experiment_name = args.model
    channel_of_interest = args.channel_disentangle

    # get ARVs
    with open(os.path.join(args.store_dir, 'results_get_cosine_similarities', experiment_name + '_importance_vectors_for_all_channels_and_corresponding_important_classes_unnormalized.pkl'), 'rb') as fp:
        importance_vectors_for_all_channels_and_corresponding_important_classes = pickle.load(fp)
        print('importance_vectors_for_all_channels_and_corresponding_important_classes loaded successfully.')
    importance_vectors_for_channel = importance_vectors_for_all_channels_and_corresponding_important_classes[args.channel_disentangle]
    assert len(args.classes_for_concept1) == 1
    assert len(args.classes_for_concept2) == 1 # for now support only single classes

    importance_vectors_for_channel_for_class1 = importance_vectors_for_channel[args.classes_for_concept1[0]][0] # C
    importance_vectors_for_channel_for_class2 = importance_vectors_for_channel[args.classes_for_concept2[0]][0]

    source_neurons_class1 = []
    source_neurons_class2 = []

    for i in range(importance_vectors_for_channel_for_class1.shape[0]):

        if importance_vectors_for_channel_for_class1[i] * f > importance_vectors_for_channel_for_class2[i]:
            source_neurons_class1.append(i)
        elif importance_vectors_for_channel_for_class2[i] * f > importance_vectors_for_channel_for_class1[i]:
            source_neurons_class2.append(i)
   
    # this function performs the actual disentanglement
    def disentangle_neurons(layer_of_interest, channel_of_interest, source_channels1, source_channels2, bias=False, padding=0):

        # add 2 new neurons to layer of interest for disentangled concept and overlap
        out_channels, in_channels, kernel,_ = layer_of_interest[0].weight.shape
        original_weight = layer_of_interest[0].weight.clone()
        if bias:
            original_bias = layer_of_interest[0].bias.clone()
        layer_of_interest[0] = nn.Conv2d(in_channels, out_channels+2, kernel_size=kernel, stride=1, padding=padding, bias=False)

        # set the corresponding weights from source layer to layer of interest
        new_weight = torch.zeros((out_channels+2, in_channels, kernel, kernel)).to(device)
        new_weight[:out_channels, :,:,:] = original_weight
        new_weight[out_channels, :,:,:] = original_weight[channel_of_interest,:,:,:] # this is the one for class2
        new_weight[out_channels+1, :,:,:] = original_weight[channel_of_interest,:,:,:] # this is the one for overlap

        new_weight[channel_of_interest,source_channels2,:,:] = 0.
        new_weight[out_channels,source_channels1,:,:] = 0.
        new_weight[out_channels+1, source_channels1,:,:] = 0.
        new_weight[out_channels+1, source_channels2,:,:] = 0.

        layer_of_interest[0].weight = nn.Parameter(new_weight)
        
        if bias:
            new_bias = torch.zeros((out_channels+2)).to(device)
            new_bias[:out_channels] = original_bias
            new_bias[out_channels] = original_bias[channel_of_interest]
            new_bias[out_channels+1] = original_bias[channel_of_interest]

            layer_of_interest[0].bias = nn.Parameter(new_bias)
        
        # add 2 new sources to merging layer
        out_channels, in_channels, _,_ = layer_of_interest[1].weight.shape
        original_weight = layer_of_interest[1].weight.clone()
        layer_of_interest[1] = nn.Conv2d(in_channels+2, out_channels, kernel_size=1, stride=1, bias=False)

        # set the corresponding weights of merging layer
        new_weight = torch.zeros((out_channels, in_channels+2, 1, 1)).to(device)
        new_weight[:, :in_channels,:,:] = original_weight
        new_weight[channel_of_interest, in_channels,:,:] = 1.
        new_weight[channel_of_interest, in_channels+1,:,:] = -1.
        layer_of_interest[1].weight = nn.Parameter(new_weight)

        return

    print('SOURCE NEURONS')
    print(source_neurons_class1)
    print(source_neurons_class2)
    if args.model == 'resnet50':
        disentangle_neurons(model_disentangled.layer4[2].conv3, channel_of_interest, source_neurons_class1, source_neurons_class2)
    else:
        print('MODEL NOT IMPLEMENTED!')
        assert False

    random_input = torch.randn((1,3,224,224)).to(device)
    print('Are equal:')
    print(torch.isclose(model(random_input), model_disentangled(random_input), atol=1e-03).all())

    # store weights in new model (only relevant edges) and save
    class TrainingModel(torch.nn.Module):
        def __init__(self, channel_of_interest, input_channels, kernel_size, stride, padding=0, bias=False):
            super(TrainingModel, self).__init__()

            self.disentangled = nn.Conv2d(input_channels, 3, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
            self.merged = nn.Conv2d(3, 1, kernel_size=1, stride=1, bias=False)
            merged_weight = self.merged.weight.clone()
            merged_weight[0,0,0,0] = 1.
            merged_weight[0,1,0,0] = 1.
            merged_weight[0,2,0,0] = -1.
            self.merged.weight = nn.Parameter(merged_weight)

            self.channel_of_interest = channel_of_interest

        def forward(self, x):

            disentangled = self.disentangled(x)
            merged = self.merged(disentangled)

            return merged, disentangled

    if args.model == 'resnet50':
        model_disentangled_relevant_edges = TrainingModel(args.channel_disentangle, 512, 1, 1).to(device) 
    else:
        print('MODEL NOT IMPLEMENTED!')
        assert False

    if args.model == 'resnet50':
        out_channels, in_channels, kernel,_ = model.layer4[2].conv3.weight.shape
        new_weight = torch.zeros((3, in_channels, kernel, kernel)).to(device)
        new_weight[0,:,:,:] = model_disentangled.layer4[2].conv3[0].weight[channel_of_interest,:,:,:]
        new_weight[1,:,:,:] = model_disentangled.layer4[2].conv3[0].weight[out_channels,:,:,:]
        new_weight[2,:,:,:] = model_disentangled.layer4[2].conv3[0].weight[out_channels+1,:,:,:]
        model_disentangled_relevant_edges.disentangled.weight = nn.Parameter(new_weight)
    else:
        print('MODEL NOT IMPLEMENTED!')
        assert False

    torch.save(model_disentangled_relevant_edges.state_dict(), os.path.join(store_path, args.model + '_channel' + str(args.channel_disentangle) + '_classes1' + str(args.classes_for_concept1) + '_classes2' + str(args.classes_for_concept2) + '_f' + str(f) + '_unnormalized.pth'))

    model_disentangled_relevant_edges_for_fs[f] = model_disentangled_relevant_edges









# find best f
traindir = os.path.join(args.data_dir, 'train')

input_activation = {}
def get_input(name):
    def hook(model, input, output):
        input_activation[name] = input[0].detach()
    return hook

if args.model == 'resnet50':
    model.layer4[2].conv3.register_forward_hook(get_input('layer_of_interest'))
else:
    print('MODEL NOT IMPLEMENTED!')
    raise

# images containing concept1
concept1_ratios_for_fs = {}
for f in fs:
    concept1_ratios_for_fs[f] = []
train_dataset = get_subdataset_for_classes(traindir, args.classes_for_concept1)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size, shuffle=False,
    num_workers=0, pin_memory=True)

for images, targets in train_loader:
    images = images.to(device)
    targets = targets.to(device)

    _ = model(images)

    intermediate_activations = input_activation['layer_of_interest']

    for f in fs:
        disentangled_channels = model_disentangled_relevant_edges_for_fs[f](intermediate_activations)[1]
        disentangled_channels_pos = torch.clamp(disentangled_channels, min=0.0, max=None).sum(dim=(2,3)) # B, 3
        concept1_ratios_for_fs[f].append((disentangled_channels_pos[:, 0] / (disentangled_channels_pos[:, 1] + 1e-5)).detach())

for f in fs:
    concept1_ratios_for_fs[f] = torch.cat(concept1_ratios_for_fs[f], dim=0).mean().item()

print('concept1_ratios_for_fs')
print(concept1_ratios_for_fs)

# images containing concept2
concept2_ratios_for_fs = {}
for f in fs:
    concept2_ratios_for_fs[f] = []
train_dataset = get_subdataset_for_classes(traindir, args.classes_for_concept2)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=128, shuffle=False,
    num_workers=0, pin_memory=True)

for images, targets in train_loader:
    images = images.to(device)
    targets = targets.to(device)

    _ = model(images)

    intermediate_activations = input_activation['layer_of_interest']

    for f in fs:
        disentangled_channels = model_disentangled_relevant_edges_for_fs[f](intermediate_activations)[1]
        disentangled_channels_pos = torch.clamp(disentangled_channels, min=0.0, max=None).sum(dim=(2,3)) # B, 3
        concept2_ratios_for_fs[f].append((disentangled_channels_pos[:, 1] / (disentangled_channels_pos[:, 0] + 1e-5)).detach())

# compute mean of means
for f in fs:
    concept2_ratios_for_fs[f] = torch.cat(concept2_ratios_for_fs[f], dim=0).mean().item()

print('concept2_ratios_for_fs')
print(concept2_ratios_for_fs)

# get best f
final_ratios = {}
for f in fs:
    final_ratios[f] = concept1_ratios_for_fs[f] + concept2_ratios_for_fs[f] # we want to maximize the ratios

print('final_ratios')
print(final_ratios)
best_f = max(final_ratios, key=final_ratios.get)
print('best_f')
print(best_f)

torch.save(model_disentangled_relevant_edges_for_fs[best_f].state_dict(), os.path.join(store_path, args.model + '_channel' + str(args.channel_disentangle) + '_classes1' + str(args.classes_for_concept1) + '_classes2' + str(args.classes_for_concept2) + '_f' + '_best' + '_unnormalized.pth'))
