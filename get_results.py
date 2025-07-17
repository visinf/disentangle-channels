import argparse
import os
import copy

from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

import torch
import torch.nn as nn

from models.resnet import resnet50

from utils.visualization import get_subdataset_for_classes

parser = argparse.ArgumentParser(description='Qualitative Plots')
parser.add_argument('--data_dir', metavar='DIR', default='/datasets/imagenet/',
                    help='path to dataset')
parser.add_argument('--model_dir', metavar='DIR', default='/results/models_disentangled/',
                    help='path to store results')
parser.add_argument('--model', required=True,
                    choices=['resnet50'],
                    help='model architecture')
parser.add_argument('--batch_size', default=64, type=int,
                    help='batch size')
parser.add_argument('--nr_channels_in_layer_of_interest', type=int, required=True,
                    help='Number of channels in layer of interest')
parser.add_argument('--layer_of_interest', required=True,
                    choices=['last_conv'])
parser.add_argument('--classes_for_concept1', required=True, nargs="+", type=int)
parser.add_argument('--classes_for_concept2', required=True, nargs="+", type=int)
parser.add_argument('--channel_of_interest', required=True, type=int)

parser.add_argument('--f', type=float, required=False, default=None,
                    help='factor how much higher L-1 activation must be for one concept than for the other. If None, best will be used.')

# for quantitative analysis
parser.add_argument('--get_qualitative', action='store_true',
                    help='set true if you want to get qualitative results (density plot)')

# for quantitative analysis
parser.add_argument('--get_quantitative', action='store_true',
                    help='set true if you want to get quantitative results')
parser.add_argument('--proportion_attribution_required', default=0.03, type=float,
                    help='how much relative attribution is required for a channel to be important (tau in paper); only needed for quantitative analysis')


args = parser.parse_args()

if not os.path.exists('./plots'):
    os.makedirs('./plots')
    print('CREATING PLOT DIR:')
    print('./plots')

else:
    print('PLOT DIR ALREADY EXISTS:')
    print('./plots')

if not os.path.exists('./results_quantitative'):
    os.makedirs('./results_quantitative')
    print('CREATING PLOT DIR:')
    print('./results_quantitative')

else:
    print('PLOT DIR ALREADY EXISTS:')
    print('./results_quantitative')

device = 'cuda:0'
if args.model == 'resnet50':
    model = resnet50(pretrained=True)
else:
    print('MODEL NOT IMPLEMENTED!')
    assert False

model = model.to(device)
model.eval()
print('Standard model done.')

model_disentangled_final = copy.deepcopy(model)
model_disentangled_final = model_disentangled_final.to(device)
model_disentangled_final.eval()

class TrainingModel(torch.nn.Module):
    def __init__(self, channel_of_interest, input_channels, kernel_size, stride, padding=0, bias=False):
        super(TrainingModel, self).__init__()

        self.disentangled = nn.Conv2d(input_channels, 3, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
        self.merged = nn.Conv2d(3, 1, kernel_size=1, stride=1, bias=False)
        merged_weight = self.merged.weight.clone()
        merged_weight[0,0,0,0] = 1.
        merged_weight[0,1,0,0] = 1.
        merged_weight[0,2,0,0] = -1
        self.merged.weight = nn.Parameter(merged_weight)

        self.channel_of_interest = channel_of_interest

    def forward(self, x):

        disentangled = self.disentangled(x)
        merged = self.merged(disentangled)

        return merged, disentangled
    
if args.model == 'resnet50':
    model_disentangled = TrainingModel(args.channel_of_interest, 512, 1, 1).to(device)
    f = args.f
    if f == None:
        f = '_best'
    
    model_disentangled.load_state_dict(torch.load(os.path.join(args.model_dir, args.model + '_channel' + str(args.channel_of_interest) + '_classes1' + str(args.classes_for_concept1) + '_classes2' + str(args.classes_for_concept2) + '_f' + str(f) + '_unnormalized.pth')))
    model_disentangled_final.layer4[2].conv3 = nn.Sequential(
            nn.Conv2d(512, 2048+2, kernel_size=1, stride=1, bias=False),
            nn.Conv2d(2048+2, 2048, kernel_size=1, stride=1, bias=False)
        ).to(device)

    new_weight_Lmin1_L = model.layer4[2].conv3.weight.clone()

    new_weight_Lmin1_L[args.channel_of_interest, :,:,:] = model_disentangled.disentangled.weight[0,:,:,:].clone()
    new_weight_Lmin1_L = torch.cat([new_weight_Lmin1_L,model_disentangled.disentangled.weight[1:2,:,:,:].clone()], dim=0) 
    new_weight_Lmin1_L = torch.cat([new_weight_Lmin1_L,model_disentangled.disentangled.weight[2:3,:,:,:].clone()], dim=0) 

    model_disentangled_final.layer4[2].conv3[0].weight = nn.Parameter(new_weight_Lmin1_L)

    identity = torch.zeros((2048, 2048+2, 1, 1)).to(device)
    for i in range(2048):
        identity[i,i,0,0] = 1.
    identity[args.channel_of_interest,2048,0,0] = 1.
    identity[args.channel_of_interest,2048+1,0,0] = -1.
    model_disentangled_final.layer4[2].conv3[1].weight = nn.Parameter(identity)
else:
    print('MODEL NOT IMPLEMENTED!')
    assert False

print('Disentangled model done.')

valdir = os.path.join(args.data_dir, 'val')

activation_hook = {}
def get_activation(name):
    def hook(model, input, output):
        activation_hook[name] = output.detach()
    return hook

gradient = {}
def get_gradient(name):
    def hook(model, input, output):
        #print(len(output))
        gradient[name] = output[0].detach()
    return hook

if args.model == 'resnet50':
    model.layer4[2].conv3.register_forward_hook(get_activation('conv3_original'))
    model.layer4[2].conv3.register_backward_hook(get_gradient('conv3_original'))

    model_disentangled_final.layer4[2].conv3[0].register_forward_hook(get_activation('conv3_disentangled'))
    model_disentangled_final.layer4[2].conv3[0].register_backward_hook(get_gradient('conv3_disentangled'))



# classes for concept 1
dataset = get_subdataset_for_classes(valdir, args.classes_for_concept1)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64, shuffle=False,
    num_workers=0, pin_memory=True)

activations_channel_original_concept1 = []
activations_channel_disentangled1_concept1 = []
activations_channel_disentangled2_concept1 = []
activations_channel_residue_concept1 = []

relative_attribution_channel_original_concept1 = []


for images, targets in tqdm(loader):
        
    images = images.to(device)
    images.requires_grad = True
    targets = targets.to(device)

    outputs_original = model(images)

    target_outputs = torch.gather(outputs_original, 1, targets.unsqueeze(-1))
    _ = torch.autograd.grad(torch.unbind(target_outputs), images, create_graph=False)[0]
    attribution = activation_hook['conv3_original'] * gradient['conv3_original']
    attribution = torch.clamp(attribution, min=0.0, max=None) # discard negative attribution, we are only interested in positive evidence
    attribution_aggregated = attribution.sum(dim=(2,3)) # B, C
    relative_attribution_channel_of_interest = attribution_aggregated[:, args.channel_of_interest] / attribution_aggregated.sum(dim=1) # get proportion of attribution (0,1) range
    relative_attribution_channel_original_concept1.append(relative_attribution_channel_of_interest.detach().cpu().numpy())

    activation = activation_hook['conv3_original']
    activation_aggregated = activation.sum(dim=(2,3)) # B, C
    activation_aggregated_channel_of_interest = activation_aggregated[:, args.channel_of_interest]
    activations_channel_original_concept1.append(activation_aggregated_channel_of_interest.detach().cpu().numpy())

    outputs_disentangled = model_disentangled_final(images)

    activation = activation_hook['conv3_disentangled']
    activation_aggregated = activation.sum(dim=(2,3)) # B, C
    activation_aggregated_channel_of_interest = activation_aggregated[:, args.channel_of_interest]
    activations_channel_disentangled1_concept1.append(activation_aggregated_channel_of_interest.detach().cpu().numpy())

    activation_aggregated_channel_of_interest = activation_aggregated[:, args.nr_channels_in_layer_of_interest]
    activations_channel_disentangled2_concept1.append(activation_aggregated_channel_of_interest.detach().cpu().numpy())

    activation_aggregated_channel_of_interest = activation_aggregated[:, args.nr_channels_in_layer_of_interest+1]
    activations_channel_residue_concept1.append(activation_aggregated_channel_of_interest.detach().cpu().numpy())

activations_channel_original_concept1 = np.concatenate(activations_channel_original_concept1, axis=0)
activations_channel_disentangled1_concept1 = np.concatenate(activations_channel_disentangled1_concept1, axis=0)
activations_channel_disentangled2_concept1 = np.concatenate(activations_channel_disentangled2_concept1, axis=0)
activations_channel_residue_concept1 = np.concatenate(activations_channel_residue_concept1, axis=0)
relative_attribution_channel_original_concept1 = np.concatenate(relative_attribution_channel_original_concept1, axis=0)


# classes for concept 2
dataset = get_subdataset_for_classes(valdir, args.classes_for_concept2)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64, shuffle=False,
    num_workers=0, pin_memory=True)

activations_channel_original_concept2 = []
activations_channel_disentangled1_concept2 = []
activations_channel_disentangled2_concept2 = []
activations_channel_residue_concept2 = []

relative_attribution_channel_original_concept2 = []


for images, targets in tqdm(loader):
        
    images = images.to(device)
    images.requires_grad = True
    targets = targets.to(device)

    outputs_original = model(images)

    target_outputs = torch.gather(outputs_original, 1, targets.unsqueeze(-1))
    _ = torch.autograd.grad(torch.unbind(target_outputs), images, create_graph=False)[0]
    attribution = activation_hook['conv3_original'] * gradient['conv3_original']
    attribution = torch.clamp(attribution, min=0.0, max=None) # discard negative attribution, we are only interested in positive evidence
    attribution_aggregated = attribution.sum(dim=(2,3)) # B, C
    relative_attribution_channel_of_interest = attribution_aggregated[:, args.channel_of_interest] / attribution_aggregated.sum(dim=1) # get proportion of attribution (0,1) range
    relative_attribution_channel_original_concept2.append(relative_attribution_channel_of_interest.detach().cpu().numpy())


    activation = activation_hook['conv3_original']
    activation_aggregated = activation.sum(dim=(2,3)) # B, C
    activation_aggregated_channel_of_interest = activation_aggregated[:, args.channel_of_interest]
    activations_channel_original_concept2.append(activation_aggregated_channel_of_interest.detach().cpu().numpy())

    outputs_disentangled = model_disentangled_final(images)

    activation = activation_hook['conv3_disentangled']
    activation_aggregated = activation.sum(dim=(2,3)) # B, C
    activation_aggregated_channel_of_interest = activation_aggregated[:, args.channel_of_interest]
    activations_channel_disentangled1_concept2.append(activation_aggregated_channel_of_interest.detach().cpu().numpy())

    activation_aggregated_channel_of_interest = activation_aggregated[:, args.nr_channels_in_layer_of_interest]
    activations_channel_disentangled2_concept2.append(activation_aggregated_channel_of_interest.detach().cpu().numpy())

    activation_aggregated_channel_of_interest = activation_aggregated[:, args.nr_channels_in_layer_of_interest+1]
    activations_channel_residue_concept2.append(activation_aggregated_channel_of_interest.detach().cpu().numpy())

activations_channel_original_concept2 = np.concatenate(activations_channel_original_concept2, axis=0)
activations_channel_disentangled1_concept2 = np.concatenate(activations_channel_disentangled1_concept2, axis=0)
activations_channel_disentangled2_concept2 = np.concatenate(activations_channel_disentangled2_concept2, axis=0)
activations_channel_residue_concept2 = np.concatenate(activations_channel_residue_concept2, axis=0)
relative_attribution_channel_original_concept2 = np.concatenate(relative_attribution_channel_original_concept2, axis=0)


if args.get_qualitative:

    with open("results_disentanglement/density_concept1.txt", "w") as text_file:
        x_original, y_original = sns.kdeplot(np.array(activations_channel_original_concept1)).lines[0].get_data()
        text_file.write(r'\begin{filecontents*}{original_c1.dat}')
        text_file.write('\n')
        for coord in zip(x_original,y_original):
            text_file.write(str(coord[0]) + " " + str(coord[1]) + "\n")
        text_file.write(r'\end{filecontents*}')
        text_file.write('\n')
        plt.clf() # clear is required to get new values

        x_d1, y_d1 = sns.kdeplot(np.array(activations_channel_disentangled1_concept1)).lines[0].get_data()
        text_file.write(r'\begin{filecontents*}{disentangled1_c1.dat}')
        text_file.write('\n')
        for coord2 in zip(x_d1,y_d1):
            text_file.write(str(coord2[0]) + " " + str(coord2[1]) + "\n")
        text_file.write(r'\end{filecontents*}')
        text_file.write('\n')
        plt.clf()

        x_d2, y_d2 = sns.kdeplot(np.array(activations_channel_disentangled2_concept1)).lines[0].get_data()
        text_file.write(r'\begin{filecontents*}{disentangled2_c1.dat}')
        text_file.write('\n')
        for coord in zip(x_d2,y_d2):
            text_file.write(str(coord[0]) + " " + str(coord[1]) + "\n")
        text_file.write(r'\end{filecontents*}')
        text_file.write('\n')
        plt.clf()

    sns.set_style('whitegrid')
    sns.kdeplot(np.array(activations_channel_original_concept1), label="Original")
    sns.kdeplot(np.array(activations_channel_disentangled1_concept1), label="Disentangled 1")
    sns.kdeplot(np.array(activations_channel_disentangled2_concept1), label="Disentangled 2")
    plt.xlabel("Activation")
    plt.legend()
    plt.savefig("plots/density_concept1.png")
    plt.clf()

    with open("results_disentanglement/density_concept2.txt", "w") as text_file:
        x_original, y_original = sns.kdeplot(np.array(activations_channel_original_concept2)).lines[0].get_data()
        text_file.write(r'\begin{filecontents*}{original_c2.dat}')
        text_file.write('\n')
        for coord in zip(x_original,y_original):
            text_file.write(str(coord[0]) + " " + str(coord[1]) + "\n")
        text_file.write(r'\end{filecontents*}')
        text_file.write('\n')
        plt.clf() # clear is required to get new values

        x_d1, y_d1 = sns.kdeplot(np.array(activations_channel_disentangled1_concept2)).lines[0].get_data()
        text_file.write(r'\begin{filecontents*}{disentangled1_c2.dat}')
        text_file.write('\n')
        for coord2 in zip(x_d1,y_d1):
            text_file.write(str(coord2[0]) + " " + str(coord2[1]) + "\n")
        text_file.write(r'\end{filecontents*}')
        text_file.write('\n')
        plt.clf()

        x_d2, y_d2 = sns.kdeplot(np.array(activations_channel_disentangled2_concept2)).lines[0].get_data()
        text_file.write(r'\begin{filecontents*}{disentangled2_c2.dat}')
        text_file.write('\n')
        for coord in zip(x_d2,y_d2):
            text_file.write(str(coord[0]) + " " + str(coord[1]) + "\n")
        text_file.write(r'\end{filecontents*}')
        text_file.write('\n')
        plt.clf()

    sns.set_style('whitegrid')
    sns.kdeplot(np.array(activations_channel_original_concept2), label="Original")
    sns.kdeplot(np.array(activations_channel_disentangled1_concept2), label="Disentangled 1")
    sns.kdeplot(np.array(activations_channel_disentangled2_concept2), label="Disentangled 2")
    plt.xlabel("Activation")
    plt.legend()
    plt.savefig("plots/density_concept2.png")
    plt.clf()







# get quantitative results

if args.get_quantitative:
    out_string_concept1 = ""

    for i in range(activations_channel_original_concept1.shape[0]):
        if relative_attribution_channel_original_concept1[i] >= args.proportion_attribution_required:
            fraction_disentangled1 = activations_channel_disentangled1_concept1[i] / activations_channel_original_concept1[i]
            fraction_disentangled2 = activations_channel_disentangled2_concept1[i] / activations_channel_original_concept1[i]
            fraction_residue = activations_channel_residue_concept1[i] / activations_channel_original_concept1[i]
            out_string_concept1 += str(fraction_disentangled1) + "," + str(fraction_disentangled2) + "," + str(fraction_residue) + "\n"


    out_string_concept2 = ""
    for i in range(activations_channel_original_concept2.shape[0]):
        if relative_attribution_channel_original_concept2[i] >= args.proportion_attribution_required:
            fraction_disentangled1 = activations_channel_disentangled1_concept2[i] / activations_channel_original_concept2[i]
            fraction_disentangled2 = activations_channel_disentangled2_concept2[i] / activations_channel_original_concept2[i]
            fraction_residue = activations_channel_residue_concept2[i] / activations_channel_original_concept2[i]
            out_string_concept2 += str(fraction_disentangled1) + "," + str(fraction_disentangled2) + "," + str(fraction_residue) + "\n"
    
    #print(out_string_concept1)
    #print(out_string_concept2)

    # either append or create new
    writepath = os.path.join('results_quantitative/', 'quantitative_concept1_required' + str(args.proportion_attribution_required) + '.txt')
    mode = 'a+' if os.path.exists(writepath) else 'w'
    with open(writepath, mode) as f:
        f.write(out_string_concept1)

    writepath = os.path.join('results_quantitative/', 'quantitative_concept2_required' + str(args.proportion_attribution_required) + '.txt')
    mode = 'a+' if os.path.exists(writepath) else 'w'
    with open(writepath, mode) as f:
        f.write(out_string_concept2)