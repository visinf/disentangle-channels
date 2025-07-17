import os
import copy
import argparse
from tqdm import tqdm
import pickle
import numpy as np
from numpy import dot
from numpy.linalg import norm
#from sklearn import cluster

import torch
from torch.utils.data import Dataset

from models.resnet import resnet50

from utils.visualization import get_number_of_images_per_class, get_subdataset_for_classes


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_dir', metavar='DIR', default='/fastdata/rhesse/datasets/imagenet/',
                    help='path to dataset')
parser.add_argument('--store_dir', metavar='DIR', default='/fastdata/rhesse/phd_remove_superposition_clean',
                    help='path to store results')
parser.add_argument('--model', required=True,
                    choices=['resnet50'],
                    help='model architecture')
parser.add_argument('--batch_size', default=64, type=int,
                    help='batch size')
parser.add_argument('--nr_channels_in_layer_of_interest', type=int, required=True,
                    help='how many channels does the layer of interest have')
parser.add_argument('--proportion_attribution_required', default=0.05, type=float,
                    help='how much attribution is required for a channel to be important')
parser.add_argument('--p', default=0.75, type=float,
                    help='for what fraction of images from the class must the channel be important.')
parser.add_argument('--cos_sim_low', default=0., type=float,
                    help='minimum cosine similarity to be included in final list')
parser.add_argument('--cos_sim_high', default=1., type=float,
                    help='maximum cosine similarity to be included in final list')

args = parser.parse_args()

store_path = os.path.join(args.store_dir, 'results_get_cosine_similarities')
if not os.path.exists(store_path):
    os.makedirs(store_path)
    print('CREATING STORE DIR:')
    print(store_path)

else:
    print('STORE DIR ALREADY EXISTS:')
    print(store_path)

device = 'cuda:0'

if args.model == 'resnet50':
    model = resnet50(pretrained=True)
else:
    print('MODEL NOT IMPLEMENTED!')
    raise

model = model.to(device)
model.eval()
print('Model done.')

def cos_similarity(a,b):
    return dot(a, b)/(norm(a)*norm(b))

class ImportanceDataset(Dataset):
    
    def __init__(self, store_path, only_class_importances, transform=None):
        """
        Arguments:
            store_path (string): Path to the folder with importances.
            only_class_importances (bool): True if only channel importance wrt target class should be returned
            transform (callable, optional): Optional transform to be applied
                on a sample (NOT SUPPORTED).
        """
        self.store_path = store_path
        self.only_class_importances = only_class_importances
        self.transform = transform

    def __len__(self):
        return 1281167 # number of images in ImageNet train set


    def __getitem__(self, idx):
        file_name = 'img' + str(idx)
        target = torch.load(os.path.join(self.store_path, file_name + '_target.pt'))
        prediction = torch.load(os.path.join(self.store_path, file_name + '_prediction.pt'))
        channel_importance_layer_wrt_class = torch.load(os.path.join(self.store_path, file_name + '_channel_importance_layer_wrt_class.pt'))
        if self.only_class_importances:
            channel_importance_before_layer_wrt_each_channel = 0
        else:
            channel_importance_before_layer_wrt_each_channel = torch.load(os.path.join(self.store_path, file_name + '_channel_importance_before_layer_wrt_each_channel.pt'))

        sample = {'target': target, 'prediction': prediction, 'channel_importance_layer_wrt_class': channel_importance_layer_wrt_class, 'channel_importance_before_layer_wrt_each_channel': channel_importance_before_layer_wrt_each_channel}

        return sample
    
nr_valid_samples = 0 # where prediction is correct
proportion_attribution_required = args.proportion_attribution_required # proportion of attribution that must be "explained" by selected channels in layer to disentangle
experiment_name = args.model
nr_channels_in_layer_of_interest = args.nr_channels_in_layer_of_interest
traindir = os.path.join(args.data_dir, 'train')

# for each channel get list of classes that use it (i.e. at least X% of attribution is for this channel) in more than p% of images --> then it is considered to be important
p = args.p

# init all channels for all classes with 0
class_histograms = [] # for each class, count how often a channel is important first list is for class, second list for how often each channel is active for that class
for i in range(1000):
    class_histograms.append([])
    for c in range(nr_channels_in_layer_of_interest):
        class_histograms[i].append(0)


importance_dataset = ImportanceDataset(store_path=os.path.join(args.store_dir, args.model + '_channel_importances'), only_class_importances=True)
importance_loader = torch.utils.data.DataLoader(
        importance_dataset,
        batch_size=1024*16, shuffle=False,
        num_workers=0, pin_memory=False)

# compute how often each channel is important for each class
if not os.path.isfile(os.path.join(store_path, experiment_name + '_' + 'required' + str(proportion_attribution_required) + '_class_histograms.pkl')):
    for sample in tqdm(importance_loader):
        targets = sample['target']
        targets = targets.to(device)
        predictions = sample['prediction']
        channel_importance_layer_wrt_class = sample['channel_importance_layer_wrt_class']
        channel_importance_layer_wrt_class = channel_importance_layer_wrt_class.to(device)
        B,C = channel_importance_layer_wrt_class.shape
        assert nr_channels_in_layer_of_interest == C
        
        total_attribution_pos = torch.clamp(channel_importance_layer_wrt_class, min=0.0, max=None).sum(dim=1)
        total_attribution_for_sample_required = total_attribution_pos * proportion_attribution_required
    
        channels_important = (channel_importance_layer_wrt_class >= total_attribution_for_sample_required.unsqueeze(-1)).int() # B,C and 1 if channel is important; 0 if not
        
        # count how often each channel idx is available per class
        targets_available = torch.unique(targets)
        for target_available in targets_available:
            true_if_target = targets == target_available
            channels_important_for_target = channels_important[true_if_target, :]

            counts = channels_important_for_target.sum(dim=0)
            for c in range(C):
                class_histograms[target_available][c] += counts[c].item()
        
    
    with open(os.path.join(store_path, experiment_name + '_' + 'required' + str(proportion_attribution_required) + '_class_histograms.pkl'), 'wb+') as fp:
        pickle.dump(class_histograms, fp)
        print('Dictionary saved successfully to file.')

else:
    with open(os.path.join(store_path, experiment_name + '_' + 'required' + str(proportion_attribution_required) + '_class_histograms.pkl'), 'rb') as fp:
        class_histograms = pickle.load(fp)
        print('Dictionary loaded successfully.')




# for each neuron, get how often each class uses it.
channel_histograms = []
for c in range(nr_channels_in_layer_of_interest):
    channel_histograms.append([])
    for i in range(1000):
        channel_histograms[c].append(class_histograms[i][c])

number_of_images_per_class = get_number_of_images_per_class(traindir)

# for each channel get list of classes that use it in more than p% of images
classes_for_channel = []
for c in range(nr_channels_in_layer_of_interest):
    classes_for_channel.append([])
    for i in range(1000):
        if channel_histograms[c][i] >= p * number_of_images_per_class[i]:
            classes_for_channel[c].append(i)

# ITERATE OVER ALL CHANNELS AND RESPECTIVE IMPORTANT CLASSES, FOR EACH CLASS GO OVER DATASET AND COMPUTE AVERAGE VECTOR FOR SOURCE LAYER ##########
# Average vector for source: Compute importances vector in L-1 and get average over entire dataset #########################
print('For each channel that is important for multiple classes: compute average activation vector in L-1 for that class.')

model_copy = copy.deepcopy(model)
model_copy = model_copy.to(device)
model_copy.eval()

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
        gradient[name] = output[0].detach()
    return hook

if args.model == 'resnet50':
    model.layer4[2].conv3.register_forward_hook(get_input('layer_of_interest'))
    model.layer4[2].conv3.register_forward_hook(get_activation('layer_of_interest'))
    model.layer4[2].conv3.register_backward_hook(get_gradient('layer_of_interest'))
else:
    print('MODEL NOT IMPLEMENTED!')
    raise


number_of_images_per_class = get_number_of_images_per_class(traindir) # this is used to get image indices


# this file grows over time as it is independent of the required attribution and other parameters

if os.path.isfile(os.path.join(store_path, experiment_name + '_importance_vectors_for_all_channels_and_corresponding_important_classes_unnormalized.pkl')):
    with open(os.path.join(store_path, experiment_name + '_importance_vectors_for_all_channels_and_corresponding_important_classes_unnormalized.pkl'), 'rb') as fp:
        importance_vectors_for_all_channels_and_corresponding_important_classes = pickle.load(fp)
        print('importance_vectors_for_all_channels_and_corresponding_important_classes loaded successfully.')
else:
    print('importance_vectors_for_all_channels_and_corresponding_important_classes not existing yet, create new one')
    importance_vectors_for_all_channels_and_corresponding_important_classes = {}
    
for channel_of_interest in tqdm(range(nr_channels_in_layer_of_interest)):
    
    classes_of_interest = classes_for_channel[channel_of_interest]
    # if channel is not important for at least two classes, we don't need it
    if len(classes_of_interest) < 2:
        continue

    if channel_of_interest in importance_vectors_for_all_channels_and_corresponding_important_classes.keys():
        class_vectors_avg = importance_vectors_for_all_channels_and_corresponding_important_classes[channel_of_interest]
    else:
        class_vectors_avg = {}
    
    for class_of_interest in classes_of_interest:
        if class_of_interest in class_vectors_avg.keys():
            continue # entry already exists

        class_vectors_all = [] # store all importance vectors for each sample in the class

        # build loader for class of interest and store L-1 attributions wrt channel of interest
        dataset = get_subdataset_for_classes(traindir, [class_of_interest])
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=8, pin_memory=True)
        
        start_index_for_class = sum(number_of_images_per_class[:class_of_interest])
        nr_iterations = 0
        for images, targets in loader:
            
            B,C,H,W = images.shape

            # load or compute all the L-1 attributions
            Lmin1_attributions = []
        
            images = images.to(device)
            images.requires_grad = True
            targets = targets.to(device)
            outputs = model(images)
            
            inputs_intermediate = input_activation['layer_of_interest'].clone()
            inputs_intermediate.requires_grad = True

            if args.model == 'resnet50':
                outputs_intermediate = model_copy.layer4[2].conv3(inputs_intermediate)
            else:
                print('MODEL NOT IMPLEMENTED!')
                raise
            
            outputs_intermediate = outputs_intermediate.sum(dim=(2,3))

            channel_target = (torch.ones_like(targets) * channel_of_interest).type(torch.int64)
            target_outputs = torch.gather(outputs_intermediate, 1, channel_target.unsqueeze(-1))

            gradients = torch.autograd.grad(torch.unbind(target_outputs), inputs_intermediate, create_graph=False)[0]
            attribution = gradients * inputs_intermediate
            channel_attribution = attribution.sum(dim=(2,3)) # B, C
            channel_attribution = channel_attribution.to(torch.float16)
            Lmin1_attributions = channel_attribution

            channel_importance_before_layer_wrt_channel_of_interest_unnormalized = Lmin1_attributions
            
            class_vectors_all.append(channel_importance_before_layer_wrt_channel_of_interest_unnormalized.detach().cpu())
            nr_iterations += 1

        class_vectors_all = torch.cat(class_vectors_all, dim=0)
        class_vectors_avg[class_of_interest] = (class_vectors_all.mean(dim=0, keepdim=True).detach().cpu())

    importance_vectors_for_all_channels_and_corresponding_important_classes[channel_of_interest] = class_vectors_avg


with open(os.path.join(store_path, experiment_name + '_importance_vectors_for_all_channels_and_corresponding_important_classes_unnormalized.pkl'), 'wb') as fp:
    pickle.dump(importance_vectors_for_all_channels_and_corresponding_important_classes, fp)



final_list = [] # contains dicts with relevant information        

cosine_sims = []
for channel_of_interest in range(nr_channels_in_layer_of_interest):

    classes_of_interest = classes_for_channel[channel_of_interest]

    # if channel is not important for at least two classes, we don't need it
    if len(classes_of_interest) < 2:
        continue

    # iterate over all possible pairs
    for it1 in range(len(classes_of_interest)-1):
        for it2 in range(it1+1, len(classes_of_interest)):
            cosine_sim = cos_similarity(importance_vectors_for_all_channels_and_corresponding_important_classes[channel_of_interest][classes_of_interest[it1]][0],importance_vectors_for_all_channels_and_corresponding_important_classes[channel_of_interest][classes_of_interest[it2]][0])
            cosine_sims.append(cosine_sim)

            final_list.append({'channel': channel_of_interest, 'class1': classes_of_interest[it1], 'class2': classes_of_interest[it2], 'cosine_similarity': cosine_sim})

final_list = sorted(final_list, key=lambda x: x['cosine_similarity'], reverse = True)

print('Objects with cosine similarity below threshold:')
final_list_below_threshold = list(filter(lambda x: x['cosine_similarity'] < args.cos_sim_high and x['cosine_similarity'] >= args.cos_sim_low, final_list))
print(final_list_below_threshold)
print(len(final_list_below_threshold))

out_string = "Channel,Class1,Class2,CosineSimilarity\n"
for element in final_list_below_threshold:
    out_string = out_string + str(element['channel']) + "," + str(element['class1']) + "," + str(element['class2']) + "," + str(element['cosine_similarity']) + "\n"

writepath = './results_get_cosine_similarities/' + args.model + '_required' + str(args.proportion_attribution_required) + '_p' + str(args.p) + '_cossim' + str(args.cos_sim_low) + 'to' + str(args.cos_sim_high) + '.txt'
with open(writepath, 'w') as f:
    f.write(out_string)

