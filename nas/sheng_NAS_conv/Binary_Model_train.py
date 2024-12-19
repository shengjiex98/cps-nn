'''
    this file is for NAS training
'''
import argparse
import os
import torch
import gc
#import psutil
from torch.utils.data import Dataset
from PIL import Image
import os
import sys
from torch.utils.data import random_split
from torchvision import transforms
import copy
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
import time
# solve the HTTP forbidden 304 problem
from six.moves import urllib
from experiment_log import PytorchExperimentLogger
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

num_users_dict = {"ISIC": 1,
                  "Altas": 1,
                  "Derm": 1,
                  "Black": 1}
local_weight_dict = {"ISIC": 1,
                  "Altas": 1,
                  "Derm": 1,
                  "Black": 1}


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, raw_dataset, transform=None, transform_no_norm=None):
        super(AugmentedDataset, self).__init__()

        # keep the raw dataset as it is, since it can be shared among multiple AugmentedDataset()
        self.dataset = raw_dataset
        self.transform = transform
        self.transform_no_norm = transform_no_norm
        self.targets = []
        # Subset.ConcatDataset
        for dataset in raw_dataset.dataset.datasets:
            self.targets.extend(dataset.targets)
        self.targets = np.array(self.targets)[np.array(raw_dataset.indices)].tolist()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_raw, label = self.dataset.__getitem__(index)
        if self.transform:
            img = self.transform(img_raw)

        if self.transform_no_norm:
            img_no_norm = self.transform_no_norm(img_raw)
            return img, img_no_norm, label
        else:
            return img, label
def iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

# Dataset Class Definition


class Distance_Original(Dataset):
    def __init__(self, label_file_path, image_folder_path, transform=None):
        # Parse the labeling file
        with open(label_file_path, 'r') as file:
            lines = file.readlines()

        # Split lines into image filenames and distances
        self.image_filenames = []
        self.distances = []
        for line in lines:
            filename, distance = line.split()
            self.image_filenames.append(filename)
            self.distances.append(int(distance[:-2]))

        self.image_folder_path = image_folder_path
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.image_folder_path, self.image_filenames[idx])
        image = Image.open(image_path).convert('RGB')

        # Get corresponding distance
        distance = self.distances[idx]

        # Apply transformation if any
        if self.transform:
            image = self.transform(image)

        return image, distance

from torch.utils.data import random_split
from torchvision import transforms

# Apply standard transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size of AlexNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset with transformations
dataset = Distance_Original(label_file_path="D:/Distance_dataset/label_Original_Images.txt",
                            image_folder_path="D:/Distance_dataset/Original_Images/",
                            transform=transform)

# Split the dataset
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

def get_dataset():
    # return train_data_loader, test_data_set, dict_users_train, dict_users_test
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the input size of AlexNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = Distance_Original(
        label_file_path=r"D:/Distance_dataset/label_Original_Images.txt",
        image_folder_path=r"D:/Distance_dataset/Original_Images/",
        transform=transform)  # Change to your own path

    # Split the dataset
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


    #return train_dataset, test_dataset, dict_users_train, dict_users_test
    return train_dataset, test_dataset



def binary_train(model, train_epochs, train_loader, test_dataset,optimizer,args, criterion,architecture, device):
    # train for binary model
    model.to(device)
    print('Train begining...')
    timestamp = str(int(time.time()))
    for epoch in range(train_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            print(target)
            # sys.exit(0)

            output = model(data)
            # loss = F.nll_loss(output, target)
            loss = criterion(output.float(), target.float())
            # print(type(loss))
            # sys.exit(0)
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            # if batch_idx % 200 == 0:
            #     print("Train epoch[{}] ({}/{}): loss {:.4f}".format(epoch, batch_idx,
            #                                                         len(train_loader.dataset) // len(data) + 1 , loss))
        del data, target, output, loss
        torch.cuda.empty_cache()
        _,_=test_img(model, test_dataset,args, device,1, -1)

     # save model
     #if architecture is not None:
    save_file = '../model/effinet_' +timestamp+'_'+str(architecture)+ '.pkl'
    print("file saved")
    torch.save(model, save_file)

# def memory_info(label=None):
#     info = psutil.virtual_memory()
#     print(label)
#     print('memory_used: ', psutil.Process(os.getpid()).memory_info().rss)
#     print('memory_all:', info.total)
#     print('memory_percent: ', info.percent)
#     print('cpu_num: ', psutil.cpu_count())


def model(binary_model, args, architecture, train_data_loader, test_dataset1,
          device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(binary_model.parameters(), args.lr)
    save_file = '../efficient' + str(architecture) + '_1.pkl'.format(+ 1)
    if os.path.exists(save_file):
        binary_model = torch.load(save_file)
    else:
        binary_train(binary_model, args.train_epochs,
                     train_data_loader, test_dataset1,
                     optimizer,args, criterion, architecture, device)
    # memory_info('binary_train')
    with torch.no_grad():
        binary_model1 = copy.deepcopy(binary_model)
        accuracy0,test_loss0 = test_img(binary_model1, test_dataset1,args,device,1,-1)

    return accuracy0


def RL_reward(args, binary_model, architecture, device):
    # train and evaluate for rl reward
    dataset_isic_train, dataset_isic_test = get_dataset()

    # memory_info('get_dataset')


    dataset_train = ConcatDataset([dataset_isic_train])
    # memory_info('random_split')
    del dataset_isic_train


    gc.collect()
    # memory_info('del random_split')
    train_data_loader = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=0,
                                   drop_last=False)

    dataset_test_white_skin = ConcatDataset([dataset_isic_test])


    # memory_info('DataLoader')
    accuracy = model(binary_model, args, architecture, train_data_loader,
                      dataset_test_white_skin,
                       device)

    return (accuracy)


def parse_args():
    parser = argparse.ArgumentParser(description='NAS for binary NAS')
    # parser.add_argument('--model', type=str, default='MLP2',
    #                     help='the model type for training')
    parser.add_argument('--train-batch', type=int, default=512,
                        help='the batch size for train data')
    parser.add_argument('--test-batch', type=int, default=1,
                        help=' the batch size for test data')
    parser.add_argument('--train-epochs', type=int, default=10,
                        help='train epochs of NAS')
    parser.add_argument('--bs', type=int, default=32,
                        help='batch size of NAS')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers')
    parser.add_argument('--gpu', type=int, default=None, help="GPU ID, None for CPU")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='momentum of optimizer')
    parser.add_argument('--constraint', type=int, default=50000,
                        help='the power constraint')
    parser.add_argument('--num_classes', type=int, default=1, help="number of classes")

    args = parser.parse_args()
    return args

def num_correct_pred(output, target, class_num=None):
    if class_num != None:
        mask = (target == class_num)
        output = output[mask]
        target = target[mask]

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    return correct.sum(), correct


def test_img(net_g, datatest, args,device, return_probs=True, user_idx=-1):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    criterion = nn.MSELoss()
    data_loader = DataLoader(datatest, batch_size=args.bs, num_workers=args.num_workers,shuffle=False,drop_last=True)
    l = len(data_loader)
    probs = []
    targets = []
    num_data_per_class = [0] * args.num_classes
    num_correct_data_per_class = [0] * args.num_classes
    # Calculate the RMSE on the test set
    squared_errors = []
    for idx, (data, target) in enumerate(tqdm(data_loader)):
        if args.gpu != -1:
            data, target = data.to(device), target.to(device)
        log_probs = net_g(data)
        squared_errors.append((log_probs - target).pow(2))

        targets.append(target)
        print("*********************||*****************************")

        probs.append(log_probs.detach())

        # sum up batch loss
        loss = criterion(log_probs, target)
        test_loss+=loss.item()
        print("test_loss:", test_loss)
        # get the index of the max log-probability
        #########softmax

        # Concatenate all squared errors and take square root of the mean
    rmse = torch.cat(squared_errors).mean().sqrt().item()
    if return_probs:
        return rmse, test_loss
    else:
        return rmse, test_loss


# if __name__ == '__main__':
#     args = parse_args()
#     if args.model == 'LeNet':
#         architecture = [(1, 5, 6),(1, 5, 16),(2, 2)]
#         model = LeNet(architecture)
#     elif 'MLP' in args.model:
#         if args.model == 'MLP3':
#             model = MLP3()
#         elif args.model == 'MLP2':
#             model = MLP2()
#         elif args.model == 'MLP1':
#             model = MLP1()
#     accuracy = RL_reward(args, model)





