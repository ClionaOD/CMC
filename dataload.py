from dataset import twoImageFolderInstance
import torch
from torchvision import transforms, datasets
from dataset import RGB2Lab

data_folder = '/home/clionaodoherty/Desktop/fyp2020/stimuli/'
color_transfer = RGB2Lab()

train_transform = transforms.Compose([transforms.RandomResizedCrop(224, 
        scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        color_transfer,
        transforms.ToTensor(),
])

train_dataset = twoImageFolderInstance(data_folder, time_lag=3, transform=train_transform)
train_sampler = None

# train loader
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1, shuffle=(train_sampler is None), sampler=train_sampler)

n_data = len(train_dataset)
print('number of samples: {}'.format(n_data))

for idx, [(inputs1, _, index), (inputs2, _, lagged_index)] in enumerate(train_loader):

    bsz = inputs1.size(0)
    inputs1 = inputs1.float()
    inputs2 = inputs2.float()
    if torch.cuda.is_available():
        index = index.cuda(non_blocking=True)
        inputs = inputs.cuda()