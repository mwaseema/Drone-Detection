import os

# Number of classes in the dataset, in my case background and foreground (drone)
n_classes = 2
# Total training epochs
epochs = 100
# number of samples which fit on GPU memory at a time and gradient is calculated for all at once
batch_size = 3
# number of samples randomly selected for training in an epoch
steps_per_epoch = 1024

# absolute folder path
my_base_path = ""
dataset_base_path = "drones_dataset"

dataset_base_path = os.path.join(my_base_path, dataset_base_path)

model_used = 'Model_name'

dataset_links = {
    'train': {
        # folder where annotation masks are stored
        'annotations': os.path.join(dataset_base_path, 'train', 'gt_cropped'),
        # folder where images are stored
        'images': os.path.join(dataset_base_path, 'train', 'images_cropped'),
        # folder where to save model weights after every epoch
        'temp': os.path.join(dataset_base_path, 'temp', model_used, model_used),
    },
}

# if you are going to fine tune the model with given weights, set following is_finetune to True
# and set absolute path to previously saved weight
is_finetune = False
# absolute path to weight file
old_weight_path = ''
