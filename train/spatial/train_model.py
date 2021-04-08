import sys
import time
from os import path

import keras_segmentation

if path.abspath(path.join(path.dirname(__file__), '..', '..')) not in sys.path:
    sys.path.append(path.abspath(path.join(path.dirname(__file__), '..', '..')))

from train.spatial.config import dataset_links, n_classes, epochs, batch_size, steps_per_epoch, is_finetune, \
    old_weight_path
from utils.functions import *


def main():
    # creating folder for storing model weights
    os.makedirs(os.path.abspath(os.path.join(dataset_links['train']['temp'], '..')), exist_ok=True)

    model = keras_segmentation.models.pspnet.pspnet_50_with_weighted_output(n_classes=n_classes)
    model = prepare_fine_tune(model, is_finetune, old_weight_path)

    # Training on data
    start_time = time.time()
    print(f"Starting training on data with {epochs} epochs")
    model.train(
        train_images=dataset_links['train']['images'],
        train_annotations=dataset_links['train']['annotations'],
        checkpoints_path=dataset_links['train']['temp'],
        epochs=epochs,
        verify_dataset=False,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
    )
    training_time = time.time() - start_time
    print("\nCompleted training in {} seconds".format(round(training_time)))


if __name__ == '__main__':
    main()
