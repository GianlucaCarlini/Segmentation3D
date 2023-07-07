# Segmentation3D
Just a repo to store models and utilities to work with 3D images in pytorch.

## Loaders

The [PatchDataLoader](./loaders/lazy_loaders.py) is a pytorch data loader that loads random patches from a 3D image. It can be used to train a 3D segmentation model with random patches. It is a lazy loader, so it does not load the whole image in memory, but only the patches that are needed.

## Models

The [Unet3D](./models/lightning_models.py) is a pytorch lightning module that implements a 3D Unet. It can be trained using 3D patches of the original volume. Once trained, the prediction is performed by sliding the model over the volume and averaging the predictions of the overlapping patches. In other words, it uses a 3D sliding window to predict the segmentation of the whole volume.

## Modules

The modules folder contains some pytorch modules to instantiate standard [Residual Blocks](./modules/blocks.py) with some flexibility and build a [3D segmentation model](./modules/segmentation_model.py) with custom number of layers and filters per layer.

## Working with 3D images

### Window operations
The [utils](./utils) folder contains some utilities to work with 3D images. The [window_operations](./utils/window_operations.py) module contains two functions; the first can be used to pad a 3D volume so that a sliding window can be used on it, while the second function calculates the coordinates of all the possible patches that can be extracted from the padded volume. Both the function can work with either a numpy array or a torch tensor. However, even in the case of a tensor, they expect the input to be 3D with shape (z, y, x) (i.e., depth, height, width).

### Prediction loop

The [prediction_loop](./utils/prediction_loops.py) module contains a function that can be used to predict the segmentation of a 3D volume using a sliding window. It uses the functions in window operations to extract the patch coordinates, then performs the prediction patch by patch in a for loop. The verbose attribute can be used to make the function print the number of patches already predicted.



