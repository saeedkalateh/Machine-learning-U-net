YCB Dataset U-Net Model for Semantic Segmentation

Introduction: This repository contains the implementation of a U-Net model for semantic segmentation using the YCB dataset. Semantic segmentation is a computer vision task where each pixel in an image is classified into a specific class, enabling precise object localization and understanding.

Dataset: The YCB dataset is utilized for training and evaluation. The dataset consists of RGB images and corresponding ground truth masks. RGB images are stored in the rgb directory, and the masks are stored in the masks/gt directory.

Model Architecture: The U-Net architecture is employed for semantic segmentation. The model consists of a contracting path to capture context and a expansive path for precise localization. The contracting path involves convolutional and max-pooling layers, while the expansive path uses transposed convolutions for upsampling. Skip connections are incorporated to retain spatial information.

Training Procedure: The RGB images and masks are loaded, preprocessed, and split into training and validation sets. Data augmentation techniques such as random flips and rotations can be applied during training to enhance model generalization.

Model Training: The model is trained using the Adam optimizer and binary cross-entropy loss. Early stopping and model checkpointing are employed to prevent overfitting and save the best model based on validation loss.

Results: Training and validation loss, as well as accuracy, are plotted over epochs to visualize the model's learning progress and performance.

libraries:

    TensorFlow
    NumPy
    Matplotlib
    scikit-image
    tqdm

