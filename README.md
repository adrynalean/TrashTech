

# TrashTech

![TrashTech](https://miro.medium.com/v2/resize:fit:1400/1*bBLBnrUMDxjym9HM53qf0g.gif)


This project is a waste classification system built using PyTorch, which utilizes the ResNet18 architecture for distinguishing between various types of waste. The classifier is trained to identify and categorize waste into different classes, enabling efficient waste management.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Waste Sorting Classifier leverages a deep learning model to classify images of waste into distinct categories. This model can be integrated into various applications such as automated waste sorting systems, helping in the separation of recyclables from general waste.

## Dataset

The dataset used for training and testing the model is sourced from Kaggle:

- **Dataset Name**: Real Waste
- **Dataset URL**: [Real Waste on Kaggle](https://www.kaggle.com/datasets/joebeachcapital/realwaste)
- **Classes**:
  - Cardboard
  - Food Organics
  - Glass
  - Metal
  - Miscellaneous Trash
  - Paper
  - Plastic
  - Textile Trash
  - Vegetation

The dataset contains images of various waste items, categorized into the above classes. The dataset was split into training and validation sets in an 80:20 ratio.

## Model Architecture

This project uses the **ResNet18** model, a deep residual learning framework. The ResNet18 model was chosen for its efficiency and effectiveness in image classification tasks. The final fully connected layer of the model was modified to match the number of waste classes.

## Training

The model was trained using the following configuration:

- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam Optimizer with a learning rate of 0.001
- **Epochs**: 25
- **Batch Size**: 32

### Key Transformations:

- **Training Set**:
  - RandomResizedCrop(224)
  - RandomHorizontalFlip()
  - ToTensor()
  - Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

- **Validation Set**:
  - Resize(256)
  - CenterCrop(224)
  - ToTensor()
  - Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

## Evaluation

After training, the model was evaluated on the validation dataset. The overall accuracy achieved on the validation set was **76.34%**.

## Results

The model's performance was visualized by making predictions on validation images and comparing them with the ground truth labels. The visualizations showed that the model performed well in distinguishing between different classes of waste.

## Installation

To set up and run the project, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/waste-sorting-classifier.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle and place it in the appropriate directory.

4. Run the training script:

   ```bash
   python train.py
   ```

## Usage

To use the trained model for classifying waste images:

1. Load the model:

   ```python
   model = models.resnet18()
   model.load_state_dict(torch.load('resnet_waste_classifier.pth'))
   model.eval()
   ```

2. Make predictions on new images:

   ```python
   output = model(input_image)
   _, predicted_class = torch.max(output, 1)
   print(f'Predicted class: {class_names[predicted_class]}')
   ```

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please fork the repository and create a pull request with your changes.
