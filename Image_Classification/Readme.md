# Image Classification with PyTorch Models

## Project Overview
This project implements several deep learning models for image classification using PyTorch, a popular deep learning framework. The models included are AlexNet, Inception (GoogleNet), LeNet5, VGG (pretrained), and VGG (transfer learning). These models are designed to classify images into predefined categories by leveraging convolutional neural networks (CNNs). The project uses the CIFAR-10 dataset (10 classes, 60,000 images) as the primary dataset for training and evaluation, though other datasets like ImageNet or MNIST can be adapted.
The project includes:

- **Model Implementation:** PyTorch scripts for AlexNet, Inception, LeNet5, VGG (pretrained), and VGG (transfer learning).
- **Training and Evaluation:** Scripts to train models from scratch (where applicable) and evaluate performance using metrics like accuracy.
- **Transfer Learning:** A VGG model fine-tuned using transfer learning for improved performance on CIFAR-10.
- **Dataset:** CIFAR-10 is used by default, with preprocessing steps like normalization and data augmentation.
- **Deployment:** Example scripts to run inference on new images using trained models.

## Installation
To set up the project locally, follow these steps:

1. **Clone the Repository:**
````git clone https://github.com/your-username/image-classification-pytorch.git
cd image-classification-pytorch
````
2. **Create a Virtual Environment (optional but recommended):**
````python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
````

3. **Install Dependencies:Install the required Python packages using the provided requirements.txt:**
````
pip install -r requirements.txt
````

The ``requirements.txt`` includes:

- ``torch`` (PyTorch for model implementation)
- ``torchvision`` (for datasets and pretrained models)
- ``numpy`` (for numerical operations)
- ``matplotlib`` (for visualization, optional)
- Other dependencies as needed

4. **Download the Dataset:** The CIFAR-10 dataset is automatically downloaded by ``PyTorch’s torchvision.datasets.CIFAR10`` when running the scripts. Ensure an internet connection for the first run. Alternatively, manually download it from Kaggle and place it in the ``data/`` directory.

5. **Hardware Requirements:**

- A GPU is recommended for faster training (CUDA-compatible if using PyTorch with GPU support).
- Minimum 8GB RAM for CPU-based training.


## Usage

1. **Train a Model:** Train one of the models (e.g., AlexNet) on CIFAR-10:
````
python train.py --model alexnet
````

- Supported models: alexnet, inception, lenet5, vgg_pretrained, vgg_transfer.
- The script handles data loading, preprocessing, training, and saves the trained model to the models/ directory.

2. **Evaluate a Model:** Evaluate a trained model’s performance:
````
python evaluate.py --model alexnet --checkpoint models/alexnet.pth
````

- Outputs metrics like accuracy on the test set.

3. **Run Inference:** Use a trained model to classify a single image:
````
python infer.py --model alexnet --checkpoint models/alexnet.pth --image path/to/image.jpg
````

- The script outputs the predicted class label.


4. **Customize Training:** Modify hyperparameters (e.g., learning rate, epochs) in config.py or pass them via command-line arguments:
````
python train.py --model vgg_transfer --lr 0.001 --epochs 20
````

## Models
The repository includes the following image classification models implemented in PyTorch:

- **AlexNet:** A deep CNN with multiple convolutional and fully connected layers, originally designed for ImageNet.
- **Inception (GoogleNet):** Uses inception modules to capture multi-scale features efficiently.
- **LeNet5:** A lightweight CNN suitable for simpler datasets like MNIST or CIFAR-10.
- **VGG (Pretrained):** A VGG model pretrained on ImageNet, used as-is or fine-tuned.
- **VGG (Transfer Learning):** A VGG model with pretrained weights, fine-tuned on CIFAR-10 for improved performance.

Each model is implemented in a modular script under the ``models/`` directory, with training and evaluation scripts supporting all models.


## File Structure
````
image-classification-pytorch/
├── data/
│   └── cifar-10/               # CIFAR-10 dataset (downloaded automatically)
├── models/
│   ├── alexnet.py              # AlexNet model definition
│   ├── inception.py            # Inception (GoogleNet) model definition
│   ├── lenet5.py               # LeNet5 model definition
│   ├── vgg_pretrained.py       # VGG pretrained model
│   ├── vgg_transfer.py         # VGG transfer learning model
│   └── saved/                  # Saved model checkpoints (e.g., alexnet.pth)
├── scripts/
│   ├── train.py                # Script to train models
│   ├── evaluate.py             # Script to evaluate models
│   ├── infer.py                # Script for inference on single images
│   └── config.py               # Configuration for hyperparameters
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
````

## Datasets
The project uses the CIFAR-10 dataset by default, which includes 60,000 32x32 color images across 10 classes (e.g., airplane, cat, dog). Other datasets mentioned in the reference document (e.g., ImageNet, MNIST, Fashion-MNIST) can be used by modifying the data loading logic in ``train.py`` and ``evaluate.py``.

## Metrics
Model performance is evaluated using:

- **Accuracy:** The percentage of correctly classified images in the test set.
- Additional metrics (e.g., precision, recall, F1-score) can be implemented by extending ``evaluate.py``.

## Notes

- **Transfer Learning:** The VGG transfer learning model uses pretrained weights from ImageNet, with the final layer fine-tuned for CIFAR-10. This approach typically yields better performance than training from scratch.
- **Hardware:** Training on a GPU significantly reduces computation time. Ensure CUDA is installed for PyTorch GPU support.
- **Extensibility:** The codebase is modular, allowing easy addition of new models or datasets. Update train.py and evaluate.py to include new model definitions.
- **Data Preprocessing:** The scripts include standard preprocessing (normalization, data augmentation like random flips and rotations) for CIFAR-10.

## Contributing
Contributions are welcome! Please:

1. Fork the repository.
2. Create a feature branch ``(git checkout -b feature/your-feature)``.
3. Commit your changes ``(git commit -m 'Add your feature')``.
4. Push to the branch ``(git push origin feature/your-feature)``.
5. Open a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For questions or suggestions, please open an issue on the GitHub repository or contact [shamreen.tabassum@mailbox.tu-dresden.de].
