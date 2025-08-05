# Handwritten Digit Classifier using PyTorch

### Project Description
This repository contains a simple feedforward neural network built with **PyTorch** to classify handwritten digits from the **MNIST dataset**. This project demonstrates an understanding of foundational deep learning concepts and the practical implementation of a neural network for a classification task.

### Key Features
* **Neural Network Architecture:** A simple feedforward network with two hidden layers.
* **Dataset:** Utilizes the industry-standard MNIST dataset, consisting of 60,000 training images and 10,000 test images of handwritten digits (0-9).
* **Training & Evaluation:** The model is trained using **Cross-Entropy Loss** and the **Adam optimizer**. It's evaluated on a separate test set to measure its generalization performance.
* **High Accuracy:** Achieves an accuracy of **over 95%** on the MNIST test dataset.

### Technologies Used
* **PyTorch**: The deep learning framework used for building and training the neural network.
* **torchvision**: Used for convenient loading of the MNIST dataset.
* **Python**: The core programming language.

### How to Run the Project
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Puliya07/mnist-pytorch-classifier.git](https://github.com/Puliya07/mnist-pytorch-classifier.git)
    cd mnist-pytorch-classifier
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the code in jupyter notebook**

    The script will automatically download the MNIST dataset, train the model, and print the final test accuracy.

### What I Learned
* **Data Loading:** Gained hands-on experience with `torchvision` and `DataLoader` for efficient data handling and batching.
* **Model Definition:** Implemented a neural network from scratch using PyTorch's `nn.Module` and layers like `nn.Linear` and `ReLU`.
* **Training Loop:** Understood the end-to-end process of training, including the forward pass, loss calculation, backpropagation (`loss.backward()`), and weight updates (`optimizer.step()`).
* **Evaluation Metrics:** Learned how to evaluate a classification model's performance using accuracy on a dedicated test set.
