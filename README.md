# Iris Classification Neural Network
This is a simple binary Rust crate that implements a basic classification neural network from scratch for the Iris dataset. The application uses `cargo run` command to load the dataset, split it into a train and test set, trains the model, and prints the accuracy metric on the screen.

## Prerequisites
Before running the application, make sure you have Rust installed on your system. If you don't have Rust installed, you can follow the official installation guide: https://www.rust-lang.org/tools/install

## Installation
Clone this repository to your local machine:
```
git clone https://github.com/your-username/iris_classification.git
cd iris_classification
```
Build the Rust binary:
```
cargo build --release
```

## Usage
To run the application and train the neural network on the Iris dataset, use the following command:
```
cargo run
```
The application will automatically load the Iris dataset, split it into a train and test set, train the model, and then print the accuracy metric on the screen.

## Dataset
The Iris dataset used in this project is a classic dataset in machine learning and consists of three classes of iris plants with 50 samples each. Each sample has four features (sepal length, sepal width, petal length, and petal width). The classes to predict are 'setosa', 'versicolor', and 'virginica'.

Neural Network Architecture
The neural network used in this project is a simple feedforward neural network with one hidden layer. The architecture is as follows:

Input layer: 4 nodes (corresponding to the four features of the Iris dataset)
Hidden layer: variable number of nodes (configurable in the source code)
Output layer: 3 nodes (corresponding to the three classes of iris plants)

## Configuration
You can modify the configuration parameters in the source code to experiment with different settings. Some of the configurable options include:

- Number of nodes in the hidden layer
- Learning rate
- Number of training epochs

Feel free to tweak these parameters and observe how they affect the model's performance.

## Contributing
If you find any issues with this crate or want to improve it, feel free to open an issue or submit a pull request on the GitHub repository: https://github.com/navaneethsdk/rust-nn

## License
This mini project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
This project was inspired by the desire to understand the basics of neural networks and their implementation from scratch. It serves as a learning exercise and may not be suitable for production use.

The Iris dataset used in this project is obtained from the UCI Machine Learning Repository.

Author
- Navaneeth D (@navaneethsdk)

## References
- [Rust Programming Language](https://www.rust-lang.org/)
- [Iris Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)