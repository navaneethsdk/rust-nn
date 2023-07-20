mod iris_data_processing;
mod nn;

use iris_data_processing::*;
use nn::*;
fn main() {
    // Iris dataset
    let iris_dataset: Vec<(Vec<f64>, Vec<f64>)> =
        read_iris_data("src/data/Iris.csv").expect("Failed to read Iris data");
    println!("[info] Loaded Iris Dataset");
    let test_ratio = 0.2;
    let (train_data, test_data) = split_train_test(iris_dataset, test_ratio);
    // Create and train the neural network
    let mut neural_network = NeuralNetwork::new(4, 1, 3);
    let learning_rate = 0.1;
    let epochs = 1000;

    for _ in 0..epochs {
        for (input, target) in &train_data {
            neural_network.train(&input, &target, learning_rate);
        }
    }
    println!("[info] Completed training process");

    // Test the trained neural network

    // Uncomment the below code snippet to see the prediction for a specific test sample
    // let test_input = vec![5.9, 3.0, 5.1, 1.8]; // Unknown Iris
    // let prediction = neural_network.feedforward(&test_input);
    // println!("Prediction for {:?}: {:?}", test_input, prediction[1]);

    performance_evaluation(&test_data, &neural_network);
}
