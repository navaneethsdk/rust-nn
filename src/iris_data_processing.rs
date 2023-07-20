use csv::Error;
use rand::seq::SliceRandom;
use serde::Deserialize;
use std::fs::File;
use std::io::Read;

use crate::nn::NeuralNetwork;

#[derive(Debug, Deserialize)]
struct IrisData {
    sepal_length: f64,
    sepal_width: f64,
    petal_length: f64,
    petal_width: f64,
    class: String,
}

pub fn read_iris_data(file_path: &str) -> Result<Vec<(Vec<f64>, Vec<f64>)>, Error> {
    let mut file = File::open(file_path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let mut reader = csv::Reader::from_reader(contents.as_bytes());
    let mut data = Vec::new();

    for result in reader.deserialize::<IrisData>() {
        let record: IrisData = result?;
        let features = vec![
            record.sepal_length,
            record.sepal_width,
            record.petal_length,
            record.petal_width,
        ];
        let labels = match record.class.as_str() {
            "Iris-setosa" => vec![1.0, 0.0, 0.0],
            "Iris-versicolor" => vec![0.0, 1.0, 0.0],
            "Iris-virginica" => vec![0.0, 0.0, 1.0],
            _ => vec![],
        };

        if !labels.is_empty() {
            data.push((features, labels));
        }
    }

    Ok(data)
}

pub fn split_train_test(
    iris_data: Vec<(Vec<f64>, Vec<f64>)>,
    test_ratio: f64,
) -> (Vec<(Vec<f64>, Vec<f64>)>, Vec<(Vec<f64>, Vec<f64>)>) {
    let mut rng = rand::thread_rng();
    let mut data = iris_data.clone();
    data.shuffle(&mut rng);

    let test_size = (data.len() as f64 * test_ratio).ceil() as usize;
    let test_data = data[..test_size].to_vec();
    let train_data = data[test_size..].to_vec();

    (train_data, test_data)
}

fn argmax(vector: &Vec<f64>) -> Vec<f64> {
    let max_value = vector.iter().max_by(|a, b| a.total_cmp(b));
    let mut argmax_vector: Vec<f64> = vec![];
    for value in vector {
        if value == max_value.unwrap() {
            argmax_vector.push(1.0);
        } else {
            argmax_vector.push(0.0);
        }
    }
    argmax_vector
}

fn matching_vectors(vec1: &Vec<f64>, vec2: &Vec<f64>) -> bool {
    let matching_count = vec1
        .iter()
        .zip(vec2.iter())
        .filter(|&(vec1_element, vec2_element)| &vec1_element == &vec2_element)
        .count();
    matching_count == vec1.len()
}

pub fn performance_evaluation(
    test_data: &Vec<(Vec<f64>, Vec<f64>)>,
    neural_network: &NeuralNetwork,
) {
    let mut correct_predictions = 0;
    for record in test_data {
        let prediction = neural_network.feedforward(&record.0);
        let prediction_argmax = argmax(&prediction[1]);
        if matching_vectors(&prediction_argmax, &record.1) {
            correct_predictions += 1;
        };
    }
    let accuracy = (correct_predictions as f32) / (test_data.len() as f32);
    println!("Accuracy is: {}", accuracy);
}
