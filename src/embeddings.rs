use bytemuck::cast_slice;
use ndarray::Array2;
use rand::Rng;
use rand::distr::{Distribution, Uniform};
use rand_distr::Normal;
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors, serialize_to_file};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

pub struct Embeddings {
    matrix: Array2<f32>,
    vocab_size: usize,
    embedding_dim: usize,
}

impl Embeddings {
    /// Initialization of a new embedding matrix (uniform distribution with random filling)
    ///
    /// # Parameters
    /// - `vocab_size`: Size of the vocabulary (number of tokens).
    /// - `embedding_dim`: Dimensionality of the embeddings (length of the vector for each token).
    ///
    /// # Returns
    /// An `Embeddings` instance with a matrix filled with values from a uniform distribution in the range `[-1.0, 1.0]`.
    pub fn new_uniform(vocab_size: usize, embedding_dim: usize) -> Self {
        let mut rng = rand::rng();
        let uniform = Uniform::new_inclusive(-1.0, 1.0).expect("Fail to create a new Uniform instance");

        let matrix = Array2::from_shape_fn((embedding_dim, vocab_size), |_| uniform.sample(&mut rng));

        Self {
            matrix,
            vocab_size,
            embedding_dim,
        }
    }

    /// Initializing a new embedding matrix (Gaussian distribution)
    ///
    /// # Parameters
    /// - `vocab_size`: Size of the vocabulary (number of tokens).
    /// - `embedding_dim`: Dimensionality of the embeddings (length of the vector for each token).
    ///
    /// # Returns
    /// An `Embeddings` instance with a matrix filled with values from the normal distribution.
    pub fn new_gaussian(vocab_size: usize, embedding_dim: usize, mean: f32, std_dev: f32) -> Self {
        let mut rng = rand::rng();
        let normal = Normal::new(mean, std_dev).expect("Fail to create a new Normal instance");

        let matrix = Array2::from_shape_fn((embedding_dim, vocab_size), |_| normal.sample(&mut rng));

        Self {
            matrix,
            vocab_size,
            embedding_dim,
        }
    }

    /// Initializing a new embedding matrix (Xavier (Glorot))
    ///
    /// # Parameters
    /// - `vocab_size`: Size of the vocabulary (number of tokens).
    /// - `embedding_dim`: Dimensionality of the embeddings (length of the vector for each token).
    ///
    /// # Returns
    /// An `Embeddings` instance with a matrix filled with values initialized by the Xavier method.
    pub fn new_xavier(vocab_size: usize, embedding_dim: usize) -> Self {
        let mut rng = rand::rng();
        let std_dev = (6.0 / (vocab_size as f32 + embedding_dim as f32)).sqrt();

        let matrix = Array2::from_shape_fn((embedding_dim, vocab_size), |_| rng.random_range(-std_dev..std_dev));

        Self {
            matrix,
            vocab_size,
            embedding_dim,
        }
    }

    /// Getter for embedding matrix
    ///
    /// # Returns
    /// A reference to the `Array2<f32>` matrix containing the embeddings.
    pub fn get_matrix(&self) -> &Array2<f32> {
        &self.matrix
    }

    /// Transforming a vector of tokens into an embedding matrix
    ///
    /// # Parameters
    /// - `tokens`: Array of token indices to convert to embeddings.
    ///
    /// # Returns
    /// - `Ok(Array2<f32>)`: Matrix of embeddings, where each column corresponds to a token embedding.
    /// - `Err(String)`: Error if any token is outside the dictionary.
    pub fn tokens_to_embeddings(&self, tokens: &[usize]) -> Result<Array2<f32>, String> {
        let mut embeddings = Array2::zeros((self.embedding_dim, tokens.len()));

        for (i, &token) in tokens.iter().enumerate() {
            if token >= self.vocab_size {
                return Err("Token is out of vocabulary bounds".to_string());
            }

            embeddings.column_mut(i).assign(&self.matrix.column(token));
        }

        Ok(embeddings)
    }

    /// Saving the embedding matrix to a file
    ///
    /// # Parameters
    /// - `embeddings`: The `Array2<f32>` embedding matrix to save.
    /// - `file_path`: The path to the file where the matrix will be saved.
    ///
    /// # Returns
    /// - `Ok(())`: If saving was successful.
    /// - `Err(String)`: Error if there was a problem creating the tensor, serializing it, or writing it to the file.
    pub fn save_embeddings_to_file(embeddings: &Array2<f32>, file_path: &str) -> Result<(), String> {
        let shape = embeddings.shape().to_vec();

        let data = embeddings
            .as_slice()
            .unwrap()
            .iter()
            .flat_map(|float| float.to_le_bytes())
            .collect::<Vec<_>>();

        let tensor = TensorView::new(Dtype::F32, shape, &*data).map_err(|err| format!("Tensor create error: {err}"))?;

        let mut tensors_map = HashMap::new();
        tensors_map.insert("embeddings", tensor);

        serialize_to_file(tensors_map, &None, file_path.as_ref())
            .map_err(|err| format!("Failed serialize error: {err}"))?;

        println!("Embedding matrix successfully saved to file: {file_path}");
        Ok(())
    }

    /// Loading embedding matrix from file
    ///
    /// # Parameters
    /// - `file_path`: Path to the file from which to load the matrix.
    ///
    /// # Returns
    /// - `Ok(Array2<f32>)`: The embedding matrix loaded from the file.
    /// - `Err(String)`: Error if there was a problem opening the file, reading, deserializing, or converting the data.
    pub fn load_embeddings_from_file(file_path: &str) -> Result<Array2<f32>, String> {
        let mut buffer = Vec::new();

        File::open(file_path)
            .map_err(|err| format!("File open error: {err}"))?
            .read_to_end(&mut buffer)
            .map_err(|err| format!("Read error: {err}"))?;

        let safe_tensors = SafeTensors::deserialize(&buffer).map_err(|err| format!("Deserialize error: {err}"))?;

        let tensor = safe_tensors
            .tensor("embeddings")
            .map_err(|err| format!("Tensor error: {err}"))?;

        if tensor.dtype() != Dtype::F32 {
            return Err(format!("Invalid Dtype: expected F32, got {:?}", tensor.dtype()));
        }

        let data_bytes = tensor.data();
        let data_f32 = cast_slice(data_bytes);
        let shape = tensor.shape();

        let embeddings = Array2::from_shape_vec((shape[0], shape[1]), data_f32.to_vec())
            .map_err(|err| format!("Conversion error: {err}"))?;

        println!("Embedding matrix successfully loaded from file: {file_path}");
        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::remove_file;
    use std::io::Write;

    #[test]
    fn test_embeddings_new_uniform() {
        let vocab_size = 10;
        let embedding_dim = 5;

        let embeddings_uniform = Embeddings::new_uniform(vocab_size, embedding_dim);

        assert_eq!(embeddings_uniform.matrix.shape(), &[embedding_dim, vocab_size]);
        assert_eq!(embeddings_uniform.vocab_size, vocab_size);
        assert_eq!(embeddings_uniform.embedding_dim, embedding_dim);

        for &value in embeddings_uniform.matrix.iter() {
            assert!(value >= -1.0 && value <= 1.0);
        }
    }

    #[test]
    fn test_embeddings_new_gaussian() {
        let mean = 0.0f32;
        let std_dev = 0.01f32;
        let vocab_size = 10;
        let embedding_dim = 5;

        let embeddings_gaussian = Embeddings::new_gaussian(vocab_size, embedding_dim, mean, std_dev);

        assert_eq!(embeddings_gaussian.matrix.shape(), &[embedding_dim, vocab_size]);
        assert_eq!(embeddings_gaussian.vocab_size, vocab_size);
        assert_eq!(embeddings_gaussian.embedding_dim, embedding_dim);

        let sum = embeddings_gaussian.matrix.iter().sum::<f32>();
        let avg = sum / (vocab_size * embedding_dim) as f32;

        assert!((avg - mean).abs() < 0.1);
    }

    #[test]
    fn test_embeddings_new_xavier() {
        let vocab_size = 10;
        let embedding_dim = 5;

        let embeddings_xavier = Embeddings::new_xavier(vocab_size, embedding_dim);

        assert_eq!(embeddings_xavier.matrix.shape(), &[embedding_dim, vocab_size]);
        assert_eq!(embeddings_xavier.vocab_size, vocab_size);
        assert_eq!(embeddings_xavier.embedding_dim, embedding_dim);

        for &value in embeddings_xavier.matrix.iter() {
            assert!(value >= -1.0 && value <= 1.0);
        }
    }

    #[test]
    fn test_tokens_to_embeddings() {
        let vocab_size = 10;
        let embedding_dim = 5;

        let embeddings = Embeddings::new_uniform(vocab_size, embedding_dim);

        let tokens = vec![0, 3, 7];
        let result = embeddings.tokens_to_embeddings(&tokens).unwrap();

        assert_eq!(result.shape(), &[embedding_dim, tokens.len()]);

        for (i, &token) in tokens.iter().enumerate() {
            let expected_embedding = embeddings.matrix.column(token);
            let actual_embedding = result.column(i);

            for (expected, actual) in expected_embedding.iter().zip(actual_embedding.iter()) {
                assert_eq!(*expected, *actual);
            }
        }
    }

    #[test]
    fn test_tokens_to_embeddings_out_of_bounds() {
        let vocab_size = 10;
        let embedding_dim = 5;

        let embeddings = Embeddings::new_uniform(vocab_size, embedding_dim);

        let tokens = vec![1, 5, 15];
        let result = embeddings.tokens_to_embeddings(&tokens);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Token is out of vocabulary bounds");
    }

    #[test]
    fn test_tokens_to_embeddings_empty() {
        let vocab_size = 10;
        let embedding_dim = 5;

        let embeddings = Embeddings::new_uniform(vocab_size, embedding_dim);

        let tokens = vec![];
        let result = embeddings.tokens_to_embeddings(&tokens).unwrap();

        assert_eq!(result.shape(), &[embedding_dim, 0]);
    }

    #[test]
    fn test_get_matrix() {
        let vocab_size = 10;
        let embedding_dim = 5;

        let embeddings = Embeddings::new_uniform(vocab_size, embedding_dim);

        let matrix = embeddings.get_matrix();

        assert_eq!(matrix.shape(), &[embedding_dim, vocab_size]);
        assert_eq!(*matrix, embeddings.matrix);
    }

    #[test]
    fn test_save_and_load_embeddings() {
        let file_path = "test_embeddings.safetensors";
        let embeddings = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        Embeddings::save_embeddings_to_file(&embeddings, file_path).expect("Failed to save embeddings");

        let loaded_embeddings = Embeddings::load_embeddings_from_file(file_path).expect("Failed to load embeddings");

        assert_eq!(embeddings, loaded_embeddings);

        remove_file(file_path).expect("Failed to delete test file");
    }

    #[test]
    fn test_load_invalid_file() {
        let file_path = "invalid_embeddings.safetensors";
        let mut file = File::create(file_path).expect("Failed to create test file");

        file.write_all(b"invalid data").expect("Failed to write test data");

        let result = Embeddings::load_embeddings_from_file(file_path);

        assert!(result.is_err());

        remove_file(file_path).expect("Failed to delete test file");
    }

    #[test]
    fn test_save_invalid_path() {
        let file_path = "non_existent_directory/test_embeddings.safetensors";
        let embeddings = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let result = Embeddings::save_embeddings_to_file(&embeddings, file_path);
        assert!(result.is_err());
    }
}
