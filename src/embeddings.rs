use crate::error::EmbeddingError;
use anyhow::{Error, Result};
use ndarray::{Array1, Array2, Axis};
use rand::RngCore;
use rand::SeedableRng;
use rand::distr::{Distribution, Uniform};
use rand::rngs::SmallRng;
use rand_distr::Normal;
use rayon::prelude::*;

pub struct Embeddings {
    embedding_matrix: Array2<f32>,
    vocab_size: usize,
    embedding_dim: usize,
}

impl Embeddings {
    const CHUNK_SIZE: usize = 4096;

    /// Initialization of a new embedding matrix (uniform distribution with random filling)
    ///
    /// # Parameters
    /// - `vocab_size`: Size of the vocabulary (number of tokens).
    /// - `embedding_dim`: Dimensionality of the embeddings (length of the vector for each token).
    ///
    /// # Returns
    /// - `Ok(Self)`: An `Embeddings` instance with a matrix filled with values from a uniform distribution in the range `[-1.0, 1.0]`.
    /// - `Err(anyhow::Error)`: An error is returned if the uniform distribution cannot be created, or input parameters are invalid.
    ///
    /// # Errors
    /// - `InvalidInput`: Occurs if the input parameters are invalid.
    /// - `UniformCreationFailed`: Occurs if the uniform distribution cannot be created within the specified range `[-1.0, 1.0]`.
    /// - `MatrixCreationFailed`: Occurs if the embedding matrix cannot be reshaped.
    pub fn new_uniform(vocab_size: usize, embedding_dim: usize) -> Result<Self, Error> {
        if vocab_size == 0 || embedding_dim == 0 {
            return Err(Error::from(EmbeddingError::InvalidInput(
                "Parameters vocab_size and embedding_dim must be greater than zero".to_string(),
            )));
        }

        let uniform =
            Uniform::new_inclusive(-1.0, 1.0).map_err(|err| EmbeddingError::UniformCreationFailed(err.to_string()))?;

        let random_values = Self::generate_random_values(&vocab_size, &embedding_dim, uniform);

        let embedding_matrix = Self::create_embedding_matrix(embedding_dim, vocab_size, random_values)?;

        Ok(Self {
            embedding_matrix,
            vocab_size,
            embedding_dim,
        })
    }

    /// Initializing a new embedding matrix (Gaussian distribution)
    ///
    /// # Parameters
    /// - `vocab_size`: The size of the vocabulary, i.e., the number of unique tokens that need to be embedded.
    /// - `embedding_dim`: The dimensionality of the embedding vectors, i.e., the length of the vector representation for each token.
    /// - `mean`: The mean (μ) of the normal distribution from which values are sampled.
    /// - `std_dev`: The standard deviation (σ) of the normal distribution from which values are sampled.
    ///
    /// # Returns
    /// - `Ok(Self)`: An `Embeddings` instance with a matrix filled with values from the normal distribution.
    /// - `Err(anyhow::Error)`: An error is returned if the normal distribution cannot be created, or input parameters are invalid.
    ///
    /// # Errors
    /// - `InvalidInput`: Occurs if the input parameters are invalid.
    /// - `NormalCreationFailed`: Occurs if the normal (Gaussian) distribution cannot be created with the specified mean and standard deviation.
    /// - `MatrixCreationFailed`: Occurs if the embedding matrix cannot be reshaped.
    pub fn new_gaussian(vocab_size: usize, embedding_dim: usize, mean: f32, std_dev: f32) -> Result<Self, Error> {
        if vocab_size == 0 || embedding_dim == 0 {
            return Err(Error::from(EmbeddingError::InvalidInput(
                "Parameters vocab_size and embedding_dim must be greater than zero".to_string(),
            )));
        }

        if std_dev <= 0.0 {
            return Err(Error::from(EmbeddingError::InvalidInput(
                "Standard deviation must be positive".to_string(),
            )));
        }

        let normal = Normal::new(mean, std_dev).map_err(|err| EmbeddingError::NormalCreationFailed(err.to_string()))?;

        let random_values = Self::generate_random_values(&vocab_size, &embedding_dim, normal);

        let embedding_matrix = Self::create_embedding_matrix(embedding_dim, vocab_size, random_values)?;

        Ok(Self {
            embedding_matrix,
            vocab_size,
            embedding_dim,
        })
    }

    /// Initializing a new embedding matrix (Xavier (Glorot))
    ///
    /// # Parameters
    /// - `vocab_size`: Size of the vocabulary (number of tokens).
    /// - `embedding_dim`: Dimensionality of the embeddings (length of the vector for each token).
    ///
    /// # Returns
    /// - `Ok(Self)`: An `Embeddings` instance with a matrix filled with values initialized by the Xavier method.
    /// - `Err(anyhow::Error)`: An error is returned if the input parameters are invalid (zero).
    ///
    /// # Errors
    /// - `InvalidInput`: Occurs if the input parameters are invalid.
    /// - `MatrixCreationFailed`: Occurs if the embedding matrix cannot be reshaped.
    pub fn new_xavier(vocab_size: usize, embedding_dim: usize) -> Result<Self, Error> {
        if vocab_size == 0 || embedding_dim == 0 {
            return Err(Error::from(EmbeddingError::InvalidInput(
                "Parameters vocab_size and embedding_dim must be greater than zero".to_string(),
            )));
        }

        let std_dev = (6.0 / (vocab_size as f32 + embedding_dim as f32)).sqrt();

        let uniform = Uniform::new_inclusive(-std_dev, std_dev)
            .map_err(|err| EmbeddingError::UniformCreationFailed(err.to_string()))?;

        let random_values = Self::generate_random_values(&vocab_size, &embedding_dim, uniform);

        let embedding_matrix = Self::create_embedding_matrix(embedding_dim, vocab_size, random_values)?;

        Ok(Self {
            embedding_matrix,
            vocab_size,
            embedding_dim,
        })
    }

    /// Construction of the resulting embedding matrix for an array of tokens (indices) from the original embedding matrix
    ///
    /// # Parameters
    /// - `tokens`: Array of token indices.
    ///
    /// # Returns
    /// - `Ok(Array2<f32>)`: A matrix of embeddings, where each column corresponds to the embedding of a token.
    /// - `Err(anyhow::Error)`: An error is returned if any token index is out of the vocabulary bounds.
    ///
    /// # Errors
    /// - `OutOfVocabularyError`: Occurs if any token index in the `tokens` array is out of bounds for the vocabulary.
    pub fn get_embeddings(&self, tokens: &[usize]) -> Result<Array2<f32>, Error> {
        if tokens.iter().any(|&token| token >= self.vocab_size) {
            return Err(Error::from(EmbeddingError::OutOfVocabularyError));
        }

        Ok(self.embedding_matrix.select(Axis(1), tokens).to_owned())
    }

    /// Obtaining an embedding for a specific token (index) from the initial matrix of embeddings
    ///
    /// # Parameters
    /// - `token`: Token index.
    ///
    /// # Returns
    /// - `Ok(Array1<f32>)`: Embedding (one-dimensional array) for a specific token (index).
    /// - `Err(anyhow::Error)`: Returns an error if the token index is out of bounds of the dictionary.
    ///
    /// # Errors
    /// - `OutOfVocabularyError`: Raised if the token index is out of bounds of the dictionary.
    pub fn get_embedding(&self, token: usize) -> Result<Array1<f32>, Error> {
        if token >= self.vocab_size {
            return Err(Error::from(EmbeddingError::OutOfVocabularyError));
        }

        Ok(self.embedding_matrix.column(token).to_owned())
    }

    /// Updating the embedding for a specific token (index) in the embedding matrix
    ///
    /// # Parameters
    /// - `index`: Token index for which the embedding needs to be updated.
    /// - `new_embedding`: New embedding (one-dimensional array) to replace the existing one.
    ///
    /// # Returns
    /// - `Ok(())`: Indicates that the embedding was successfully updated.
    /// - `Err(anyhow::Error)`: Returns an error if the token index is out of bounds of the dictionary, or the dimension of the new embedding does not match the expected.
    ///
    /// # Errors
    /// - `OutOfVocabularyError`: Raised if the token index is out of bounds of the dictionary.
    /// - `DimensionMismatchError`: Raised if the dimension of the new embedding does not match the expected embedding dimension.
    pub fn update_embedding(&mut self, index: usize, new_embedding: &Array1<f32>) -> Result<(), Error> {
        if index >= self.vocab_size {
            return Err(Error::from(EmbeddingError::OutOfVocabularyError));
        }

        if new_embedding.len() != self.embedding_dim {
            return Err(Error::from(EmbeddingError::DimensionMismatchError));
        }

        self.embedding_matrix.column_mut(index).assign(new_embedding);

        Ok(())
    }

    /// Getter for embedding matrix
    ///
    /// # Returns
    /// A reference to the `Array2<f32>` matrix containing the embeddings.
    pub fn embedding_matrix(&self) -> &Array2<f32> {
        &self.embedding_matrix
    }

    fn generate_random_values<D>(vocab_size: &usize, embedding_dim: &usize, distribution: D) -> Vec<f32>
    where
        D: Distribution<f32> + Send + Sync,
    {
        let total_elements = vocab_size * embedding_dim;

        (0..total_elements)
            .into_par_iter()
            .chunks(Self::CHUNK_SIZE)
            .flat_map(|chunk| {
                let mut local_rng = SmallRng::seed_from_u64(rand::rng().next_u64());

                let mut block_values = Vec::with_capacity(chunk.len());

                for _ in chunk {
                    block_values.push(distribution.sample(&mut local_rng));
                }

                block_values
            })
            .collect::<Vec<_>>()
    }

    fn create_embedding_matrix(
        embedding_dim: usize,
        vocab_size: usize,
        random_values: Vec<f32>,
    ) -> Result<Array2<f32>, EmbeddingError> {
        Array2::from_shape_vec((embedding_dim, vocab_size), random_values)
            .map_err(|err| EmbeddingError::MatrixCreationFailed(err.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_new_uniform() {
        let vocab_size = 10;
        let embedding_dim = 5;

        let embeddings_uniform =
            Embeddings::new_uniform(vocab_size, embedding_dim).expect("Failed to create new embedding matrix");

        assert_eq!(embeddings_uniform.embedding_matrix.shape(), &[embedding_dim, vocab_size]);
        assert_eq!(embeddings_uniform.vocab_size, vocab_size);
        assert_eq!(embeddings_uniform.embedding_dim, embedding_dim);

        for &value in embeddings_uniform.embedding_matrix.iter() {
            assert!(value >= -1.0 && value <= 1.0);
        }
    }

    #[test]
    fn test_new_gaussian() {
        let mean = 0.0;
        let std_dev = 0.01;
        let vocab_size = 10;
        let embedding_dim = 5;

        let embeddings_gaussian = Embeddings::new_gaussian(vocab_size, embedding_dim, mean, std_dev)
            .expect("Failed to create new embedding matrix");

        assert_eq!(embeddings_gaussian.embedding_matrix.shape(), &[embedding_dim, vocab_size]);
        assert_eq!(embeddings_gaussian.vocab_size, vocab_size);
        assert_eq!(embeddings_gaussian.embedding_dim, embedding_dim);

        let sum = embeddings_gaussian.embedding_matrix.iter().sum::<f32>();
        let avg = sum / (vocab_size * embedding_dim) as f32;

        assert!(
            (avg - mean).abs() < 0.1,
            "Average value of embeddings ({}) deviates too much from expected mean ({})",
            avg,
            mean
        );
    }

    #[test]
    fn test_new_xavier() {
        let vocab_size = 10;
        let embedding_dim = 5;

        let embeddings_xavier =
            Embeddings::new_xavier(vocab_size, embedding_dim).expect("Failed to create new embedding matrix");

        assert_eq!(embeddings_xavier.embedding_matrix.shape(), &[embedding_dim, vocab_size]);
        assert_eq!(embeddings_xavier.vocab_size, vocab_size);
        assert_eq!(embeddings_xavier.embedding_dim, embedding_dim);

        for &value in embeddings_xavier.embedding_matrix.iter() {
            assert!(value >= -1.0 && value <= 1.0);
        }
    }

    #[test]
    fn test_get_embeddings() {
        let vocab_size = 10;
        let embedding_dim = 5;
        let tokens = vec![0, 3, 7];

        let embeddings =
            Embeddings::new_uniform(vocab_size, embedding_dim).expect("Failed to create new embedding matrix");
        let result = embeddings
            .get_embeddings(&tokens)
            .expect("Failed to get embeddings for a specific tokens");

        assert_eq!(result.shape(), &[embedding_dim, tokens.len()]);

        for (i, &token) in tokens.iter().enumerate() {
            let expected_embedding = embeddings.embedding_matrix.column(token);
            let actual_embedding = result.column(i);

            for (expected, actual) in expected_embedding.iter().zip(actual_embedding.iter()) {
                assert_eq!(*expected, *actual);
            }
        }
    }

    #[test]
    fn test_get_embeddings_out_of_bounds() {
        let vocab_size = 10;
        let embedding_dim = 5;
        let tokens = vec![1, 5, 15];

        let embeddings =
            Embeddings::new_uniform(vocab_size, embedding_dim).expect("Failed to create new embedding matrix");
        let result = embeddings.get_embeddings(&tokens);

        assert!(result.is_err(), "Expected error for out of bounds, but got Ok");
    }

    #[test]
    fn test_get_embeddings_empty() {
        let vocab_size = 10;
        let embedding_dim = 5;
        let tokens = vec![];

        let embeddings =
            Embeddings::new_uniform(vocab_size, embedding_dim).expect("Failed to create new embedding matrix");
        let result = embeddings
            .get_embeddings(&tokens)
            .expect("Failed to get embeddings for a specific tokens");

        assert_eq!(result.shape(), &[embedding_dim, 0]);
    }

    #[test]
    fn test_get_embedding() {
        let vocab_size = 10;
        let embedding_dim = 5;
        let token = 0;

        let embeddings =
            Embeddings::new_uniform(vocab_size, embedding_dim).expect("Failed to create new embedding matrix");
        let actual_embedding = embeddings
            .get_embedding(token)
            .expect("Failed to get embedding for a specific token");

        assert_eq!(actual_embedding.shape(), &[embedding_dim]);

        let expected_embedding = embeddings.embedding_matrix.column(token);

        assert_eq!(expected_embedding, actual_embedding);
    }

    #[test]
    fn test_get_embedding_out_of_bounds() {
        let vocab_size = 10;
        let embedding_dim = 5;
        let token = 15;

        let embeddings =
            Embeddings::new_uniform(vocab_size, embedding_dim).expect("Failed to create new embedding matrix");
        let result = embeddings.get_embedding(token);

        assert!(result.is_err(), "Expected error for out of bounds, but got Ok");
    }

    #[test]
    fn test_update_embedding() {
        let vocab_size = 10;
        let embedding_dim = 5;
        let token = 0;
        let new_embedding = arr1(&[0.0101, 0.2189, -0.1, 0.54, -0.0001]);

        let mut embeddings =
            Embeddings::new_uniform(vocab_size, embedding_dim).expect("Failed to create new embedding matrix");

        embeddings
            .update_embedding(token, &new_embedding)
            .expect("Failed to update embedding for a specific token");

        let updated_embedding = embeddings
            .get_embedding(token)
            .expect("Failed to get embedding for a specific token");

        assert_eq!(updated_embedding, new_embedding);
    }

    #[test]
    fn test_update_embedding_out_of_bounds() {
        let vocab_size = 10;
        let embedding_dim = 5;
        let token = 15;
        let new_embedding = arr1(&[0.0101, 0.2189, -0.1, 0.54, -0.0001]);

        let mut embeddings =
            Embeddings::new_uniform(vocab_size, embedding_dim).expect("Failed to create new embedding matrix");

        let result = embeddings.update_embedding(token, &new_embedding);

        assert!(result.is_err(), "Expected error for out of bounds, but got Ok");
    }

    #[test]
    fn test_update_embedding_dimension_mismatch() {
        let vocab_size = 10;
        let embedding_dim = 5;
        let token = 5;
        let new_embedding = arr1(&[0.0101, 0.2189, 0.54]);

        let mut embeddings =
            Embeddings::new_uniform(vocab_size, embedding_dim).expect("Failed to create new embedding matrix");
        let result = embeddings.update_embedding(token, &new_embedding);

        assert!(result.is_err(), "Expected error for embedding dimension mismatch, but got Ok");
    }

    #[test]
    fn test_embedding_matrix() {
        let vocab_size = 10;
        let embedding_dim = 5;

        let embeddings =
            Embeddings::new_uniform(vocab_size, embedding_dim).expect("Failed to create new embedding matrix");

        let matrix = embeddings.embedding_matrix();

        assert_eq!(matrix.shape(), &[embedding_dim, vocab_size]);
        assert_eq!(*matrix, embeddings.embedding_matrix);
    }
}
