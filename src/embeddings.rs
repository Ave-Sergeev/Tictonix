use crate::error::EmbeddingError;
use anyhow::{Error, Result};
use ndarray::{Array1, Array2};
use rand::distr::{Distribution, Uniform};
use rand_distr::Normal;

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
    /// - `Ok(Self)`: An `Embeddings` instance with a matrix filled with values from a uniform distribution in the range `[-1.0, 1.0]`.
    /// - `Err(anyhow::Error)`: An error is returned if the uniform distribution cannot be created, or input parameters are invalid.
    ///
    /// # Errors
    /// - `InvalidInput`: Occurs if the input parameters are invalid.
    /// - `UniformCreationFailed`: Occurs if the uniform distribution cannot be created within the specified range `[-1.0, 1.0]`.
    pub fn new_uniform(vocab_size: usize, embedding_dim: usize) -> Result<Self, Error> {
        if vocab_size == 0 || embedding_dim == 0 {
            return Err(Error::from(EmbeddingError::InvalidInput(
                "Parameters vocab_size and embedding_dim must be greater than zero".to_string(),
            )));
        }

        let mut rng = rand::rng();
        let uniform =
            Uniform::new_inclusive(-1.0, 1.0).map_err(|err| EmbeddingError::UniformCreationFailed(err.to_string()))?;
        let matrix = Array2::from_shape_fn((embedding_dim, vocab_size), |_| uniform.sample(&mut rng));

        Ok(Self {
            matrix,
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

        let mut rng = rand::rng();
        let normal = Normal::new(mean, std_dev).map_err(|err| EmbeddingError::NormalCreationFailed(err.to_string()))?;
        let matrix = Array2::from_shape_fn((embedding_dim, vocab_size), |_| normal.sample(&mut rng));

        Ok(Self {
            matrix,
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
    pub fn new_xavier(vocab_size: usize, embedding_dim: usize) -> Result<Self, Error> {
        if vocab_size == 0 || embedding_dim == 0 {
            return Err(Error::from(EmbeddingError::InvalidInput(
                "Parameters vocab_size and embedding_dim must be greater than zero".to_string(),
            )));
        }

        let mut rng = rand::rng();
        let std_dev = (6.0 / (vocab_size as f32 + embedding_dim as f32)).sqrt();

        let uniform = Uniform::new_inclusive(-std_dev, std_dev)
            .map_err(|e| EmbeddingError::UniformCreationFailed(e.to_string()))?;
        let matrix = Array2::from_shape_fn((embedding_dim, vocab_size), |_| uniform.sample(&mut rng));

        Ok(Self {
            matrix,
            vocab_size,
            embedding_dim,
        })
    }

    /// Getter for embedding matrix
    ///
    /// # Returns
    /// A reference to the `Array2<f32>` matrix containing the embeddings.
    pub fn getter_matrix(&self) -> &Array2<f32> {
        &self.matrix
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
        let mut embeddings = Array2::zeros((self.embedding_dim, tokens.len()));

        for (i, &token) in tokens.iter().enumerate() {
            if token >= self.vocab_size {
                return Err(Error::from(EmbeddingError::OutOfVocabularyError));
            }

            embeddings.column_mut(i).assign(&self.matrix.column(token));
        }

        Ok(embeddings)
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

        Ok(self.matrix.column(token).to_owned())
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

        self.matrix.column_mut(index).assign(new_embedding);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_embeddings_new_uniform() {
        let vocab_size = 10;
        let embedding_dim = 5;

        let embeddings_uniform = Embeddings::new_uniform(vocab_size, embedding_dim).unwrap();

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

        let embeddings_gaussian = Embeddings::new_gaussian(vocab_size, embedding_dim, mean, std_dev).unwrap();

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

        let embeddings_xavier = Embeddings::new_xavier(vocab_size, embedding_dim).unwrap();

        assert_eq!(embeddings_xavier.matrix.shape(), &[embedding_dim, vocab_size]);
        assert_eq!(embeddings_xavier.vocab_size, vocab_size);
        assert_eq!(embeddings_xavier.embedding_dim, embedding_dim);

        for &value in embeddings_xavier.matrix.iter() {
            assert!(value >= -1.0 && value <= 1.0);
        }
    }

    #[test]
    fn test_get_embeddings() {
        let vocab_size = 10;
        let embedding_dim = 5;

        let embeddings = Embeddings::new_uniform(vocab_size, embedding_dim).unwrap();

        let tokens = vec![0, 3, 7];
        let result = embeddings.get_embeddings(&tokens).unwrap();

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
    fn test_get_embeddings_out_of_bounds() {
        let vocab_size = 10;
        let embedding_dim = 5;

        let embeddings = Embeddings::new_uniform(vocab_size, embedding_dim).unwrap();

        let tokens = vec![1, 5, 15];
        let result = embeddings.get_embeddings(&tokens);

        assert!(result.is_err());
    }

    #[test]
    fn test_get_embeddings_empty() {
        let vocab_size = 10;
        let embedding_dim = 5;

        let embeddings = Embeddings::new_uniform(vocab_size, embedding_dim).unwrap();

        let tokens = vec![];
        let result = embeddings.get_embeddings(&tokens).unwrap();

        assert_eq!(result.shape(), &[embedding_dim, 0]);
    }

    #[test]
    fn test_get_embedding() {
        let vocab_size = 10;
        let embedding_dim = 5;

        let embeddings = Embeddings::new_uniform(vocab_size, embedding_dim).unwrap();

        let token = 0;
        let result = embeddings.get_embedding(token).unwrap();

        assert_eq!(result.shape(), &[embedding_dim]);

        let expected_embedding = embeddings.matrix.column(token);
        let actual_embedding = result;

        assert_eq!(expected_embedding, actual_embedding);
    }

    #[test]
    fn test_get_embedding_out_of_bounds() {
        let vocab_size = 10;
        let embedding_dim = 5;

        let embeddings = Embeddings::new_uniform(vocab_size, embedding_dim).unwrap();

        let token = 15;
        let result = embeddings.get_embedding(token);

        assert!(result.is_err());
    }

    #[test]
    fn test_update_embedding() {
        let vocab_size = 10;
        let embedding_dim = 5;

        let mut embeddings = Embeddings::new_uniform(vocab_size, embedding_dim).unwrap();

        let token = 0;
        let new_embedding = arr1(&[0.0101, 0.2189, -0.1, 0.54, -0.0001]);

        embeddings.update_embedding(token, &new_embedding).unwrap();

        let updated_embedding = embeddings.get_embedding(token).unwrap();
        assert_eq!(updated_embedding, new_embedding);
    }

    #[test]
    fn test_update_embedding_out_of_bounds() {
        let vocab_size = 10;
        let embedding_dim = 5;

        let mut embeddings = Embeddings::new_uniform(vocab_size, embedding_dim).unwrap();

        let token = 15;
        let new_embedding = arr1(&[0.0101, 0.2189, -0.1, 0.54, -0.0001]);

        let result = embeddings.update_embedding(token, &new_embedding);

        assert!(result.is_err());
    }

    #[test]
    fn test_update_embedding_dimension_mismatch() {
        let vocab_size = 10;
        let embedding_dim = 5;

        let mut embeddings = Embeddings::new_uniform(vocab_size, embedding_dim).unwrap();

        let token = 5;
        let new_embedding = arr1(&[0.0101, 0.2189, 0.54]);

        let result = embeddings.update_embedding(token, &new_embedding);

        assert!(result.is_err());
    }

    #[test]
    fn test_getter_matrix() {
        let vocab_size = 10;
        let embedding_dim = 5;

        let embeddings = Embeddings::new_uniform(vocab_size, embedding_dim).unwrap();

        let matrix = embeddings.getter_matrix();

        assert_eq!(matrix.shape(), &[embedding_dim, vocab_size]);
        assert_eq!(*matrix, embeddings.matrix);
    }
}
