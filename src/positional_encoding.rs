use crate::error::PositionalEncodingError;
use anyhow::{Error, Result};
use ndarray::{Array1, Array2, s};

pub struct PositionalEncoding {
    encoding: Array2<f32>,
    max_seq_len: usize,
    embedding_dim: usize,
}

impl PositionalEncoding {
    /// Initialization of the new matrix of positional encodings SPE (Sinusoidal Positional Encoding)
    ///
    /// # Parameters
    /// - `max_seq_len`: Maximum sequence length (number of tokens).
    /// - `embedding_dim`: Dimension of embeddings (length of vector for each token).
    ///
    /// # Returns
    /// An instance of a structure with a matrix of sinusoidal embeddings.
    pub fn new_sinusoidal(max_seq_len: usize, embedding_dim: usize) -> Self {
        let mut encoding = Array2::zeros((embedding_dim, max_seq_len));

        for pos in 0..max_seq_len {
            for i in 0..embedding_dim {
                let angle = pos as f32 / 10000.0f32.powf((2 * (i / 2)) as f32 / embedding_dim as f32);

                encoding[[i, pos]] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }

        Self {
            encoding,
            max_seq_len,
            embedding_dim,
        }
    }

    /// Initialization of the new matrix of positional encodings RPE (Relative Positional Encoding)
    ///
    /// # Parameters
    /// - `max_seq_len`: Maximum sequence length (number of tokens).
    /// - `embedding_dim`: Dimension of embeddings (length of vector for each token).
    ///
    /// # Returns
    /// An instance of a structure with a matrix of relative embeddings.
    pub fn new_relative(max_seq_len: usize, embedding_dim: usize) -> Self {
        let mut encoding = Array2::zeros((embedding_dim, max_seq_len));

        for pos in 0..max_seq_len {
            for i in 0..embedding_dim {
                let relative_pos = pos as f32 - (max_seq_len / 2) as f32;
                let angle = relative_pos / 10000.0f32.powf((2 * (i / 2)) as f32 / embedding_dim as f32);

                encoding[[i, pos]] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }

        Self {
            encoding,
            max_seq_len,
            embedding_dim,
        }
    }

    /// Initialization of the new matrix of positional encodings `RoPE` (Rotary Positional Embedding)
    ///
    /// # Parameters
    /// - `max_seq_len`: Maximum sequence length (number of tokens).
    /// - `embedding_dim`: Dimension of embeddings (length of vector for each token).
    ///
    /// # Returns
    /// A struct instance with an embedding matrix initialized using `RoPE`.
    pub fn new_rope(max_seq_len: usize, embedding_dim: usize) -> Self {
        let mut encoding = Array2::zeros((embedding_dim, max_seq_len));

        for pos in 0..max_seq_len {
            for i in 0..embedding_dim {
                let angle = pos as f32 / 10000.0f32.powf((2 * (i / 2)) as f32 / embedding_dim as f32);

                if i % 2 == 0 {
                    encoding[[i, pos]] = angle.sin();
                } else {
                    encoding[[i, pos]] = angle.cos();
                }
            }
        }

        Self {
            encoding,
            max_seq_len,
            embedding_dim,
        }
    }

    /// Applying `RoPE` to an input vector
    ///
    /// # Parameters
    /// - `input`: Input embedding matrix to apply `RoPE` to.
    ///
    /// # Returns
    /// - `Ok(Array2<f32>)`: The embedding matrix with `RoPE` applied.
    /// - `Err(anyhow::Error)`: Error if the input matrix dimensions do not match the expected dimensions.
    ///
    /// # Errors
    /// - `ShapeMismatch`: Occurs if the dimensions of the input matrix do not match the expected embedding dimension and maximum sequence length.
    pub fn apply_rope(&self, input: &Array2<f32>) -> Result<Array2<f32>, Error> {
        if input.shape() != [self.embedding_dim, self.max_seq_len] {
            return Err(Error::from(PositionalEncodingError::ShapeMismatch));
        }

        let mut output_matrix = Array2::zeros((self.embedding_dim, self.max_seq_len));

        for pos in 0..self.max_seq_len {
            for i in (0..self.embedding_dim).step_by(2) {
                let angle = pos as f32 / 10000.0f32.powf(i as f32 / self.embedding_dim as f32);

                output_matrix[[i, pos]] = input[[i, pos]] * angle.cos();

                if i + 1 < self.embedding_dim {
                    output_matrix[[i, pos]] -= input[[i + 1, pos]] * angle.sin();
                    output_matrix[[i + 1, pos]] = input[[i + 1, pos]] * angle.cos() + input[[i, pos]] * angle.sin();
                }
            }
        }

        Ok(output_matrix)
    }

    /// Applying positional encodings to the embedding matrix (for SPE, RPE)
    ///
    /// # Parameters
    /// - `embeddings`: Input matrix of embeddings to which positional embeddings are added.
    ///
    /// # Returns
    /// - `Ok(())`: If the operation was successful.
    /// - `Err(anyhow::Error)`: Error if the sequence length exceeds the maximum or if the dimensions of the embeddings do not match.
    ///
    /// # Errors
    /// - `SequenceLengthExceeded`: Occurs if the sequence length of the input embeddings exceeds the maximum sequence length supported by the positional encoding.
    /// - `EmbeddingDimensionMismatch`: Occurs if the dimensionality of the input embeddings does not match the expected embedding dimension of the positional encoding.
    pub fn add_to_embeddings(&self, embeddings: &mut Array2<f32>) -> Result<(), Error> {
        let seq_len = embeddings.shape()[1];

        if seq_len > self.max_seq_len {
            return Err(Error::from(PositionalEncodingError::SequenceLengthExceeded));
        }

        if embeddings.shape()[0] != self.embedding_dim {
            return Err(Error::from(PositionalEncodingError::EmbeddingDimensionMismatch));
        }

        let pe_slice = self.encoding.slice(s![.., ..seq_len]);
        *embeddings += &pe_slice;

        Ok(())
    }

    /// Return part of the positional encoding matrix for a sequence
    ///
    /// # Parameters
    /// - `seq_len`: Length of the sequence for which positional embeddings are required.
    ///
    /// # Returns
    /// - `Ok(Array2<f32>)`: Matrix of positional embeddings for the specified sequence length.
    /// - `Err(anyhow::Error)`: Error if the requested sequence length exceeds the maximum.
    ///
    /// # Errors
    /// - `SequenceLengthExceeded`: Occurs if the requested sequence length exceeds the maximum sequence length supported by the positional encoding.
    pub fn get_positional_encoding_slice(&self, seq_len: usize) -> Result<Array2<f32>, Error> {
        if seq_len > self.max_seq_len {
            return Err(Error::from(PositionalEncodingError::SequenceLengthExceeded));
        }

        let pe_slice = self.encoding.slice(s![.., ..seq_len]).to_owned();

        Ok(pe_slice)
    }

    /// Return the positional encoding (one-dimensional array) for a specific position in the sequence
    ///
    /// # Parameters
    /// - `position`: The position in the sequence.
    ///
    /// # Returns
    /// - `Ok(Array1<f32>)`: A vector (one-dimensional array) of positional embeddings for the specified position.
    /// - `Err(anyhow::Error)`: Error if the requested position is out of bounds.
    ///
    /// # Errors
    /// - `PositionOutOfBounds`: Occurs if the requested position exceeds the maximum sequence length supported by the positional encoding.
    pub fn get_positional_encoding(&self, position: usize) -> Result<Array1<f32>, Error> {
        if position >= self.max_seq_len {
            return Err(Error::from(PositionalEncodingError::PositionOutOfBounds));
        }

        Ok(self.encoding.column(position).to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array2;

    #[test]
    fn test_positional_encoding_new_sinusoidal() {
        let max_seq_len = 10;
        let embedding_dim = 6;

        let pe_sinusoidal = PositionalEncoding::new_sinusoidal(max_seq_len, embedding_dim);

        assert_eq!(pe_sinusoidal.encoding.shape(), &[embedding_dim, max_seq_len]);
        assert_eq!(pe_sinusoidal.max_seq_len, max_seq_len);
        assert_eq!(pe_sinusoidal.embedding_dim, embedding_dim);

        assert_abs_diff_eq!(pe_sinusoidal.encoding[[0, 0]], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(pe_sinusoidal.encoding[[0, 1]], f32::sin(1.0), epsilon = 1e-6);
        assert_abs_diff_eq!(pe_sinusoidal.encoding[[1, 0]], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_positional_encoding_new_relative() {
        let max_seq_len = 10;
        let embedding_dim = 6;
        let center = 5;

        let pe_relative = PositionalEncoding::new_relative(max_seq_len, embedding_dim);

        assert_eq!(pe_relative.encoding.shape(), &[embedding_dim, max_seq_len]);
        assert_eq!(pe_relative.max_seq_len, max_seq_len);
        assert_eq!(pe_relative.embedding_dim, embedding_dim);

        assert_abs_diff_eq!(pe_relative.encoding[[0, center]], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(pe_relative.encoding[[1, center]], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_positional_encoding_new_rope() {
        let max_seq_len = 10;
        let embedding_dim = 6;

        let pe_rope = PositionalEncoding::new_rope(max_seq_len, embedding_dim);

        assert_eq!(pe_rope.encoding.shape(), &[embedding_dim, max_seq_len]);
        assert_eq!(pe_rope.max_seq_len, max_seq_len);
        assert_eq!(pe_rope.embedding_dim, embedding_dim);

        assert!(pe_rope.encoding[[0, 0]].abs() <= 1.0);
        assert!(pe_rope.encoding[[1, 0]].abs() <= 1.0);
        assert_ne!(pe_rope.encoding[[0, 0]], pe_rope.encoding[[1, 0]]);
    }

    #[test]
    fn test_positional_encoding_apply_rope() {
        let max_seq_len = 10;
        let embedding_dim = 6;

        let pe_rope = PositionalEncoding::new_rope(max_seq_len, embedding_dim);

        let input = Array2::ones((embedding_dim, max_seq_len));
        let output = pe_rope.apply_rope(&input).unwrap();

        assert_eq!(output.shape(), &[embedding_dim, max_seq_len]);

        assert_ne!(output, input);

        for pos in 0..max_seq_len {
            for i in (0..embedding_dim).step_by(2) {
                if i + 1 < embedding_dim {
                    let input_norm = (input[[i, pos]].powi(2) + input[[i + 1, pos]].powi(2)).sqrt();
                    let output_norm = (output[[i, pos]].powi(2) + output[[i + 1, pos]].powi(2)).sqrt();

                    assert_abs_diff_eq!(input_norm, output_norm, epsilon = 1e-6);
                }
            }
        }

        let pos0_input = input.column(0).to_owned();
        let pos0_output = output.column(0).to_owned();
        assert_eq!(pos0_input, pos0_output);

        let pos = 1;
        let i = 0;
        let angle = pos as f32 / 10000.0f32.powf(i as f32 / embedding_dim as f32);
        let expected_i = angle.cos() - angle.sin();
        let expected_i1 = angle.cos() + angle.sin();

        assert_abs_diff_eq!(output[[i, pos]], expected_i, epsilon = 1e-6);
        assert_abs_diff_eq!(output[[i + 1, pos]], expected_i1, epsilon = 1e-6);
    }

    #[test]
    fn test_add_to_embeddings() {
        let seq_len = 5;
        let max_seq_len = 10;
        let embedding_dim = 4;

        let positional_encoding = PositionalEncoding::new_sinusoidal(max_seq_len, embedding_dim);

        let mut embeddings = Array2::zeros((embedding_dim, seq_len));

        for i in 0..embedding_dim {
            for j in 0..seq_len {
                embeddings[[i, j]] = (i * seq_len + j) as f32;
            }
        }

        let original_embeddings = embeddings.clone();

        let result = positional_encoding.add_to_embeddings(&mut embeddings);

        assert!(result.is_ok());

        for i in 0..embedding_dim {
            for j in 0..seq_len {
                assert_abs_diff_eq!(
                    embeddings[[i, j]],
                    original_embeddings[[i, j]] + positional_encoding.encoding[[i, j]],
                    epsilon = 1e-6
                );
            }
        }
    }

    #[test]
    fn test_add_to_embeddings_sequence_too_long() {
        let seq_len = 10;
        let max_seq_len = 5;
        let embedding_dim = 4;

        let positional_encoding = PositionalEncoding::new_sinusoidal(max_seq_len, embedding_dim);
        let mut embeddings = Array2::zeros((embedding_dim, seq_len));

        let result = positional_encoding.add_to_embeddings(&mut embeddings);

        assert!(result.is_err());
    }

    #[test]
    fn test_add_to_embeddings_dimension_mismatch() {
        let seq_len = 5;
        let wrong_dim = 6;
        let max_seq_len = 10;
        let embedding_dim = 4;

        let positional_encoding = PositionalEncoding::new_sinusoidal(max_seq_len, embedding_dim);
        let mut embeddings = Array2::zeros((wrong_dim, seq_len));

        let result = positional_encoding.add_to_embeddings(&mut embeddings);

        assert!(result.is_err());
    }

    #[test]
    fn test_get_positional_encoding_slice() {
        let seq_len = 5;
        let max_seq_len = 10;
        let embedding_dim = 4;

        let positional_encoding = PositionalEncoding::new_sinusoidal(max_seq_len, embedding_dim);

        let result = positional_encoding.get_positional_encoding_slice(seq_len);

        assert!(result.is_ok());
        let pe_subset = result.unwrap();

        assert_eq!(pe_subset.shape(), &[embedding_dim, seq_len]);

        for i in 0..embedding_dim {
            for j in 0..seq_len {
                assert_abs_diff_eq!(pe_subset[[i, j]], positional_encoding.encoding[[i, j]], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_get_positional_encoding_slice_too_long() {
        let seq_len = 10;
        let max_seq_len = 5;
        let embedding_dim = 4;

        let positional_encoding = PositionalEncoding::new_sinusoidal(max_seq_len, embedding_dim);

        let result = positional_encoding.get_positional_encoding_slice(seq_len);

        assert!(result.is_err());
    }

    #[test]
    fn test_get_positional_encoding() {
        let position = 3;
        let max_seq_len = 5;
        let embedding_dim = 4;

        let positional_encoding = PositionalEncoding::new_sinusoidal(max_seq_len, embedding_dim);

        let result = positional_encoding.get_positional_encoding(position);

        assert!(result.is_ok());
        let pe_subset = result.unwrap();

        assert_eq!(pe_subset.shape(), &[embedding_dim]);
        assert_eq!(pe_subset[0], positional_encoding.encoding[[0, position]]);
    }

    #[test]
    fn test_get_positional_encoding_out_of_bounds() {
        let position = 10;
        let max_seq_len = 5;
        let embedding_dim = 4;

        let positional_encoding = PositionalEncoding::new_sinusoidal(max_seq_len, embedding_dim);

        let result = positional_encoding.get_positional_encoding(position);

        assert!(result.is_err());
    }
}
