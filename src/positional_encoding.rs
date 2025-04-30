use crate::error::PositionalEncodingError;
use anyhow::{Error, Result};
use ndarray::{Array1, Array2, s};

pub struct PositionalEncoding {
    encoding_matrix: Array2<f32>,
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
        let mut encoding_matrix = Array2::zeros((embedding_dim, max_seq_len));

        for pos in 0..max_seq_len {
            for i in 0..embedding_dim {
                let angle = pos as f32 / 10000.0_f32.powf((2 * (i / 2)) as f32 / embedding_dim as f32);

                encoding_matrix[[i, pos]] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }

        Self {
            encoding_matrix,
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
        let mut encoding_matrix = Array2::zeros((embedding_dim, max_seq_len));

        for pos in 0..max_seq_len {
            for i in 0..embedding_dim {
                let relative_pos = pos as f32 - (max_seq_len / 2) as f32;
                let angle = relative_pos / 10000.0_f32.powf((2 * (i / 2)) as f32 / embedding_dim as f32);

                encoding_matrix[[i, pos]] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }

        Self {
            encoding_matrix,
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
        let mut encoding_matrix = Array2::zeros((embedding_dim, max_seq_len));

        for pos in 0..max_seq_len {
            for i in (0..embedding_dim).step_by(2) {
                let theta = Self::get_theta(pos, i, embedding_dim);

                encoding_matrix[[i, pos]] = theta.cos();

                if i + 1 < embedding_dim {
                    encoding_matrix[[i + 1, pos]] = theta.sin();
                }
            }
        }

        Self {
            encoding_matrix,
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

        let seq_len = input.shape()[1];
        let encoding = self.get_slice_encoding(seq_len)?;
        let mut output_matrix = input.clone();

        for pos in 0..seq_len {
            for i in (0..self.embedding_dim.saturating_sub(1)).step_by(2) {
                let cos = encoding[[i, pos]];
                let sin = encoding[[i + 1, pos]];
                let xi = input[[i, pos]];
                let xi1 = input[[i + 1, pos]];

                output_matrix[[i, pos]] = xi * cos - xi1 * sin;
                output_matrix[[i + 1, pos]] = xi * sin + xi1 * cos;
            }

            if self.embedding_dim % 2 == 1 {
                let i = self.embedding_dim - 1;
                output_matrix[[i, pos]] = input[[i, pos]];
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

        *embeddings += &self.encoding_matrix.slice(s![.., ..seq_len]);

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

        Ok(self.encoding_matrix.slice(s![.., ..seq_len]).to_owned())
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

        Ok(self.encoding_matrix.column(position).to_owned())
    }

    /// Getter for positional encoding matrix
    ///
    /// # Returns
    /// A reference to the `Array2<f32>` matrix containing the positional encoding.
    pub fn encoding_matrix(&self) -> &Array2<f32> {
        &self.encoding_matrix
    }

    fn get_theta(pos: usize, i: usize, dim: usize) -> f32 {
        let exponent: f32 = (2 * (i / 2)) as f32 / dim as f32;

        pos as f32 / 10000.0_f32.powf(exponent)
    }

    fn get_slice_encoding(&self, seq_len: usize) -> Result<Array2<f32>, Error> {
        if seq_len > self.max_seq_len {
            return Err(Error::from(PositionalEncodingError::SequenceLengthExceeded));
        }

        Ok(self.encoding_matrix.slice(s![.., ..seq_len]).to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array2;

    #[test]
    fn test_new_sinusoidal() {
        let max_seq_len = 10;
        let embedding_dim = 6;

        let pe_sinusoidal = PositionalEncoding::new_sinusoidal(max_seq_len, embedding_dim);

        assert_eq!(pe_sinusoidal.encoding_matrix.shape(), &[embedding_dim, max_seq_len]);
        assert_eq!(pe_sinusoidal.max_seq_len, max_seq_len);
        assert_eq!(pe_sinusoidal.embedding_dim, embedding_dim);

        assert_abs_diff_eq!(pe_sinusoidal.encoding_matrix[[0, 0]], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(pe_sinusoidal.encoding_matrix[[0, 1]], f32::sin(1.0), epsilon = 1e-6);
        assert_abs_diff_eq!(pe_sinusoidal.encoding_matrix[[1, 0]], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_new_relative() {
        let max_seq_len = 10;
        let embedding_dim = 6;
        let center = 5;

        let pe_relative = PositionalEncoding::new_relative(max_seq_len, embedding_dim);

        assert_eq!(pe_relative.encoding_matrix.shape(), &[embedding_dim, max_seq_len]);
        assert_eq!(pe_relative.max_seq_len, max_seq_len);
        assert_eq!(pe_relative.embedding_dim, embedding_dim);

        assert_abs_diff_eq!(pe_relative.encoding_matrix[[0, center]], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(pe_relative.encoding_matrix[[1, center]], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_new_rope() {
        let max_seq_len = 10;
        let embedding_dim = 6;

        let pe_rope = PositionalEncoding::new_rope(max_seq_len, embedding_dim);

        assert_eq!(pe_rope.encoding_matrix.shape(), &[embedding_dim, max_seq_len]);
        assert_eq!(pe_rope.max_seq_len, max_seq_len);
        assert_eq!(pe_rope.embedding_dim, embedding_dim);

        assert!(pe_rope.encoding_matrix[[0, 0]].abs() <= 1.0);
        assert!(pe_rope.encoding_matrix[[1, 0]].abs() <= 1.0);
        assert_ne!(pe_rope.encoding_matrix[[0, 0]], pe_rope.encoding_matrix[[1, 0]]);
    }

    #[test]
    fn test_apply_rope() {
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
        let angle = pos as f32 / 10000.0_f32.powf(i as f32 / embedding_dim as f32);
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

        assert!(result.is_ok(), "Failed to add embeddings");

        for i in 0..embedding_dim {
            for j in 0..seq_len {
                assert_abs_diff_eq!(
                    embeddings[[i, j]],
                    original_embeddings[[i, j]] + positional_encoding.encoding_matrix[[i, j]],
                    epsilon = 1e-6
                );
            }
        }
    }

    #[test]
    fn test_add_to_embeddings_sequence_too_long() {
        let wrong_seq_len = 10;
        let max_seq_len = 5;
        let embedding_dim = 4;

        let positional_encoding = PositionalEncoding::new_sinusoidal(max_seq_len, embedding_dim);
        let mut embeddings = Array2::zeros((embedding_dim, wrong_seq_len));

        let result = positional_encoding.add_to_embeddings(&mut embeddings);

        assert!(result.is_err(), "Expected error for sequence length exceeded, but got Ok");
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

        assert!(result.is_err(), "Expected error for embedding dimension mismatch, but got Ok");
    }

    #[test]
    fn test_encoding_matrix() {
        let vocab_size = 10;
        let embedding_dim = 5;

        let positional_encoding = PositionalEncoding::new_sinusoidal(vocab_size, embedding_dim);

        let matrix = positional_encoding.encoding_matrix();

        assert_eq!(matrix.shape(), &[embedding_dim, vocab_size]);
        assert_eq!(*matrix, positional_encoding.encoding_matrix);
    }

    #[test]
    fn test_get_positional_encoding_slice() {
        let seq_len = 5;
        let max_seq_len = 10;
        let embedding_dim = 4;

        let positional_encoding = PositionalEncoding::new_sinusoidal(max_seq_len, embedding_dim);

        let result = positional_encoding.get_positional_encoding_slice(seq_len);

        assert!(result.is_ok(), "Failed to retrieve PE slice for sequence length {}", seq_len);

        let pe_subset = result.expect("Failed to get subset");

        assert_eq!(pe_subset.shape(), &[embedding_dim, seq_len]);

        for i in 0..embedding_dim {
            for j in 0..seq_len {
                assert_abs_diff_eq!(pe_subset[[i, j]], positional_encoding.encoding_matrix[[i, j]], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_get_positional_encoding_slice_too_long() {
        let wrong_seq_len = 10;
        let max_seq_len = 5;
        let embedding_dim = 4;

        let positional_encoding = PositionalEncoding::new_sinusoidal(max_seq_len, embedding_dim);

        let result = positional_encoding.get_positional_encoding_slice(wrong_seq_len);

        assert!(result.is_err(), "Expected error for sequence length exceeded, but got Ok");
    }

    #[test]
    fn test_get_positional_encoding() {
        let position = 3;
        let max_seq_len = 5;
        let embedding_dim = 4;

        let positional_encoding = PositionalEncoding::new_sinusoidal(max_seq_len, embedding_dim);

        let result = positional_encoding.get_positional_encoding(position);

        assert!(result.is_ok(), "Failed to get positional encoding");

        let pe_subset = result.expect("Failed to get subset");

        assert_eq!(pe_subset.shape(), &[embedding_dim]);
        assert_eq!(pe_subset[0], positional_encoding.encoding_matrix[[0, position]]);
    }

    #[test]
    fn test_get_positional_encoding_out_of_bounds() {
        let wrong_position = 10;
        let max_seq_len = 5;
        let embedding_dim = 4;

        let positional_encoding = PositionalEncoding::new_sinusoidal(max_seq_len, embedding_dim);

        let result = positional_encoding.get_positional_encoding(wrong_position);

        assert!(result.is_err(), "Expected error for out of bounds, but got OK");
    }
}
