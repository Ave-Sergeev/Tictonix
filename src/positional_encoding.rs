use ndarray::{Array2, s};

pub struct PositionalEncoding {
    encoding: Array2<f32>,
    max_seq_len: usize,
    embedding_dim: usize,
}

impl PositionalEncoding {
    /// Создание новой матрицы позиционных кодировок
    pub fn new(max_seq_len: usize, embedding_dim: usize) -> Self {
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

    /// Применение позиционных кодировок к матрице эмбеддингов
    pub fn add_to_embeddings(&self, embeddings: &mut Array2<f32>) -> Result<(), &'static str> {
        let seq_len = embeddings.shape()[1];

        if seq_len > self.max_seq_len {
            return Err("Sequence length exceeds maximum sequence length for positional encoding");
        }

        if embeddings.shape()[0] != self.embedding_dim {
            return Err("Embedding dimension mismatch");
        }

        let pe_slice = self.encoding.slice(s![.., ..seq_len]);
        *embeddings += &pe_slice;

        Ok(())
    }

    /// Возврат части матрицы позиционных кодировок для последовательности
    pub fn for_sequence(&self, seq_len: usize) -> Result<Array2<f32>, &'static str> {
        if seq_len > self.max_seq_len {
            return Err("Requested sequence length exceeds maximum sequence length");
        }

        Ok(self.encoding.slice(s![.., ..seq_len]).to_owned())
    }
}
