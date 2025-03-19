use ndarray::{Array2, s};

pub struct PositionalEncoding {
    encoding: Array2<f32>,
    max_seq_len: usize,
    embedding_dim: usize,
}

impl PositionalEncoding {
    /// Инициализация новой матрицы позиционных кодировок SPE (Sinusoidal Positional Encoding)
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

    /// Инициализация новой матрицы позиционных кодировок RPE (Relative Positional Encoding)
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

    /// Инициализация новой матрицы позиционных кодировок RoPE (Rotary Positional Embedding)
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

    /// Применение RoPE к входному вектору (использовать для RoPE)
    pub fn apply_rope(&self, input: &Array2<f32>) -> Array2<f32> {
        assert_eq!(input.shape(), &[self.embedding_dim, self.max_seq_len]);

        let mut output = Array2::zeros((self.embedding_dim, self.max_seq_len));

        for pos in 0..self.max_seq_len {
            for i in (0..self.embedding_dim).step_by(2) {
                let angle = pos as f32 / 10000.0f32.powf(i as f32 / self.embedding_dim as f32);

                output[[i, pos]] = input[[i, pos]] * angle.cos();

                if i + 1 < self.embedding_dim {
                    output[[i, pos]] -= input[[i + 1, pos]] * angle.sin();
                    output[[i + 1, pos]] = input[[i + 1, pos]] * angle.cos() + input[[i, pos]] * angle.sin();
                }
            }
        }

        output
    }

    /// Применение позиционных кодировок к матрице эмбеддингов (использовать для SPE, RPE)
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array2;

    #[test]
    fn test_positional_encoding_new_sinusoidal() {
        let max_seq_len = 10;
        let embedding_dim = 6;

        let positional_encoding = PositionalEncoding::new_sinusoidal(max_seq_len, embedding_dim);

        // Проверяем размеры матрицы
        assert_eq!(positional_encoding.encoding.shape(), &[embedding_dim, max_seq_len]);
        assert_eq!(positional_encoding.max_seq_len, max_seq_len);
        assert_eq!(positional_encoding.embedding_dim, embedding_dim);

        // Проверяем правильность формулы для нескольких значений
        // Для pos=0, i=0: sin(0 / 10000^(0/6)) = sin(0) = 0
        assert_abs_diff_eq!(positional_encoding.encoding[[0, 0]], 0.0, epsilon = 1e-6);

        // Для pos=1, i=0: sin(1 / 10000^(0/6)) = sin(1/1) = sin(1)
        assert_abs_diff_eq!(positional_encoding.encoding[[0, 1]], f32::sin(1.0), epsilon = 1e-6);

        // Для pos=0, i=1: cos(0 / 10000^(0/6)) = cos(0) = 1
        assert_abs_diff_eq!(positional_encoding.encoding[[1, 0]], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_positional_encoding_new_relative() {
        let max_seq_len = 10;
        let embedding_dim = 6;
        let center = 5;

        let positional_encoding = PositionalEncoding::new_relative(max_seq_len, embedding_dim);

        // Проверяем размеры матрицы
        assert_eq!(positional_encoding.encoding.shape(), &[embedding_dim, max_seq_len]);
        assert_eq!(positional_encoding.max_seq_len, max_seq_len);
        assert_eq!(positional_encoding.embedding_dim, embedding_dim);

        // Проверка, что центральная позиция имеет нулевое кодирование
        assert_abs_diff_eq!(positional_encoding.encoding[[0, center]], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(positional_encoding.encoding[[1, center]], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_positional_encoding_new_rope() {
        let max_seq_len = 10;
        let embedding_dim = 6;

        let positional_encoding = PositionalEncoding::new_rope(max_seq_len, embedding_dim);

        // Проверяем размеры матрицы
        assert_eq!(positional_encoding.encoding.shape(), &[embedding_dim, max_seq_len]);
        assert_eq!(positional_encoding.max_seq_len, max_seq_len);
        assert_eq!(positional_encoding.embedding_dim, embedding_dim);

        // Проверка чередования синуса и косинуса
        assert!(positional_encoding.encoding[[0, 0]].abs() <= 1.0);
        assert!(positional_encoding.encoding[[1, 0]].abs() <= 1.0);
        assert_ne!(positional_encoding.encoding[[0, 0]], positional_encoding.encoding[[1, 0]]);
    }

    #[test]
    fn test_positional_encoding_apply_rope() {
        let max_seq_len = 10;
        let embedding_dim = 6;

        let positional_encoding = PositionalEncoding::new_rope(max_seq_len, embedding_dim);
        let input = Array2::ones((embedding_dim, max_seq_len));
        let output = positional_encoding.apply_rope(&input);

        // Проверка формы выходного массива
        assert_eq!(output.shape(), &[embedding_dim, max_seq_len]);

        // Проверка, что выход отличается от входа (кроме позиции 0)
        assert_ne!(output, input);

        // Проверка сохранения нормы для каждой пары элементов
        for pos in 0..max_seq_len {
            for i in (0..embedding_dim).step_by(2) {
                if i + 1 < embedding_dim {
                    let input_norm = (input[[i, pos]].powi(2) + input[[i+1, pos]].powi(2)).sqrt();
                    let output_norm = (output[[i, pos]].powi(2) + output[[i+1, pos]].powi(2)).sqrt();

                    assert_abs_diff_eq!(input_norm, output_norm, epsilon = 1e-6);
                }
            }
        }

        // Дополнительная проверка для позиции 0 (должна сохранять значения)
        let pos0_input = input.column(0).to_owned();
        let pos0_output = output.column(0).to_owned();
        assert_eq!(pos0_input, pos0_output);

        // Проверка специфичных значений для позиции 1
        let pos = 1;
        let i = 0;
        let angle = pos as f32 / 10000.0f32.powf(i as f32 / embedding_dim as f32);
        let expected_i = angle.cos() - angle.sin();
        let expected_i1 = angle.cos() + angle.sin();

        assert_abs_diff_eq!(output[[i, pos]], expected_i, epsilon = 1e-6);
        assert_abs_diff_eq!(output[[i+1, pos]], expected_i1, epsilon = 1e-6);
    }

    #[test]
    fn test_add_to_embeddings() {
        let max_seq_len = 10;
        let embedding_dim = 4;
        let seq_len = 5;

        let positional_encoding = PositionalEncoding::new_sinusoidal(max_seq_len, embedding_dim);

        let mut embeddings = Array2::zeros((embedding_dim, seq_len));

        // Заполняем эмбеддинги некоторыми значениями
        for i in 0..embedding_dim {
            for j in 0..seq_len {
                embeddings[[i, j]] = (i * seq_len + j) as f32;
            }
        }

        let original_embeddings = embeddings.clone();

        let result = positional_encoding.add_to_embeddings(&mut embeddings);

        assert!(result.is_ok());

        // Проверяем, что эмбеддинги изменились правильно
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
        let max_seq_len = 5;
        let embedding_dim = 4;
        let seq_len = 10;

        let positional_encoding = PositionalEncoding::new_sinusoidal(max_seq_len, embedding_dim);
        let mut embeddings = Array2::zeros((embedding_dim, seq_len));

        let result = positional_encoding.add_to_embeddings(&mut embeddings);

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "Sequence length exceeds maximum sequence length for positional encoding"
        );
    }

    #[test]
    fn test_add_to_embeddings_dimension_mismatch() {
        let max_seq_len = 10;
        let embedding_dim = 4;
        let wrong_dim = 6;
        let seq_len = 5;

        let positional_encoding = PositionalEncoding::new_sinusoidal(max_seq_len, embedding_dim);
        let mut embeddings = Array2::zeros((wrong_dim, seq_len));

        let result = positional_encoding.add_to_embeddings(&mut embeddings);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Embedding dimension mismatch");
    }

    #[test]
    fn test_for_sequence() {
        let max_seq_len = 10;
        let embedding_dim = 4;
        let seq_len = 5;

        let positional_encoding = PositionalEncoding::new_sinusoidal(max_seq_len, embedding_dim);

        let result = positional_encoding.for_sequence(seq_len);

        assert!(result.is_ok());
        let pe_subset = result.unwrap();

        assert_eq!(pe_subset.shape(), &[embedding_dim, seq_len]);

        // Проверяем, что значения соответствуют ожидаемым
        for i in 0..embedding_dim {
            for j in 0..seq_len {
                assert_abs_diff_eq!(pe_subset[[i, j]], positional_encoding.encoding[[i, j]], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_for_sequence_too_long() {
        let max_seq_len = 5;
        let embedding_dim = 4;
        let seq_len = 10; // Больше, чем max_seq_len

        let positional_encoding = PositionalEncoding::new_sinusoidal(max_seq_len, embedding_dim);

        let result = positional_encoding.for_sequence(seq_len);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Requested sequence length exceeds maximum sequence length");
    }
}
