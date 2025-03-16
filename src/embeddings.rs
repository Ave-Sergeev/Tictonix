use ndarray::Array2;
use rand::distr::{Distribution, Uniform};
use rand::rng;
use rand_distr::Normal;
use std::path::Path;

pub struct Embeddings {
    matrix: Array2<f32>,
    vocab_size: usize,
    embedding_dim: usize,
}

impl Embeddings {
    /// Инициализация новой матрицы эмбеддингов (равномерное распределение со случайным наполнением)
    pub fn new_uniform(vocab_size: usize, embedding_dim: usize) -> Self {
        let mut rng = rng();
        let uniform = Uniform::new_inclusive(-1.0, 1.0).expect("Fail to create a new Uniform instance");

        let matrix = Array2::from_shape_fn((embedding_dim, vocab_size), |_| uniform.sample(&mut rng));

        Self {
            matrix,
            vocab_size,
            embedding_dim,
        }
    }

    /// Инициализация новой матрицы эмбеддингов (Гауссово распределение)
    pub fn new_gaussian(vocab_size: usize, embedding_dim: usize, mean: f32, std_dev: f32) -> Self {
        let mut rng = rand::rng();
        let normal = Normal::new(mean, std_dev).expect("Fail to create a new Normal instance");

        let matrix = Array2::from_shape_fn((embedding_dim, vocab_size), |_| normal.sample(&mut rng) as f32);

        Self {
            matrix,
            vocab_size,
            embedding_dim,
        }
    }

    /// Инициализация новой матрицы эмбеддингов (Xavier (Glorot))
    pub fn new_xavier(vocab_size: usize, embedding_dim: usize) -> Self {
        let mut rng = rand::rng();
        let std_dev = (2.0 / (vocab_size as f32 + embedding_dim as f32)).sqrt();
        let normal = Normal::new(0.0, std_dev).expect("Fail to create a new Normal instance");

        let matrix = Array2::from_shape_fn((embedding_dim, vocab_size), |_| normal.sample(&mut rng) as f32);

        Self {
            matrix,
            vocab_size,
            embedding_dim,
        }
    }

    /// Геттер для матрицы эмбеддингов
    pub fn get_matrix(&self) -> &Array2<f32> {
        &self.matrix
    }

    /// Преобразование вектора токенов в матрицу эмбеддингов
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

    /// Сохранение матрицы эмбеддингов в файл
    pub fn save_to_file(&self, path: &Path) -> Result<(), String> {
        unimplemented!()
    }

    /// Загрузки матрицы эмбеддингов из файла
    pub fn load_from_file(path: &Path) -> Result<Self, String> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embeddings_new_uniform() {
        let vocab_size = 10;
        let embedding_dim = 5;

        let embeddings = Embeddings::new_uniform(vocab_size, embedding_dim);

        // Проверяем размеры матрицы
        assert_eq!(embeddings.matrix.shape(), &[embedding_dim, vocab_size]);
        assert_eq!(embeddings.vocab_size, vocab_size);
        assert_eq!(embeddings.embedding_dim, embedding_dim);

        // Проверяем, что все значения находятся в диапазоне [-1.0, 1.0]
        for &value in embeddings.matrix.iter() {
            assert!(value >= -1.0 && value <= 1.0);
        }
    }

    #[test]
    fn test_embeddings_new_gaussian() {
        let vocab_size = 10;
        let embedding_dim = 5;
        let mean = 0.0f32;
        let std_dev = 0.01f32;

        let embeddings = Embeddings::new_gaussian(vocab_size, embedding_dim, mean, std_dev);

        // Проверяем размеры матрицы
        assert_eq!(embeddings.matrix.shape(), &[embedding_dim, vocab_size]);
        assert_eq!(embeddings.vocab_size, vocab_size);
        assert_eq!(embeddings.embedding_dim, embedding_dim);

        // Проверяем, что значения близки к среднему (mean)
        let sum: f32 = embeddings.matrix.iter().sum();
        let avg = sum / (vocab_size * embedding_dim) as f32;
        // Проверяем, что среднее близко к 0.0
        assert!((avg - mean).abs() < 0.1);
    }

    #[test]
    fn test_embeddings_new_xavier() {
        let vocab_size = 10;
        let embedding_dim = 5;

        let embeddings = Embeddings::new_xavier(vocab_size, embedding_dim);

        // Проверяем размеры матрицы
        assert_eq!(embeddings.matrix.shape(), &[embedding_dim, vocab_size]);
        assert_eq!(embeddings.vocab_size, vocab_size);
        assert_eq!(embeddings.embedding_dim, embedding_dim);

        // Проверяем, что все значения находятся в диапазоне [-1.0, 1.0]
        for &value in embeddings.matrix.iter() {
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

        // Проверяем размеры результирующей матрицы
        assert_eq!(result.shape(), &[embedding_dim, tokens.len()]);

        // Проверяем, что эмбеддинги соответствуют ожидаемым
        for (i, &token) in tokens.iter().enumerate() {
            let expected_embedding = embeddings.matrix.column(token);
            let actual_embedding = result.column(i);

            for (e, a) in expected_embedding.iter().zip(actual_embedding.iter()) {
                assert_eq!(*e, *a);
            }
        }
    }

    #[test]
    fn test_tokens_to_embeddings_out_of_bounds() {
        let vocab_size = 10;
        let embedding_dim = 5;
        let embeddings = Embeddings::new_uniform(vocab_size, embedding_dim);

        // Токен 15 превышает размер словаря (10)
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

        // Проверяем, что получили матрицу правильной размерности (embedding_dim x 0)
        assert_eq!(result.shape(), &[embedding_dim, 0]);
    }

    #[test]
    fn test_get_matrix() {
        let vocab_size = 10;
        let embedding_dim = 5;
        let embeddings = Embeddings::new_uniform(vocab_size, embedding_dim);

        let matrix = embeddings.get_matrix();

        // Проверяем, что геттер возвращает матрицу с правильными размерами
        assert_eq!(matrix.shape(), &[embedding_dim, vocab_size]);
        // Проверяем, что матрица совпадает с оригинальной
        assert_eq!(*matrix, embeddings.matrix);
    }
}
