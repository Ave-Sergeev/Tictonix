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
    /// Инициализация новой матрицы эмбеддингов (равномерное распределение со случайным наполнением)
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

    /// Инициализация новой матрицы эмбеддингов (Гауссово распределение)
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

    /// Инициализация новой матрицы эмбеддингов (Xavier (Glorot))
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

        serialize_to_file(tensors_map, &None, file_path.as_ref()).map_err(|err| format!("Failed serialize: {err}"))?;

        println!("Матрица эмбеддингов успешно сохранена в файл: {file_path}");
        Ok(())
    }

    /// Загрузки матрицы эмбеддингов из файла
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

        println!("Матрица эмбеддингов успешно загружена из файла: {file_path}");
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

        // Проверяем размеры матрицы
        assert_eq!(embeddings_uniform.matrix.shape(), &[embedding_dim, vocab_size]);
        assert_eq!(embeddings_uniform.vocab_size, vocab_size);
        assert_eq!(embeddings_uniform.embedding_dim, embedding_dim);

        // Проверяем, что все значения находятся в диапазоне [-1.0, 1.0]
        for &value in embeddings_uniform.matrix.iter() {
            assert!(value >= -1.0 && value <= 1.0);
        }
    }

    #[test]
    fn test_embeddings_new_gaussian() {
        let vocab_size = 10;
        let embedding_dim = 5;
        let mean = 0.0f32;
        let std_dev = 0.01f32;

        let embeddings_gaussian = Embeddings::new_gaussian(vocab_size, embedding_dim, mean, std_dev);

        // Проверяем размеры матрицы
        assert_eq!(embeddings_gaussian.matrix.shape(), &[embedding_dim, vocab_size]);
        assert_eq!(embeddings_gaussian.vocab_size, vocab_size);
        assert_eq!(embeddings_gaussian.embedding_dim, embedding_dim);

        // Проверяем, что значения близки к среднему (mean)
        let sum: f32 = embeddings_gaussian.matrix.iter().sum();
        let avg = sum / (vocab_size * embedding_dim) as f32;
        // Проверяем, что среднее близко к 0.0
        assert!((avg - mean).abs() < 0.1);
    }

    #[test]
    fn test_embeddings_new_xavier() {
        let vocab_size = 10;
        let embedding_dim = 5;

        let embeddings_xavier = Embeddings::new_xavier(vocab_size, embedding_dim);

        // Проверяем размеры матрицы
        assert_eq!(embeddings_xavier.matrix.shape(), &[embedding_dim, vocab_size]);
        assert_eq!(embeddings_xavier.vocab_size, vocab_size);
        assert_eq!(embeddings_xavier.embedding_dim, embedding_dim);

        // Проверяем, что все значения находятся в диапазоне [-1.0, 1.0]
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

    #[test]
    fn test_save_and_load_embeddings() {
        // Создаем тестовую матрицу эмбеддингов
        let embeddings = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let file_path = "test_embeddings.safetensors";

        // Сохраняем матрицу в файл
        Embeddings::save_embeddings_to_file(&embeddings, file_path).expect("Failed to save embeddings");

        // Загружаем матрицу из файла
        let loaded_embeddings = Embeddings::load_embeddings_from_file(file_path).expect("Failed to load embeddings");

        // Проверяем, что загруженная матрица совпадает с исходной
        assert_eq!(embeddings, loaded_embeddings);

        // Удаляем временный файл после теста
        remove_file(file_path).expect("Failed to delete test file");
    }

    #[test]
    fn test_load_invalid_file() {
        // Создаем файл с некорректными данными
        let file_path = "invalid_embeddings.safetensors";
        let mut file = File::create(file_path).expect("Failed to create test file");
        file.write_all(b"invalid data").expect("Failed to write test data");

        // Пытаемся загрузить матрицу из файла
        let result = Embeddings::load_embeddings_from_file(file_path);

        // Проверяем, что функция возвращает ошибку
        assert!(result.is_err());

        // Удаляем временный файл после теста
        remove_file(file_path).expect("Failed to delete test file");
    }

    #[test]
    fn test_save_invalid_path() {
        // Пытаемся сохранить матрицу в несуществующую директорию
        let embeddings = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let file_path = "non_directory/test_embeddings.safetensors";

        // Проверяем, что функция возвращает ошибку
        let result = Embeddings::save_embeddings_to_file(&embeddings, file_path);

        assert!(result.is_err());
    }
}
