use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let tokens = vec![5, 6, 3, 8];
    let embedding_dimension = 3;
    let std_dev = 0.01f32;
    let max_seq_len = 10;
    let vocab_size = 10;
    let mean = 0.0f32;

    // ---------------------------------------------------------------------------------- //

    // Инициализируем новую матрицу эмбеддингов (Xavier (Glorot))
    let embeddings_xavier = tictonix::Embeddings::new_xavier(vocab_size, embedding_dimension);
    println!("Исходная матрица Xavier:\n{}", embeddings_xavier.get_matrix());

    // Инициализируем новую матрицу эмбеддингов (Gaussian)
    let embeddings_gaussian = tictonix::Embeddings::new_gaussian(vocab_size, embedding_dimension, mean, std_dev);
    println!("Исходная матрица Gaussian:\n{}", embeddings_gaussian.get_matrix());

    // Инициализируем новую матрицу эмбеддингов (Uniform)
    let embeddings_uniform = tictonix::Embeddings::new_uniform(vocab_size, embedding_dimension);
    println!("Исходная матрица Uniform:\n{}", embeddings_uniform.get_matrix());

    // ---------------------------------------------------------------------------------- //

    // Получим позиционные кодировки (Sinusoidal Positional Encoding)
    let positional_sinusoidal = tictonix::PositionalEncoding::new_sinusoidal(max_seq_len, embedding_dimension);
    println!("Позиционные кодировки (SPE):\n{}", positional_sinusoidal.for_sequence(tokens.len())?);

    // Получим позиционные кодировки (Relative Positional Encoding)
    let positional_relative = tictonix::PositionalEncoding::new_relative(max_seq_len, embedding_dimension);
    println!("Позиционные кодировки (RPE):\n{}", positional_relative.for_sequence(tokens.len())?);

    // Получим позиционные кодировки RoPE (Rotary Positional Embedding)
    let positional_rope = tictonix::PositionalEncoding::new_rope(max_seq_len, embedding_dimension);
    println!("Позиционные кодировки (RoPE):\n{}", positional_rope.for_sequence(tokens.len())?);

    // Применение RoPE к входной матрице
    let input_matrix = Array2::ones((embedding_dimension, max_seq_len));
    let output = positional_rope.apply_rope(&input_matrix);
    println!("Applied RoPE:\n{}", output);

    // ---------------------------------------------------------------------------------- //

    // Преобразуем вектор токенов в матрицу эмбеддингов (возьмем Uniform)
    let mut token_embeddings = embeddings_uniform.tokens_to_embeddings(&tokens)?;
    println!("Эмбеддинги токенов:\n{}", token_embeddings);

    // Применим позиционные кодировки (возьмем SRE) к матрице эмбеддингов
    positional_sinusoidal.add_to_embeddings(&mut token_embeddings)?;
    println!("Эмбеддинги с позиционными кодировками:\n{}", token_embeddings);

    // Сохраним эмбеддинги токенов с позиционными кодировками (матрицу) в файл
    tictonix::Embeddings::save_embeddings_to_file(&token_embeddings, "./example/test.safetensors")?;

    // Достанем эмбеддинги токенов с позиционными кодировками (матрицу) из файла
    // Матрица из файла должна совпадать с `Эмбеддинги с позиционными кодировками`
    let load_matrix = tictonix::Embeddings::load_embeddings_from_file("./example/test.safetensors")?;
    println!("Матрица полученная из файла:\n{}", load_matrix);

    Ok(())
}
