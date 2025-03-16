use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let tokens = vec![5, 6, 3, 8];
    let embedding_dimension = 3;
    let vocab_size = 10;

    // Для примера используем равномерное распределение со случайным наполнением (new_uniform)
    let embeddings = tictonix::Embeddings::new_uniform(vocab_size, embedding_dimension);
    println!("Исходная матрица эмбеддингов:\n{}", embeddings.get_matrix());

    let mut token_embeddings = embeddings.tokens_to_embeddings(&tokens)?;
    println!("Эмбеддинги токенов:\n{}", token_embeddings);

    let positional_encoding = tictonix::PositionalEncoding::new(100, embedding_dimension);
    println!(
        "Позиционные кодировки (для данной последовательности): \n {}",
        positional_encoding.for_sequence(tokens.len())?
    );

    positional_encoding.add_to_embeddings(&mut token_embeddings)?;
    println!("Эмбеддинги с позиционными кодировками: \n {}", token_embeddings);

    Ok(())
}
