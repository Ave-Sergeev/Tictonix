use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let tokens = vec![5, 6, 3, 8];
    let embedding_dimension = 3;
    let std_dev = 0.01f32;
    let max_seq_len = 10;
    let vocab_size = 10;
    let mean = 0.0f32;

    // ---------------------------------------------------------------------------------- //

    // Initialize a new embedding matrix (Xavier (Glorot))
    let embeddings_xavier = tictonix::Embeddings::new_xavier(vocab_size, embedding_dimension);
    println!("Initialized matrix (Xavier):\n{}", embeddings_xavier.get_matrix());

    // Initialize a new embedding matrix (Gaussian)
    let embeddings_gaussian = tictonix::Embeddings::new_gaussian(vocab_size, embedding_dimension, mean, std_dev);
    println!("Initialized matrix (Gaussian):\n{}", embeddings_gaussian.get_matrix());

    // Initialize a new embedding matrix (Uniform)
    let embeddings_uniform = tictonix::Embeddings::new_uniform(vocab_size, embedding_dimension);
    println!("Initialized matrix (Uniform):\n{}", embeddings_uniform.get_matrix());

    // ---------------------------------------------------------------------------------- //

    // Let's get positional encodings SPE (Sinusoidal Positional Encoding)
    let positional_sinusoidal = tictonix::PositionalEncoding::new_sinusoidal(max_seq_len, embedding_dimension);
    println!("Positional encodingsS (SPE):\n{}", positional_sinusoidal.for_sequence(tokens.len())?);

    // Let's get positional encodings RPE (Relative Positional Encoding)
    let positional_relative = tictonix::PositionalEncoding::new_relative(max_seq_len, embedding_dimension);
    println!("Positional encodings (RPE):\n{}", positional_relative.for_sequence(tokens.len())?);

    // Let's get positional encodings RoPE (Rotary Positional Embedding)
    let positional_rope = tictonix::PositionalEncoding::new_rope(max_seq_len, embedding_dimension);
    println!("Positional encodings (RoPE):\n{}", positional_rope.for_sequence(tokens.len())?);

    // Applying RoPE to an input matrix
    let input_matrix = Array2::ones((embedding_dimension, max_seq_len));
    let output = positional_rope.apply_rope(&input_matrix);
    println!("Applied RoPE:\n{}", output);

    // ---------------------------------------------------------------------------------- //

    // Transform the token vector into an embedding matrix (let's take Uniform)
    let mut token_embeddings = embeddings_uniform.tokens_to_embeddings(&tokens)?;
    println!("Token Embeddings:\n{}", token_embeddings);

    // Let's apply positional encodings (let's take SRE) to the embedding matrix
    positional_sinusoidal.add_to_embeddings(&mut token_embeddings)?;
    println!("Embeddings with positional encodings:\n{}", token_embeddings);

    // Let's save token embeddings with positional encodings (matrix) to a file
    tictonix::Embeddings::save_embeddings_to_file(&token_embeddings, "./example/test.safetensors")?;

    // Let's get token embeddings with positional encodings (matrix) from the file
    let load_matrix = tictonix::Embeddings::load_embeddings_from_file("./example/test.safetensors")?;
    println!("Matrix obtained from file:\n{}", load_matrix);

    Ok(())
}
