use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let tokens = vec![5, 6, 3, 8];
    let embedding_dimension = 3;
    let std_dev = 0.01f32;
    let max_seq_len = 10;
    let vocab_size = 10;
    let position = 3;
    let mean = 0.0f32;

    // ---------------------------------------------------------------------------------- //

    // Initialize a new embedding matrix (Xavier (Glorot))
    let embeddings_xavier = tictonix::Embeddings::new_xavier(vocab_size, embedding_dimension);
    println!("Initialized matrix (Xavier):\n{}", embeddings_xavier.get_matrix());

    // Initialize a new embedding matrix (Gaussian)
    let embeddings_gaussian = tictonix::Embeddings::new_gaussian(vocab_size, embedding_dimension, mean, std_dev)?;
    println!("Initialized matrix (Gaussian):\n{}", embeddings_gaussian.get_matrix());

    // Initialize a new embedding matrix (Uniform)
    let embeddings_uniform = tictonix::Embeddings::new_uniform(vocab_size, embedding_dimension)?;
    println!("Initialized matrix (Uniform):\n{}", embeddings_uniform.get_matrix());

    // ---------------------------------------------------------------------------------- //

    // Let's get positional encodings SPE (Sinusoidal Positional Encoding)
    let positional_sinusoidal = tictonix::PositionalEncoding::new_sinusoidal(max_seq_len, embedding_dimension);
    println!("Positional encodingsS (SPE):\n{}", positional_sinusoidal.get_positional_encoding_slice(tokens.len())?);

    // Let's get positional encodings RPE (Relative Positional Encoding)
    let positional_relative = tictonix::PositionalEncoding::new_relative(max_seq_len, embedding_dimension);
    println!("Positional encodings (RPE):\n{}", positional_relative.get_positional_encoding_slice(tokens.len())?);

    // Let's get positional encodings RoPE (Rotary Positional Embedding)
    let positional_rope = tictonix::PositionalEncoding::new_rope(max_seq_len, embedding_dimension);
    println!("Positional encodings (RoPE):\n{}", positional_rope.get_positional_encoding_slice(tokens.len())?);

    // Applying RoPE to an input matrix
    let input_matrix = Array2::ones((embedding_dimension, max_seq_len));
    let output = positional_rope.apply_rope(&input_matrix)?;
    println!("Applied RoPE:\n{}", output);

    // Let's get the positional encoding (one-dimensional array) for a specific position in the sequence (let's take SRE)
    let positional_encoding = positional_sinusoidal.get_positional_encoding(position)?;
    println!("Positional encoding for a specific position:\n{}", positional_encoding);

    // ---------------------------------------------------------------------------------- //

    // Obtaine of the resulting embedding matrix for an array of tokens (indices) from the original embedding matrix (let's take Uniform)
    let mut token_embeddings = embeddings_uniform.get_embeddings(&tokens)?;
    println!("Token Embeddings:\n{}", token_embeddings);

    // Obtaine an embedding for a specific token (index) from the initial matrix of embeddings (let's take Uniform)
    let token_embedding = embeddings_uniform.get_embedding(token[0])?;
    println!("Token Embedding:\n{}", token_embedding);

    // Update the embedding for a specific token (index) in the embedding matrix
    let _ = embeddings_uniform.update_embedding(token[1], token_embedding)?;
    let altered_matrix = embeddings_uniform.get_matrix();
    println!("Altered embedding matrix:\n{}", altered_matrix);

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
