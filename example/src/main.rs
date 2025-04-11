use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let tokens = vec![1, 3, 5, 7, 9];
    let embedding_dimension = 4;
    let std_dev = 0.01_f32;
    let max_seq_len = 10;
    let vocab_size = 10;
    let position = 3;
    let mean = 0.0_f32;

    // ---------------------------------------------------------------------------------- //

    // Let's consider examples of matrix creation by different methods:

    // Initialize a new embedding matrix (Xavier (Glorot))
    let embeddings_xavier = tictonix::Embeddings::new_xavier(vocab_size, embedding_dimension)?;
    println!("Initialized embedding matrix (Xavier):\n{:.6}\n", embeddings_xavier.getter_matrix());

    // Initialize a new embedding matrix (Gaussian)
    let embeddings_gaussian = tictonix::Embeddings::new_gaussian(vocab_size, embedding_dimension, mean, std_dev)?;
    println!("Initialized embedding matrix (Gaussian):\n{:.6}\n", embeddings_gaussian.getter_matrix());

    // Initialize a new embedding matrix (Uniform)
    let mut embeddings_uniform = tictonix::Embeddings::new_uniform(vocab_size, embedding_dimension)?;
    println!("Initialized embedding matrix (Uniform):\n{:.6}\n", embeddings_uniform.getter_matrix());

    // ---------------------------------------------------------------------------------- //

    // Let's look at the available operations on embeddings:

    // Let's build the resulting embedding matrix for the array of tokens (indices) from the original embedding matrix (take Uniform)
    let mut token_embeddings = embeddings_uniform.get_embeddings(&tokens)?;
    println!("Token embeddings (from Uniform):\n{:.6}\n", token_embeddings);

    // Let's get the embedding for a particular token (index) from the original embedding matrix (let's take Uniform)
    let specific_token_embedding = embeddings_uniform.get_embedding(tokens[0])?;
    println!("Specific token embedding (from Uniform):\n{:.6}\n", specific_token_embedding);

    // Let's update the embedding for a certain token (index) in the embedding matrix (let's take Uniform)
    let _ = embeddings_uniform.update_embedding(tokens[1], &specific_token_embedding)?;
    let altered_matrix = embeddings_uniform.getter_matrix();
    println!("Altered embedding matrix (from Uniform):\n{:.6}\n", altered_matrix);

    // ---------------------------------------------------------------------------------- //

    // Let's consider examples of creating a matrix of positional encodings in different ways:

    // Obtaining the matrix of position encodings by method SPE (Sinusoidal Positional Encoding) method
    let positional_sinusoidal = tictonix::PositionalEncoding::new_sinusoidal(max_seq_len, embedding_dimension);
    println!("Obtained matrix (by SPE method):\n{:.6}\n", positional_sinusoidal.getter_encoding());

    // Obtaining the matrix of position encodings by method RPE (Relative Positional Encoding)
    let positional_relative = tictonix::PositionalEncoding::new_relative(max_seq_len, embedding_dimension);
    println!("Obtained matrix (by RPE method):\n{:.6}\n", positional_relative.getter_encoding());

    // Obtaining the matrix of position encodings by method RoPE (Rotary Positional Embedding)
    let positional_rope = tictonix::PositionalEncoding::new_rope(max_seq_len, embedding_dimension);
    println!("Obtained matrix (by RoPE method):\n{:.6}\n", positional_rope.getter_encoding());

    // Applying RoPE to an input matrix
    let input_matrix = Array2::ones((embedding_dimension, max_seq_len));
    let output = positional_rope.apply_rope(&input_matrix)?;
    println!("Applied RoPE:\n{:.6}\n", output);

    // ---------------------------------------------------------------------------------- //

    // Let's look at the available operations on positional encoding:

    // Let's get the positional encoding for a sequence (let's take SRE)
    let positional_sinusoidal = tictonix::PositionalEncoding::new_sinusoidal(max_seq_len, embedding_dimension);
    println!(
        "Obtained positional encoding matrix:\n{:.6}\n",
        positional_sinusoidal.get_positional_encoding_slice(tokens.len())?
    );

    // Let's get the positional encoding (one-dimensional array) for a specific position in the sequence (let's take SRE)
    let positional_encoding = positional_sinusoidal.get_positional_encoding(position)?;
    println!("Positional encoding for a specific position:\n{:.6}\n", positional_encoding);

    // Let's apply positional encodings (let's take SRE) to the embedding matrix
    positional_sinusoidal.add_to_embeddings(&mut token_embeddings)?;
    println!("Embeddings with positional encodings:\n{:.6}\n", token_embeddings);

    // ---------------------------------------------------------------------------------- //

    // Let's look at how token embeddings can be saved and loaded:

    // Let's save token embeddings with positional encodings (matrix) to a files .npy and .safetensors
    tictonix::MatrixIO::save_to_safetensors(&token_embeddings, "./example/test.safetensors")?;
    tictonix::MatrixIO::save_to_npy(&token_embeddings, "./example/test.npy")?;
    println!("Matrix saved to file:\n{:.6}\n", token_embeddings);

    // Let's get token embeddings with positional encodings (matrix) from the file .safetensors
    let load_safetensors = tictonix::MatrixIO::load_from_safetensors("./example/test.safetensors")?;
    println!("Matrix obtained from file .safetensors:\n{:.6}\n", load_safetensors);

    // Let's get token embeddings with positional encodings (matrix) from the file .npy
    let load_npy = tictonix::MatrixIO::load_from_npy("./example/test.npy")?;
    println!("Matrix obtained from file .npy:\n{:.6}\n", load_npy);

    remove_file("./example/test.safetensors").expect("Failed to delete test file");
    remove_file("./example/test.npy").expect("Failed to delete test file");

    // ---------------------------------------------------------------------------------- //

    Ok(())
}
