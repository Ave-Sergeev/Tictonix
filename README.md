## Tictonix

---

[Русская версия](README.ru.md)

### Description

This crate is the second step (step 1 [tokenizer](https://github.com/Ave-Sergeev/Tokenomicon)) towards a
`LLM` native implementation on the `Transformer` architecture.
It contains operations related to converting tokens into embeddings, encoding their positions.

#### Provided functionality:

- Embeddings structure

1) Creation of a new embeddings matrix with random filling.
2) Converting a vector of tokens into an embeddings matrix.
3) Saving to a file (safetensors format), and retrieving the embeddings matrix from the file.

- Structure of PositionalEncoding

1) Creating a new matrix of positional encodings.
2) Applying positional encodings to the embeddings matrix.
3) Return a portion of the positional encoding matrix for the sequence.

The crate has the following dependencies:

1) [rand](https://github.com/rust-random/rand) crate to generate pseudo-random values.
2) [ndarray](https://github.com/rust-ndarray/ndarray) crate (math) for efficient matrix handling.
3) [approx](https://github.com/brendanzab/approx) crate to handle approximate comparisons of floating point numbers.
4) ...

### Usage

See [example](/example/src/main.rs) for usage.

### Glossary

- Tokenization (segmentation) is the process of breaking text into individual parts (words, characters, etc.)
- LLM (large language models) is a general-purpose mathematical model designed for a wide range of tasks related to
  natural language processing.
- Transformer is a deep neural network architecture introduced in 2017 by researchers from Google. It is designed to
  process sequences such as natural language text.
- Embedding is numerical representations of text (token).
- Positional Encoding is a technique used (e.g. in Transformers) to provide positional information to a model by adding
  position-dependent signals to word embeddings.

### P.S.

Don't forget to leave a ⭐ if you found this project useful.
