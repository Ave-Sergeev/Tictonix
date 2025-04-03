## Tictonix

---

[Русская версия](https://github.com/Ave-Sergeev/Tictonix/blob/main/README.ru.md)

### Description

This crate is the second step (step 1 [tokenizer](https://github.com/Ave-Sergeev/Tokenomicon)) towards a
`LLM` native implementation on the `Transformer` architecture.
It contains operations related to converting tokens into embeddings, encoding their positions.

### Provided functionality:

- Embeddings structure

1) Creating a new embedding matrix by various methods such as: `Gaussian`, `Xavier`, `Uniform`.
2) Constructing the resulting embedding matrix for an array of tokens (indices), and obtaining a specific embedding for a token (index).
3) Updating (replacing) the embedding for a particular token (index).
4) Saving to a file (.safetensors format), and retrieving the embedding matrix from the file.

- Structure of PositionalEncoding

1) Creating a new positional encoding matrix by various methods such as: `Sinusoidal PE`, `Relative PE`, `Rotary PE`.
2) Applying positional encodings to the embedding matrix.
3) Returning a part of the positional encoding matrix for a sequence, and a particular positional encoding by its position.

The crate has the following dependencies:

1) [rand](https://github.com/rust-random/rand) crate for generating pseudo-random values.
2) [ndarray](https://github.com/rust-ndarray/ndarray) crate (mathematical) for efficient work with matrices.
3) [anyhow](https://github.com/dtolnay/anyhow) crate for idiomatic error handling.
4) [approx](https://github.com/brendanzab/approx) crate for working with approximate comparisons of floating-point numbers.
5) [bytemuck](https://github.com/Lokathor/bytemuck) crate for converting simple data types.
6) [thiserror](https://github.com/dtolnay/thiserror) crate for convenient error output.
7) [safetensors](https://github.com/huggingface/safetensors) crate for safe storage of tensors.
8) ...

### Usage

See [example](https://github.com/Ave-Sergeev/Tictonix/blob/main/example/src/main.rs) for usage.

### Glossary

- Tokenization is the process of breaking text into separate elements called tokens.
  Tokens can be words, characters, sub-words, or other units, depending on the chosen tokenization method.
  This process is an important step in text preprocessing for Natural Language Processing (NLP) tasks.
- LLMs (large language models) are large language models based on deep learning architectures (e.g.,
  Transformer) that are trained on huge amounts of textual data. They are designed to perform a wide
  range of tasks related to natural language processing, such as text generation, translation, question answering,
  classification, and others. LLMs are capable of generalizing knowledge and performing tasks on which they have not
  been explicitly trained (zero-shot or few-shot learning).
- Transformer is a neural network architecture proposed in 2017 that uses the attention mechanism to process sequences
  of data such as text.
  The main advantage of Transformer is its ability to process long sequences and take context into account regardless of
  the distance between elements of the sequence.
  This architecture is the basis for most modern LLMs (such as GPT, BERT and others).
- Embedding is a numerical (vector) representation of text data (tokens, words, phrases or sentences).
- Positional Encoding is a technique used in the Transformer architecture to convey information about the order of
  elements in a sequence. Since Transformer has no built-in order information (unlike recurrent networks),
  positional encoding adds special signals to token embeddings that depend on their position in the sequence.
  sequence. This allows the model to take into account the order of words or other elements in the input data.

### P.S.

Don't forget to leave a ⭐ if you found this project useful.
