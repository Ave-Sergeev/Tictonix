## Tictonix

---

[Русская версия](https://github.com/Ave-Sergeev/Tictonix/blob/main/README.ru.md)

### Description

This crate is the second step (step 1 [tokenizer](https://github.com/Ave-Sergeev/Tokenomicon)) towards a
`LLM` native implementation on the `Transformer` architecture.
It contains operations related to converting tokens into embeddings, encoding their positions.

### Provided functionality:

- Embeddings structure

1) Creation of a new embedding matrix by different methods (`Gaussian`, `Xavier`, `Uniform`).
2) Converting a vector of tokens into an embeddings matrix.
3) Saving to a file (.safetensors format), and retrieving the embeddings matrix from the file.

- Structure of PositionalEncoding

1) Creation of a new positional encoding matrix by various methods such as:
   `Sinusoidal Positional Encoding`, `Relative Positional Encoding`, `Rotary Positional Embedding`.
2) Applying positional encodings to the embeddings matrix.
3) Return a portion of the positional encoding matrix for the sequence.

The crate has the following dependencies:

1) [rand](https://github.com/rust-random/rand) crate to generate pseudo-random values.
2) [ndarray](https://github.com/rust-ndarray/ndarray) crate (math) for efficient matrix handling.
3) [approx](https://github.com/brendanzab/approx) crate to handle approximate comparisons of floating point numbers.
4) [bytemuck](https://github.com/Lokathor/bytemuck) crate for converting simple data types.
5) [safetensors](https://github.com/huggingface/safetensors) crate for safe storage of tensors.
6) ...

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
