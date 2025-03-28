use thiserror::Error;

#[derive(Debug, Error)]
pub enum PositionalEncodingError {
    #[error("Input shape does not match expected dimensions")]
    ShapeMismatch,
    #[error("Sequence length exceeds maximum sequence length for positional encoding")]
    SequenceLengthExceeded,
    #[error("Embedding dimension mismatch")]
    EmbeddingDimensionMismatch,
}

#[derive(Debug, Error)]
pub enum EmbeddingError {
    #[error("Failed to create Uniform distribution: {0}")]
    UniformCreationFailed(String),
    #[error("Failed to create a new Normal instance: {0}")]
    NormalCreationFailed(String),
    #[error("Token index is out of vocabulary bounds)")]
    OutOfVocabularyError,
    #[error("Failed to convert embeddings to slice")]
    SliceConversionFailed,
    #[error("Failed to create tensor from embeddings: {0}")]
    TensorCreationFailed(String),
    #[error("Failed to serialize embeddings to file: {0}")]
    SerializationFailed(String),
    #[error("Failed to open file: {0}")]
    FileOpenError(String),
    #[error("Failed to read file: {0}")]
    FileReadError(String),
    #[error("Failed to deserialize data: {0}")]
    DeserializationError(String),
    #[error("Failed to retrieve tensor: {0}")]
    TensorRetrievalError(String),
    #[error("Invalid data type: expected F32")]
    InvalidDataType,
    #[error("Failed to convert data to embedding matrix: {0}")]
    DataConversionError(String),
}
