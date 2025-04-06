use crate::error::IOError;
use anyhow::{Error, Result};
use bytemuck::cast_slice;
use ndarray::Array2;
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors, serialize_to_file};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

pub struct MatrixIO;

impl MatrixIO {
    /// Saving the matrix to a file
    ///
    /// # Parameters
    /// - `matrix`: The `Array2<f32>` matrix to save.
    /// - `file_path`: The path to the file where the matrix will be saved.
    ///
    /// # Returns
    /// - `Ok(())`: If saving was successful.
    /// - `Err(anyhow::Error)`: If there was a problem creating the tensor, serializing it, or writing it to the file.
    ///
    /// # Errors
    /// - `SliceConversionFailed`: Occurs if the matrix cannot be converted to a contiguous slice.
    /// - `TensorCreationFailed`: Occurs if the tensor cannot be created from the matrix data.
    /// - `SerializationFailed`: Occurs if the tensor data cannot be serialized or written to the specified file.
    pub fn save_matrix_to_file(matrix: &Array2<f32>, file_path: &str) -> Result<(), Error> {
        let shape = matrix.shape().to_vec();

        let data = matrix
            .as_slice()
            .ok_or(IOError::SliceConversionFailed)?
            .iter()
            .flat_map(|float| float.to_le_bytes())
            .collect::<Vec<_>>();

        let tensor =
            TensorView::new(Dtype::F32, shape, &data).map_err(|err| IOError::TensorCreationFailed(err.to_string()))?;

        let mut tensors_map = HashMap::new();
        tensors_map.insert("matrix", tensor);

        serialize_to_file(tensors_map, &None, file_path.as_ref())
            .map_err(|err| IOError::SerializationFailed(err.to_string()))?;

        println!("Matrix successfully saved to file: {file_path}\n");

        Ok(())
    }

    /// Loading matrix from file
    ///
    /// # Parameters
    /// - `file_path`: Path to the file from which to load the matrix.
    ///
    /// # Returns
    /// - `Ok(Array2<f32>)`: The matrix loaded from the file.
    /// - `Err(anyhow::Error)`: Error if there was a problem opening the file, reading, deserializing, or converting the data.
    ///
    /// # Errors
    /// - `FileOpenError`: Occurs if the file at the specified path cannot be opened.
    /// - `FileReadError`: Occurs if the file cannot be read. This may happen due to I/O errors or corrupted file data.
    /// - `DeserializationError`: Occurs if the data in the file cannot be deserialized into a valid tensor format.
    /// - `TensorRetrievalError`: Occurs if the tensor named "matrix" cannot be retrieved from the deserialized data.
    /// - `InvalidDataType`: Occurs if the tensor data type is not `f32`. This ensures the matrix is loaded with the correct data type.
    /// - `DataConversionError`: Occurs if the tensor data cannot be converted into a 2D array (`Array2<f32>`).
    pub fn load_matrix_from_file(file_path: &str) -> Result<Array2<f32>, Error> {
        let mut buffer = Vec::new();

        File::open(file_path)
            .map_err(|err| IOError::FileOpenError(err.to_string()))?
            .read_to_end(&mut buffer)
            .map_err(|err| IOError::FileReadError(err.to_string()))?;

        let safe_tensors =
            SafeTensors::deserialize(&buffer).map_err(|err| IOError::DeserializationError(err.to_string()))?;

        let tensor = safe_tensors
            .tensor("matrix")
            .map_err(|err| IOError::TensorRetrievalError(err.to_string()))?;

        if tensor.dtype() != Dtype::F32 {
            return Err(Error::from(IOError::InvalidDataType));
        }

        let data_bytes = tensor.data();
        let data_f32 = cast_slice(data_bytes);
        let shape = tensor.shape();

        let matrix = Array2::from_shape_vec((shape[0], shape[1]), data_f32.to_vec())
            .map_err(|err| IOError::DataConversionError(err.to_string()))?;

        println!("Matrix successfully loaded from file: {file_path}\n");

        Ok(matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::remove_file;
    use std::io::Write;

    #[test]
    fn test_save_and_load_matrix() {
        let file_path = "test_matrix.safetensors";
        let matrix = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        MatrixIO::save_matrix_to_file(&matrix, file_path).expect("Failed to save matrix");

        let loaded_matrix = MatrixIO::load_matrix_from_file(file_path).expect("Failed to load matrix");

        assert_eq!(matrix, loaded_matrix);

        remove_file(file_path).expect("Failed to delete test file");
    }

    #[test]
    fn test_load_invalid_file() {
        let file_path = "invalid_matrix.safetensors";
        let mut file = File::create(file_path).expect("Failed to create test file");

        file.write_all(b"invalid data").expect("Failed to write test data");

        let loaded_matrix = MatrixIO::load_matrix_from_file(file_path);

        assert!(loaded_matrix.is_err());

        remove_file(file_path).expect("Failed to delete test file");
    }

    #[test]
    fn test_save_invalid_path() {
        let file_path = "non_existent_directory/test_matrix.safetensors";
        let matrix = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let saved_matrix = MatrixIO::save_matrix_to_file(&matrix, file_path);

        assert!(saved_matrix.is_err());
    }
}
