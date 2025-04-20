use crate::error::IOError;
use anyhow::{Error, Result};
use bytemuck::cast_slice;
use ndarray::Array2;
use regex::Regex;
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors, serialize_to_file};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Write};

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
    pub fn save_to_safetensors(matrix: &Array2<f32>, file_path: &str) -> Result<(), Error> {
        let shape = matrix.shape().to_vec();

        let matrix_contiguous = matrix.as_standard_layout();

        let data = matrix_contiguous
            .as_slice()
            .ok_or_else(|| {
                log::error!("Could not convert matrix to slice");
                IOError::SliceConversionFailed
            })?
            .iter()
            .flat_map(|float| float.to_le_bytes())
            .collect::<Vec<_>>();

        let tensor = TensorView::new(Dtype::F32, shape, &data).map_err(|err| {
            log::error!("Tensor creation failed: {err}");
            IOError::TensorCreationFailed(err.to_string())
        })?;

        let mut tensors_map = HashMap::with_capacity(1);
        tensors_map.insert("matrix", tensor);

        serialize_to_file(tensors_map, &None, file_path.as_ref()).map_err(|err| {
            log::error!("Serialization failed: {err}");
            IOError::SerializationFailed(err.to_string())
        })?;

        log::info!("Matrix successfully saved to file: {file_path}");

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
    pub fn load_from_safetensors(file_path: &str) -> Result<Array2<f32>, Error> {
        let mut buffer = Vec::new();

        File::open(file_path)
            .map_err(|err| {
                log::error!("File open error: {err}");
                IOError::FileOpenError(err.to_string())
            })?
            .read_to_end(&mut buffer)
            .map_err(|err| {
                log::error!("File read error: {err}");
                IOError::FileReadError(err.to_string())
            })?;

        let safe_tensors = SafeTensors::deserialize(&buffer).map_err(|err| {
            log::error!("Deserialize error: {err}");
            IOError::DeserializationError(err.to_string())
        })?;

        let tensor = safe_tensors.tensor("matrix").map_err(|err| {
            log::error!("Failed to retrieve tensor: {err}");
            IOError::TensorRetrievalError(err.to_string())
        })?;

        if tensor.dtype() != Dtype::F32 {
            log::error!("Invalid data type: expected F32");
            return Err(Error::from(IOError::InvalidDataType));
        }

        let data_bytes = tensor.data();
        let data_f32 = cast_slice(data_bytes);
        let shape = tensor.shape();

        let matrix = Array2::from_shape_vec((shape[0], shape[1]), data_f32.to_vec()).map_err(|err| {
            log::error!("Failed to convert data to embedding matrix");
            IOError::DataConversionError(err.to_string())
        })?;

        log::info!("Matrix successfully loaded from file: {file_path}");

        Ok(matrix)
    }

    /// Saving a matrix to an NPY file (`NumPy` binary format).
    ///
    /// # Parameters
    /// - `matrix`: A reference to a 2D array (`Array2<f32>`) containing the data to be saved.
    /// - `file_path`: Path to the output file where the NPY data will be written.
    ///
    /// # Returns
    /// - `Ok(())`: Indicates successful writing of the matrix to the file.
    /// - `Err(anyhow::Error)`: An error if file creation, header serialization, or data writing fails.
    ///
    /// # Errors
    /// - `IOError`: Occurs if the file cannot be created or written to.
    /// - `InvalidFormat`: Occurs if the matrix cannot be represented as a contiguous slice (e.g., due to non-standard strides).
    pub fn save_to_npy(matrix: &Array2<f32>, file_path: &str) -> Result<(), Error> {
        let file = File::create(file_path)?;
        let mut writer = BufWriter::new(file);

        let shape = matrix.shape();
        let header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}", shape[0], shape[1]);

        let header_len = header.len() + 1;
        let total_len = 10 + header_len;
        let padding = (64 - (total_len % 64)) % 64;

        writer.write_all(b"\x93NUMPY\x01\x00")?;
        writer.write_all(&u16::try_from(header_len + padding)?.to_le_bytes())?;
        writer.write_all(header.as_bytes())?;
        writer.write_all(b"\n")?;
        writer.write_all(&vec![b' '; padding])?;

        let matrix_contiguous = matrix.as_standard_layout();

        let data = matrix_contiguous.as_slice().ok_or_else(|| {
            log::error!("Error getting slice from contiguous matrix: Matrix is not contiguous or empty");
            IOError::InvalidFormat
        })?;

        for &value in data {
            writer.write_all(&value.to_le_bytes())?;
        }

        log::info!("Matrix successfully saved to file: {file_path}\n");

        Ok(())
    }

    /// Loads a two-dimensional array of 32-bit floats from an NPY file (`NumPy` binary format).
    ///
    /// # Parameters
    /// - `file_path`: Path to the `.npy` file to load.
    ///
    /// # Returns
    /// - `Ok(Array2<f32>)`: The deserialized matrix with shape and data matching the file.
    /// - `Err(Error)`: If the file is invalid, unsupported, or contains incompatible data.
    ///
    /// # Errors
    /// - `InvalidFormat`: Occurs if:
    ///   - The file lacks the NPY magic number (`\x93NUMPY`).
    ///   - The header is malformed (e.g., invalid JSON, missing keys, or non-float32 `descr`).
    ///   - The data cannot be parsed as little-endian `f32`.
    /// - `UnsupportedVersion`: NPY version is not `1.0`.
    /// - `FortranOrderNotSupported`: The array uses Fortran (column-major) order.
    /// - `ShapeMismatch`: The actual data length does not match the shape declared in the header.
    /// - `IOError`: File read failures (e.g., permission issues or premature EOF).
    /// - `HeaderParseError`: Failed to parse the header (e.g., invalid regex or structure).
    pub fn load_from_npy(file_path: &str) -> Result<Array2<f32>, Error> {
        let mut magic = [0_u8; 6];
        let mut version = [0_u8; 2];
        let mut header_len_bytes = [0_u8; 2];

        let mut file = File::open(file_path)?;

        file.read_exact(&mut magic)?;

        if &magic != b"\x93NUMPY" {
            log::error!("Invalid format");
            return Err(Error::from(IOError::InvalidFormat));
        }

        file.read_exact(&mut version)?;

        if version != [0x01, 0x00] {
            log::error!("Unsupported version");
            return Err(Error::from(IOError::UnsupportedVersion));
        }

        file.read_exact(&mut header_len_bytes)?;

        let header_len = u16::from_le_bytes(header_len_bytes) as usize;

        let mut header = vec![0_u8; header_len];
        file.read_exact(&mut header)?;
        let header_str = String::from_utf8(header).map_err(|_| {
            log::error!("Invalid format");
            IOError::InvalidFormat
        })?;

        let (shape, fortran_order) = Self::parse_header(&header_str)?;
        if fortran_order {
            log::error!("Fortran order not supported");
            return Err(Error::from(IOError::FortranOrderNotSupported));
        }

        let total_elements = shape.0 * shape.1;
        let mut data = vec![0.0; total_elements];

        for item in data.iter_mut().take(total_elements) {
            let mut bytes = [0_u8; 4];

            file.read_exact(&mut bytes)?;
            *item = f32::from_le_bytes(bytes);
        }

        let matrix = Array2::from_shape_vec(shape, data).map_err(|_| {
            log::error!("Failed to create matrix: shape mismatch");
            IOError::ShapeMismatch
        })?;

        log::info!("Matrix successfully loaded from file: {file_path}");

        Ok(matrix)
    }

    fn parse_header(header: &str) -> Result<((usize, usize), bool), Error> {
        let regex = Regex::new(
            r"'descr'\s*:\s*'<f4'\s*,\s*'fortran_order'\s*:\s*(True|False)\s*,\s*'shape'\s*:\s*\((\d+)\s*,\s*(\d+)\)",
        )
        .map_err(|_| {
            log::error!("Failed to parse header");
            IOError::HeaderParseError
        })?;

        let caps = regex.captures(header).ok_or_else(|| {
            log::error!("Invalid format");
            IOError::InvalidFormat
        })?;

        let fortran_order = &caps[1] == "True";
        let dim1: usize = caps[2].parse().map_err(|_| {
            log::error!("Failed to parse dimension");
            IOError::InvalidFormat
        })?;
        let dim2: usize = caps[3].parse().map_err(|_| {
            log::error!("Failed to parse dimension");
            IOError::InvalidFormat
        })?;

        Ok(((dim1, dim2), fortran_order))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::fs::remove_file;
    use std::io::Write;
    use std::path::Path;

    #[test]
    fn test_save_to_safetensors_success() {
        let file_path = "test_matrix_save.safetensors";

        let matrix = Array2::from_elem((2, 3), 1.0_f32);
        let result = MatrixIO::save_to_safetensors(&matrix, file_path);

        let metadata = fs::metadata(file_path).expect("Failed to get file information");

        assert!(result.is_ok(), "Failed to save matrix to safetensors: {:?}", result);
        assert!(Path::new(file_path).exists(), "File was not created at {}", file_path);
        assert!(metadata.len() > 0, "File is empty");

        remove_file(file_path).expect("Failed to delete test file");
    }

    #[test]
    fn test_save_to_safetensors_invalid_path() {
        let invalid_path = "non_existent_directory/test_matrix_save.safetensors";

        let matrix = Array2::from_elem((2, 3), 1.0_f32);
        let result = MatrixIO::save_to_safetensors(&matrix, invalid_path);

        assert!(result.is_err(), "Expected error for invalid path, but got Ok");
    }

    #[test]
    fn test_save_to_safetensors_empty_matrix() {
        let file_path = "test_matrix_save_empty.safetensors";

        let matrix = Array2::from_elem((0, 0), 0.0_f32);
        let result = MatrixIO::save_to_safetensors(&matrix, file_path);

        assert!(result.is_ok(), "Failed to save empty matrix: {:?}", result);
        assert!(Path::new(file_path).exists(), "File was not created at {}", file_path);

        remove_file(file_path).expect("Failed to delete test file");
    }

    #[test]
    fn test_load_from_safetensors_success() {
        let file_path = "test_matrix_load.safetensors";

        let matrix = Array2::from_elem((2, 3), 1.0_f32);

        MatrixIO::save_to_safetensors(&matrix, file_path).expect("Failed to save matrix");

        let result = MatrixIO::load_from_safetensors(file_path);

        assert!(result.is_ok(), "Failed to load matrix from safetensors: {:?}", result);

        let loaded_matrix = result.expect("Failed to load matrix");

        assert_eq!(loaded_matrix.shape(), matrix.shape(), "Matrix shapes do not match");
        assert_eq!(loaded_matrix, matrix, "Matrix contents do not match");

        remove_file(file_path).expect("Failed to delete test file");
    }

    #[test]
    fn test_load_from_safetensors_file_not_found() {
        let invalid_path = "non_existent_directory/test_matrix_load.safetensors";

        let result = MatrixIO::load_from_safetensors(invalid_path);

        assert!(result.is_err(), "Expected error for nonexistent file, but got Ok");
    }

    #[test]
    fn test_load_from_safetensors_invalid_file() {
        let file_path = "test_matrix_load_invalid.safetensors";

        let mut file = File::create(file_path).expect("Failed to create test file");
        file.write_all(b"invalid data").expect("Failed to write test data");

        let result = MatrixIO::load_from_safetensors(file_path);

        assert!(result.is_err(), "Expected error for invalid data type, but got Ok");

        remove_file(file_path).expect("Failed to delete test file");
    }

    #[test]
    fn test_save_to_npy_success() {
        let file_path = "test_matrix_save.npy";

        let matrix = Array2::from_elem((2, 3), 1.0_f32);
        let result = MatrixIO::save_to_npy(&matrix, file_path);

        let metadata = fs::metadata(file_path).expect("Failed to get file information");

        assert!(result.is_ok(), "Failed to save matrix to npy: {:?}", result);
        assert!(Path::new(file_path).exists(), "File was not created at {}", file_path);
        assert!(metadata.len() > 0, "File is empty");

        remove_file(file_path).expect("Failed to delete test file");
    }

    #[test]
    fn test_save_to_npy_invalid_path() {
        let invalid_path = "non_existent_directory/test_matrix_save.npy";

        let matrix = Array2::from_elem((2, 3), 1.0_f32);
        let result = MatrixIO::save_to_npy(&matrix, invalid_path);

        assert!(result.is_err(), "Expected error for invalid path, but got Ok");
    }

    #[test]
    fn test_save_to_npy_empty_matrix() {
        let file_path = "test_matrix_save_empty.npy";

        let matrix = Array2::from_elem((0, 0), 0.0_f32);
        let result = MatrixIO::save_to_npy(&matrix, file_path);

        assert!(result.is_ok(), "Failed to save empty matrix: {:?}", result);
        assert!(Path::new(file_path).exists(), "File was not created at {}", file_path);

        remove_file(file_path).expect("Failed to delete test file");
    }

    #[test]
    fn test_load_from_npy_success() {
        let file_path = "test_matrix_load.npy";

        let matrix = Array2::from_elem((2, 3), 1.0_f32);

        MatrixIO::save_to_npy(&matrix, file_path).expect("Failed to save matrix");

        let result = MatrixIO::load_from_npy(file_path);

        assert!(result.is_ok(), "Failed to load matrix from npy: {:?}", result);

        let loaded_matrix = result.expect("Failed to load matrix");

        assert_eq!(loaded_matrix.shape(), matrix.shape(), "Matrix shapes do not match");
        assert_eq!(loaded_matrix, matrix, "Matrix contents do not match");

        remove_file(file_path).expect("Failed to delete test file");
    }

    #[test]
    fn test_load_from_npy_file_not_found() {
        let invalid_path = "non_existent_directory/test_matrix_load.npy";

        let result = MatrixIO::load_from_npy(invalid_path);

        assert!(result.is_err(), "Expected error for nonexistent file, but got Ok");
    }

    #[test]
    fn test_load_from_npy_invalid_file() {
        let file_path = "test_matrix_load_invalid.npy";

        let mut file = File::create(file_path).expect("Failed to create test file");
        file.write_all(b"invalid data").expect("Failed to write test data");

        let result = MatrixIO::load_from_npy(file_path);

        assert!(result.is_err(), "Expected error for invalid data type, but got Ok");

        remove_file(file_path).expect("Failed to delete test file");
    }
}
