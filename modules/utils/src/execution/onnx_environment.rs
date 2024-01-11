//! This module defines the ONNX environment for the execution of ONNX models.
use once_cell::sync::Lazy;
use ort::{Environment, ExecutionProvider};
use std::sync::Arc;

// the ONNX environment which loads the library
pub static ENVIRONMENT: Lazy<Arc<Environment>> = Lazy::new(|| {
	return Arc::new(
		Environment::builder()
			.with_execution_providers([ExecutionProvider::CPU(Default::default())])
			.build()
			.unwrap(),
	);
});
