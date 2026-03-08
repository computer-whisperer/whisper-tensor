use crate::interfaces::AnyInterface;
use crate::model::Model;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// A named model produced by a loader.
pub struct LoadedModel {
    pub name: String,
    pub model: Arc<Model>,
}

/// A named interface produced by a loader.
pub struct LoadedInterface {
    pub name: String,
    pub interface: AnyInterface,
}

/// The output of running a loader: a set of named models and interfaces.
pub struct LoaderOutput {
    pub models: Vec<LoadedModel>,
    pub interfaces: Vec<LoadedInterface>,
}

/// Describes the type of a configuration field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigFieldType {
    /// A filesystem path.
    FilePath,
    /// A free-form string.
    String,
    /// An integer value with optional (min, max) bounds.
    Integer { min: Option<i64>, max: Option<i64> },
    /// A floating-point value with optional (min, max) bounds.
    Float { min: Option<f64>, max: Option<f64> },
    /// A boolean toggle.
    Bool,
    /// One of a fixed set of string options.
    Enum { options: Vec<String> },
}

/// Describes a single configuration field that a loader accepts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigField {
    /// Machine-readable key for this field.
    pub key: String,
    /// Human-readable label for the UI.
    pub label: String,
    /// Description / help text.
    pub description: String,
    /// The type and constraints of this field.
    pub field_type: ConfigFieldType,
    /// Whether this field must be provided.
    pub required: bool,
    /// Default value, if any.
    pub default: Option<ConfigValue>,
}

/// A concrete value for a configuration field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigValue {
    FilePath(PathBuf),
    String(String),
    Integer(i64),
    Float(f64),
    Bool(bool),
}

/// A set of configuration values keyed by field key.
pub type ConfigValues = HashMap<String, ConfigValue>;

/// Trait for model loaders.
///
/// A loader describes the configuration it needs (via `config_schema`),
/// and when given a valid configuration, produces a set of named models
/// and interfaces.
pub trait Loader: Send + Sync {
    /// Human-readable name of this loader (e.g. "Stable Diffusion 1.5").
    fn name(&self) -> &str;

    /// Short description of what this loader handles.
    fn description(&self) -> &str;

    /// Returns the configuration schema: what fields this loader needs.
    fn config_schema(&self) -> Vec<ConfigField>;

    /// Run the loader with the given configuration.
    fn load(&self, config: ConfigValues) -> Result<LoaderOutput, LoaderError>;
}

#[derive(Debug, thiserror::Error)]
pub enum LoaderError {
    #[error("Missing required config field: {0}")]
    MissingField(String),
    #[error("Invalid config value for field '{field}': {reason}")]
    InvalidValue { field: String, reason: String },
    #[error("Cannot identify model at path: {0}")]
    CannotIdentify(PathBuf),
    #[error("{0}")]
    LoadFailed(#[from] anyhow::Error),
}

/// Extract a required PathBuf value from a config map.
pub fn require_path(config: &ConfigValues, key: &str) -> Result<PathBuf, LoaderError> {
    match config.get(key) {
        Some(ConfigValue::FilePath(p)) => Ok(p.clone()),
        Some(_) => Err(LoaderError::InvalidValue {
            field: key.to_string(),
            reason: "expected a file path".to_string(),
        }),
        None => Err(LoaderError::MissingField(key.to_string())),
    }
}

/// Extract an optional string value from a config map.
pub fn get_string(config: &ConfigValues, key: &str) -> Result<Option<String>, LoaderError> {
    match config.get(key) {
        Some(ConfigValue::String(s)) => Ok(Some(s.clone())),
        Some(_) => Err(LoaderError::InvalidValue {
            field: key.to_string(),
            reason: "expected a string".to_string(),
        }),
        None => Ok(None),
    }
}

/// Extract an optional bool value from a config map.
pub fn get_bool(config: &ConfigValues, key: &str) -> Result<Option<bool>, LoaderError> {
    match config.get(key) {
        Some(ConfigValue::Bool(b)) => Ok(Some(*b)),
        Some(_) => Err(LoaderError::InvalidValue {
            field: key.to_string(),
            reason: "expected a boolean".to_string(),
        }),
        None => Ok(None),
    }
}
