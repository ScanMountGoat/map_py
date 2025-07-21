# map_py
Convert between Python and Rust types.

## Overview
This project defines a generic conversion trait `MapPy<T>` for fallible conversions between types that may require access to the Python interpreter. 
The trait is defined for many standard library types as well as glam types and numpy arrays.

```rust
pub trait MapPy<T> {
    fn map_py(self, py: Python) -> PyResult<T>;
}
```

Deriving `MapPy` makes it easy to convert Rust types in another project to idiomatic Python classes.
For example, the Python class can use numpy arrays instead of `Vec<glam::Mat4>` to reduce conversion overhead or `TypedList<T>` for converting `Vec<T>` to and from a normal Python list.

```rust
// pyo3 bindings
#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, MapPy)]
#[map(rust_project::Skeleton)]
pub struct Skeleton {
    pub transforms: Py<PyArray3<f32>>,
    pub names: TypedList<CollisionMesh>,
    pub parent_indices: Py<PyArray1<i32>>
}
```

```rust
// rust project
pub struct Skeleton {
    pub transforms: Vec<glam::Mat4>,
    pub names: Vec<String>,
    pub parent_indices: Vec<i32>
}
```

## Usage
This project is still experimental. Specify the commit hash for the revision in the Cargo.toml file to avoid breaking changes.

```toml
map_py = { git = "https://github.com/ScanMountGoat/map_py", rev = "..." }
```
