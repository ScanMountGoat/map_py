use crate::MapPy;
use pyo3::{PyClass, exceptions::PyTypeError, prelude::*};

pub fn from_py<T, U>(value: Py<T>, py: Python) -> PyResult<U>
where
    T: MapPy<U> + PyClass + Clone,
{
    value.extract::<T>(py)?.map_py(py)
}

pub fn into_py<T, U>(value: T, py: Python) -> PyResult<Py<U>>
where
    T: MapPy<U>,
    U: PyClass + Into<PyClassInitializer<U>>,
{
    let value: U = value.map_py(py)?;
    Py::new(py, value)
}

pub fn from_option_py<T, U>(value: Option<Py<T>>, py: Python) -> PyResult<Option<U>>
where
    T: MapPy<U> + PyClass + Clone,
{
    value.map(|v| from_py(v, py)).transpose()
}

pub fn into_option_py<T, U>(value: Option<T>, py: Python) -> PyResult<Option<Py<U>>>
where
    T: MapPy<U>,
    U: PyClass + Into<PyClassInitializer<U>>,
{
    value.map(|v| into_py(v, py)).transpose()
}

pub fn into<T, U>(value: T, _py: Python) -> PyResult<U>
where
    T: Into<U>,
{
    Ok(value.into())
}

pub fn try_into<T, U>(value: T, _py: Python) -> PyResult<U>
where
    T: TryInto<U>,
    <T as TryInto<U>>::Error: std::fmt::Debug,
{
    value
        .try_into()
        .map_err(|e| PyErr::new::<PyTypeError, _>(format!("{e:?}")))
}
