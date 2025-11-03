use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use indexmap::IndexMap;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyUntypedArray, ToPyArray};
use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};
use smol_str::SmolStr;

pub use map_py_derive::MapPy;
use std::{
    collections::{BTreeMap, HashMap},
    marker::PhantomData,
};

pub mod helpers;

// Define a mapping between types.
// This allows for deriving the Python <-> Rust conversion.
// The derive macro is mainly to automate mapping field names.
pub trait MapPy<T> {
    fn map_py(self, py: Python) -> PyResult<T>;
}

/// A statically typed [Vec] represented as un untyped Python list.
#[derive(Debug, Clone)]
pub struct TypedList<T> {
    pub list: Py<PyList>,
    _phantom: PhantomData<T>,
}

impl<T> TypedList<T> {
    pub fn empty(py: Python) -> Self {
        Self {
            list: PyList::empty(py).into(),
            _phantom: PhantomData,
        }
    }
}

impl<'py, T> IntoPyObject<'py> for TypedList<T> {
    type Target = PyList;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.list.into_bound(py))
    }
}

impl<T> FromPyObject<'_, '_> for TypedList<T> {
    type Error = PyErr;

    fn extract(ob: Borrowed<PyAny>) -> PyResult<Self> {
        Ok(Self {
            list: ob.extract()?,
            _phantom: PhantomData,
        })
    }
}

/// A statically typed [BTreeMap], [HashMap], or [IndexMap] represented as un untyped Python list.
#[derive(Debug, Clone)]
pub struct TypedDict<K, V> {
    pub dict: Py<PyDict>,
    _phantom: PhantomData<(K, V)>,
}

impl<K, V> TypedDict<K, V> {
    pub fn empty(py: Python) -> Self {
        Self {
            dict: PyDict::new(py).into(),
            _phantom: PhantomData,
        }
    }
}

impl<'py, K, V> IntoPyObject<'py> for TypedDict<K, V> {
    type Target = PyDict;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.dict.into_bound(py))
    }
}

impl<K, V> FromPyObject<'_, '_> for TypedDict<K, V> {
    type Error = PyErr;

    fn extract(ob: Borrowed<PyAny>) -> PyResult<Self> {
        Ok(Self {
            dict: ob.extract()?,
            _phantom: PhantomData,
        })
    }
}

// TODO: can this be a blanket impl?
// Implement for primitive types.

macro_rules! map_py_impl {
    ($($t:ty),*) => {
        $(
            impl MapPy<$t> for $t {
                fn map_py(self, _py: Python) -> PyResult<$t> {
                    Ok(self)
                }
            }
        )*
    }
}

map_py_impl!(
    char,
    bool,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    i8,
    i16,
    i32,
    i64,
    f32,
    f64,
    String,
    SmolStr,
    (u16, u16)
);

#[macro_export]
macro_rules! map_py_into_impl {
    ($t:ty,$u:ty) => {
        impl MapPy<$u> for $t {
            fn map_py(self, _py: Python) -> PyResult<$u> {
                Ok(self.into())
            }
        }

        impl MapPy<$t> for $u {
            fn map_py(self, _py: Python) -> PyResult<$t> {
                Ok(self.into())
            }
        }
    };
}
map_py_into_impl!(Vec2, [f32; 2]);
map_py_into_impl!(Vec3, [f32; 3]);
map_py_into_impl!(Vec4, [f32; 4]);

impl MapPy<Quat> for [f32; 4] {
    fn map_py(self, _py: Python) -> PyResult<Quat> {
        Ok(Quat::from_array(self))
    }
}

impl MapPy<[f32; 4]> for Quat {
    fn map_py(self, _py: Python) -> PyResult<[f32; 4]> {
        Ok(self.to_array())
    }
}

#[macro_export]
macro_rules! map_py_pyobject_ndarray_impl {
    ($($t:ty),*) => {
        $(
            // 1D arrays
            impl MapPy<Py<PyArray1<$t>>> for Vec<$t> {
                fn map_py(self, py: Python) -> PyResult<Py<PyArray1<$t>>> {
                    Ok(self.to_pyarray(py).into())
                }
            }

            impl MapPy<Vec<$t>> for Py<PyArray1<$t>> {
                fn map_py(self, py: Python) -> PyResult<Vec<$t>> {
                    let array = self.cast_bound::<PyArray1<$t>>(py)?;
                    Ok(array.readonly().as_slice()?.to_vec())
                }
            }

            // 1D untyped arrays
            impl MapPy<Py<PyUntypedArray>> for Vec<$t> {
                fn map_py(self, py: Python) -> PyResult<Py<PyUntypedArray>> {
                    let arr: Py<PyArray1<$t>> = self.map_py(py)?;
                    Ok(arr.bind(py).as_untyped().clone().unbind())
                }
            }

            impl MapPy<Vec<$t>> for Py<PyUntypedArray> {
                fn map_py(self, py: Python) -> PyResult<Vec<$t>> {
                    let arr = self.bind(py).cast::<PyArray1<$t>>()?;
                    arr.as_unbound().clone().map_py(py)
                }
            }

            // 2D arrays
            impl<const N: usize> MapPy<Py<PyArray2<$t>>> for Vec<[$t; N]> {
                fn map_py(self, py: Python) -> PyResult<Py<PyArray2<$t>>> {
                    // This flatten will be optimized in Release mode.
                    // This avoids needing unsafe code.
                    let count = self.len();
                    Ok(self
                        .iter()
                        .flatten()
                        .copied()
                        .collect::<Vec<$t>>()
                        .into_pyarray(py)
                        .reshape((count, N))
                        .unwrap()
                        .into())
                }
            }

            impl<const N: usize> MapPy<Vec<[$t; N]>> for Py<PyArray2<$t>> {
                fn map_py(self, py: Python) -> PyResult<Vec<[$t; N]>> {
                    let array = self.cast_bound::<PyArray2<$t>>(py)?;
                    Ok(array
                        .readonly()
                        .as_array()
                        .rows()
                        .into_iter()
                        .map(|r| r.as_slice().unwrap().try_into().unwrap())
                        .collect())
                }
            }

            // 2D untyped arrrays
            impl<const N: usize> MapPy<Py<PyUntypedArray>> for Vec<[$t; N]> {
                fn map_py(self, py: Python) -> PyResult<Py<PyUntypedArray>> {
                    let arr: Py<PyArray2<$t>> = self.map_py(py)?;
                    Ok(arr.bind(py).as_untyped().clone().unbind())
                }
            }

            impl<const N: usize> MapPy<Vec<[$t; N]>> for Py<PyUntypedArray> {
                fn map_py(self, py: Python) -> PyResult<Vec<[$t; N]>> {
                    let arr = self.bind(py).cast::<PyArray2<$t>>()?;
                    arr.as_unbound().clone().map_py(py)
                }
            }
        )*
    }
}

map_py_pyobject_ndarray_impl!(u8, u16, u32, u64, i8, i16, i32, i64, f32);

impl<T, U> MapPy<Option<U>> for Option<T>
where
    T: MapPy<U>,
{
    fn map_py(self, py: Python) -> PyResult<Option<U>> {
        self.map(|v| v.map_py(py)).transpose()
    }
}

impl<T, U> MapPy<Vec<U>> for Vec<T>
where
    T: MapPy<U>,
{
    fn map_py(self, py: Python) -> PyResult<Vec<U>> {
        self.into_iter().map(|v| v.map_py(py)).collect()
    }
}

pub fn map_list<T, U>(list: Py<PyList>, py: Python) -> PyResult<Vec<U>>
where
    for<'a, 'py> Vec<T>: FromPyObject<'a, 'py>,
    T: MapPy<U>,
{
    list.extract::<'_, '_, Vec<T>>(py)
        .map_err(Into::into)?
        .map_py(py)
}

pub fn map_vec<T, U>(value: Vec<T>, py: Python) -> PyResult<Py<PyList>>
where
    T: MapPy<U>,
    for<'a> U: IntoPyObject<'a>,
    for<'a> <U as IntoPyObject<'a>>::Output: IntoPyObject<'a>,
    for<'a> <U as IntoPyObject<'a>>::Error: std::fmt::Debug,
{
    PyList::new(
        py,
        value
            .into_iter()
            .map(|v| {
                let u: U = v.map_py(py)?;
                // TODO: avoid unwrap.
                Ok(u.into_pyobject(py).unwrap())
            })
            .collect::<PyResult<Vec<_>>>()?,
    )
    .map(Into::into)
}

impl<T, U> MapPy<Vec<U>> for TypedList<T>
where
    T: MapPy<U>,
    for<'a, 'py> Vec<T>: FromPyObject<'a, 'py>,
{
    fn map_py(self, py: Python) -> PyResult<Vec<U>> {
        map_list::<T, U>(self.list, py)
    }
}

impl<T, U> MapPy<TypedList<U>> for Vec<T>
where
    T: MapPy<U>,
    for<'a> U: IntoPyObject<'a>,
    for<'a> <U as IntoPyObject<'a>>::Output: IntoPyObject<'a>,
    for<'a> <U as IntoPyObject<'a>>::Error: std::fmt::Debug,
{
    fn map_py(self, py: Python) -> PyResult<TypedList<U>> {
        Ok(TypedList {
            list: map_vec::<T, U>(self, py)?,
            _phantom: PhantomData,
        })
    }
}

impl<K, V, K2, V2> MapPy<BTreeMap<K2, V2>> for TypedDict<K, V>
where
    K: MapPy<K2>,
    V: MapPy<V2>,
    for<'a, 'py> BTreeMap<K, V>: FromPyObject<'a, 'py>,
    K2: Ord + Eq,
{
    fn map_py(self, py: Python) -> PyResult<BTreeMap<K2, V2>> {
        self.dict
            .extract::<BTreeMap<K, V>>(py)
            .map_err(Into::into)?
            .into_iter()
            .map(|(k, v)| {
                let k2 = k.map_py(py)?;
                let v2 = v.map_py(py)?;
                Ok((k2, v2))
            })
            .collect()
    }
}

impl<K, V, K2, V2> MapPy<TypedDict<K2, V2>> for BTreeMap<K, V>
where
    K: MapPy<K2>,
    V: MapPy<V2>,
    for<'a> K2: IntoPyObject<'a>,
    for<'a> V2: IntoPyObject<'a>,
{
    fn map_py(self, py: Python) -> PyResult<TypedDict<K2, V2>> {
        let dict = PyDict::new(py);
        for (k, v) in self.into_iter() {
            let k2: K2 = k.map_py(py)?;
            let v2: V2 = v.map_py(py)?;
            dict.set_item(k2, v2)?;
        }

        Ok(TypedDict {
            dict: dict.into(),
            _phantom: PhantomData,
        })
    }
}

impl<K, V, K2, V2> MapPy<HashMap<K2, V2>> for TypedDict<K, V>
where
    K: MapPy<K2>,
    V: MapPy<V2>,
    for<'a, 'py> HashMap<K, V>: FromPyObject<'a, 'py>,
    K2: std::hash::Hash + Eq,
{
    fn map_py(self, py: Python) -> PyResult<HashMap<K2, V2>> {
        self.dict
            .extract::<HashMap<K, V>>(py)
            .map_err(Into::into)?
            .into_iter()
            .map(|(k, v)| {
                let k2 = k.map_py(py)?;
                let v2 = v.map_py(py)?;
                Ok((k2, v2))
            })
            .collect()
    }
}

impl<K, V, K2, V2, S> MapPy<TypedDict<K2, V2>> for HashMap<K, V, S>
where
    K: MapPy<K2>,
    V: MapPy<V2>,
    for<'a> K2: IntoPyObject<'a>,
    for<'a> V2: IntoPyObject<'a>,
{
    fn map_py(self, py: Python) -> PyResult<TypedDict<K2, V2>> {
        let dict = PyDict::new(py);
        for (k, v) in self.into_iter() {
            let k2: K2 = k.map_py(py)?;
            let v2: V2 = v.map_py(py)?;
            dict.set_item(k2, v2)?;
        }

        Ok(TypedDict {
            dict: dict.into(),
            _phantom: PhantomData,
        })
    }
}

impl<K, V, K2, V2, S2> MapPy<IndexMap<K2, V2, S2>> for TypedDict<K, V>
where
    K: MapPy<K2>,
    V: MapPy<V2>,
    for<'a, 'py> IndexMap<K, V>: FromPyObject<'a, 'py>,
    K2: std::hash::Hash + Eq,
    S2: std::hash::BuildHasher + Default,
{
    fn map_py(self, py: Python) -> PyResult<IndexMap<K2, V2, S2>> {
        self.dict
            .extract::<IndexMap<K, V>>(py)
            .map_err(Into::into)?
            .into_iter()
            .map(|(k, v)| {
                let k2 = k.map_py(py)?;
                let v2 = v.map_py(py)?;
                Ok((k2, v2))
            })
            .collect()
    }
}

impl<K, V, K2, V2, S> MapPy<TypedDict<K2, V2>> for IndexMap<K, V, S>
where
    K: MapPy<K2>,
    V: MapPy<V2>,
    for<'a> K2: IntoPyObject<'a>,
    for<'a> V2: IntoPyObject<'a>,
{
    fn map_py(self, py: Python) -> PyResult<TypedDict<K2, V2>> {
        let dict = PyDict::new(py);
        for (k, v) in self.into_iter() {
            let k2: K2 = k.map_py(py)?;
            let v2: V2 = v.map_py(py)?;
            dict.set_item(k2, v2)?;
        }

        Ok(TypedDict {
            dict: dict.into(),
            _phantom: PhantomData,
        })
    }
}

macro_rules! map_py_vecn_ndarray_impl {
    ($t:ty,$n:expr) => {
        impl MapPy<Py<PyArray2<f32>>> for Vec<$t> {
            fn map_py(self, py: Python) -> PyResult<Py<PyArray2<f32>>> {
                // This flatten will be optimized in Release mode.
                // This avoids needing unsafe code.
                // TODO: Double check this optimization.
                // TODO: faster to use bytemuck?
                let count = self.len();
                Ok(self
                    .into_iter()
                    .flat_map(|v| v.to_array())
                    .collect::<Vec<f32>>()
                    .into_pyarray(py)
                    .reshape((count, $n))
                    .unwrap()
                    .into())
            }
        }

        impl MapPy<Vec<$t>> for Py<PyArray2<f32>> {
            fn map_py(self, py: Python) -> PyResult<Vec<$t>> {
                let array = self.cast_bound::<PyArray2<f32>>(py)?;
                Ok(array
                    .readonly()
                    .as_array()
                    .rows()
                    .into_iter()
                    .map(|r| <$t>::from_slice(r.as_slice().unwrap()))
                    .collect())
            }
        }

        impl MapPy<Py<PyUntypedArray>> for Vec<$t> {
            fn map_py(self, py: Python) -> PyResult<Py<PyUntypedArray>> {
                let arr: Py<PyArray2<f32>> = self.map_py(py)?;
                Ok(arr.bind(py).as_untyped().clone().unbind())
            }
        }

        impl MapPy<Vec<$t>> for Py<PyUntypedArray> {
            fn map_py(self, py: Python) -> PyResult<Vec<$t>> {
                let arr = self.bind(py).cast::<PyArray2<f32>>()?;
                arr.as_unbound().clone().map_py(py)
            }
        }
    };
}
map_py_vecn_ndarray_impl!(Vec2, 2);
map_py_vecn_ndarray_impl!(Vec3, 3);
map_py_vecn_ndarray_impl!(Vec4, 4);
map_py_vecn_ndarray_impl!(Quat, 4);

impl MapPy<Py<PyArray2<f32>>> for Mat4 {
    fn map_py(self, py: Python) -> PyResult<Py<PyArray2<f32>>> {
        // Transpose since numpy is row-major.
        Ok(self
            .transpose()
            .to_cols_array()
            .to_pyarray(py)
            .readwrite()
            .reshape((4, 4))
            .unwrap()
            .into())
    }
}

impl MapPy<Mat4> for Py<PyArray2<f32>> {
    fn map_py(self, py: Python) -> PyResult<Mat4> {
        // Transpose since numpy is row-major.
        let array = self.cast_bound::<PyArray2<f32>>(py)?;
        Ok(Mat4::from_cols_slice(array.readonly().as_array().as_slice().unwrap()).transpose())
    }
}

impl MapPy<[[f32; 4]; 4]> for Py<PyArray2<f32>> {
    fn map_py(self, py: Python) -> PyResult<[[f32; 4]; 4]> {
        self.extract::<[[f32; 4]; 4]>(py)
    }
}

impl MapPy<Py<PyArray2<f32>>> for [[f32; 4]; 4] {
    fn map_py(self, py: Python) -> PyResult<Py<PyArray2<f32>>> {
        Ok(self
            .as_flattened()
            .to_pyarray(py)
            .reshape((4, 4))
            .unwrap()
            .into())
    }
}
impl MapPy<Py<PyArray3<f32>>> for Vec<Mat4> {
    fn map_py(self, py: Python) -> PyResult<Py<PyArray3<f32>>> {
        // This flatten will be optimized in Release mode.
        // Transpose since numpy is row-major.
        // TODO: transpose?
        let count = self.len();
        Ok(self
            .iter()
            .flat_map(|v| v.transpose().to_cols_array())
            .collect::<Vec<f32>>()
            .into_pyarray(py)
            .reshape((count, 4, 4))
            .unwrap()
            .into())
    }
}

impl MapPy<Vec<Mat4>> for Py<PyArray3<f32>> {
    fn map_py(self, py: Python) -> PyResult<Vec<Mat4>> {
        // Transpose since numpy is row-major.
        let array = self.cast_bound::<PyArray3<f32>>(py)?;
        let array = array.readonly();
        let array = array.as_array();
        Ok(array
            .into_shape_with_order((array.shape()[0], 16))
            .unwrap()
            .rows()
            .into_iter()
            .map(|r| Mat4::from_cols_slice(r.as_slice().unwrap()).transpose())
            .collect())
    }
}

// TODO: blanket impl using From/Into?
impl MapPy<SmolStr> for String {
    fn map_py(self, _py: Python) -> PyResult<SmolStr> {
        Ok(self.into())
    }
}

impl MapPy<String> for SmolStr {
    fn map_py(self, _py: Python) -> PyResult<String> {
        Ok(self.to_string())
    }
}

impl<T, U, const N: usize> MapPy<[U; N]> for [T; N]
where
    T: MapPy<U>,
{
    fn map_py(self, py: Python) -> PyResult<[U; N]> {
        // TODO: avoid unwrap
        Ok(self.map(|i| i.map_py(py).unwrap()))
    }
}
