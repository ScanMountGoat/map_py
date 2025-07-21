extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;

use syn::{Data, DataStruct, DeriveInput, Fields, parenthesized, parse_macro_input};

#[proc_macro_derive(MapPy, attributes(map))]
pub fn map_py_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    // ex: #[map(rust_project::RustType)]
    let map_type: syn::Path = input
        .attrs
        .iter()
        .find(|a| a.path().is_ident("map"))
        .map(|a| a.parse_args().unwrap())
        .expect("Must specify a map type");

    let name = &input.ident;

    // Assume both structs have identical field names.
    // This could be improved via skip and rename attributes in the future.
    let map_python_fields: Vec<_> = match &input.data {
        Data::Struct(DataStruct {
            fields: Fields::Named(fields),
            ..
        }) => fields
            .named
            .iter()
            .map(|field| {
                let name = field.ident.as_ref().unwrap();
                let options = FieldOptions::new(field);

                if let Some(map_from) = options.map_from {
                    quote!(#name: (#map_from)(self.#name, py)?)
                } else {
                    quote!(#name: self.#name.map_py(py)?)
                }
            })
            .collect(),
        _ => panic!("Unsupported type"),
    };

    let map_rust_fields: Vec<_> = match &input.data {
        Data::Struct(DataStruct {
            fields: Fields::Named(fields),
            ..
        }) => fields
            .named
            .iter()
            .map(|field| {
                let name = field.ident.as_ref().unwrap();
                let options = FieldOptions::new(field);

                if let Some(map_into) = options.map_into {
                    quote!(#name: (#map_into)(self.#name, py)?)
                } else {
                    quote!(#name: self.#name.map_py(py)?)
                }
            })
            .collect(),
        _ => panic!("Unsupported type"),
    };

    quote! {
        // Map from the implementing type to the map type.
        impl ::map_py::MapPy<#map_type> for #name {
            fn map_py(self, py: pyo3::Python) -> pyo3::prelude::PyResult<#map_type> {
                Ok(
                    #map_type {
                        #(#map_rust_fields),*
                    }
                )
            }
        }

        // Map from the map type to the implementing type.
        impl ::map_py::MapPy<#name> for #map_type {
            fn map_py(self, py: pyo3::Python) -> pyo3::prelude::PyResult<#name> {
                Ok(
                    #name {
                        #(#map_python_fields),*
                    }
                )
            }
        }
    }
    .into()
}

struct FieldOptions {
    map_from: Option<TokenStream2>,
    map_into: Option<TokenStream2>,
}

impl FieldOptions {
    fn new(field: &syn::Field) -> Self {
        let mut map_from = None;
        let mut map_into = None;
        for a in &field.attrs {
            if a.path().is_ident("map") {
                let _ = a.parse_nested_meta(|meta| {
                    if meta.path.is_ident("from") {
                        // #[map(from(map_from))]
                        let content;
                        parenthesized!(content in meta.input);
                        let lit: TokenStream2 = content.parse().unwrap();
                        map_from = Some(lit);
                    } else if meta.path.is_ident("into") {
                        // #[map(into(map_into))]
                        let content;
                        parenthesized!(content in meta.input);
                        let lit: TokenStream2 = content.parse().unwrap();
                        map_into = Some(lit);
                    }
                    Ok(())
                });
            }
        }

        Self { map_from, map_into }
    }
}
