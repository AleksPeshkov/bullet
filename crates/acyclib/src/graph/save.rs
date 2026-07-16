use std::{
    collections::HashMap,
    io::{self},
    rc::Rc,
};

use crate::{
    device::Device,
    graph::{Graph, Shape},
};

#[derive(Clone)]
pub struct TensorStore {
    pub values: Vec<f32>,
    pub shape: Shape,
}

impl TensorStore {
    #[deprecated(note = "You can access `.values` directly!")]
    pub fn get_dense_vals(&self) -> Option<Vec<f32>> {
        Some(self.values.clone())
    }
}

pub struct GraphWeights {
    stores: HashMap<String, TensorStore>,
}

impl<D: Device> From<&Graph<D>> for GraphWeights {
    fn from(graph: &Graph<D>) -> Self {
        let ids = graph.weight_ids();

        let mut stores = HashMap::new();

        for id in ids {
            let weight = graph.get_weights(&id);
            let values = weight.get_dense_vals().unwrap();
            let shape = weight.shape();
            let existing = stores.insert(id, TensorStore { values, shape });
            assert!(existing.is_none(), "Duplicate weight IDs in graph?!?");
        }

        Self { stores }
    }
}

impl GraphWeights {
    pub fn get(&self, id: &str) -> TensorStore {
        self.stores.get(id).cloned().unwrap()
    }

    #[deprecated(note = "Use `.get` instead!")]
    pub fn get_weights(&self, id: &str) -> TensorStore {
        self.get(id)
    }
}

type Transform = Rc<dyn Fn(&GraphWeights, Vec<f32>) -> Vec<f32>>;

#[derive(Clone)]
pub struct SavedFormat {
    custom: Option<Vec<u8>>,
    transforms: Vec<Transform>,
    id: Option<String>,

    // Atomic function: takes exactly ONE f32, returns raw bytes for ONE element
    pub element_serialiser: Rc<dyn Fn(f32) -> Vec<u8> + 'static>,
}

impl SavedFormat {
    /// Save a custom set of bytes.
    /// This should be used to add a network header, padding, etc.
    pub fn custom(bytes: impl Into<Vec<u8>>) -> Self {
        Self { custom: Some(bytes.into()), ..Self::empty() }
    }

    pub fn get_id(&self) -> Option<String> {
        self.id.clone()
    }

    /// Create a `SavedFormat` that is initialised with the weights from `id`.
    pub fn id(id: &str) -> Self {
        let id = id.to_string();
        Self { id: Some(id.clone()), ..Self::empty() }.transform(move |store, _| store.get(&id).values)
    }

    /// Create an empty `SavedFormat`
    pub fn empty() -> Self {
        SavedFormat {
            custom: None,
            id: None,
            transforms: Vec::new(),
            // Default strategy for f32: raw bitwise pass-through
            element_serialiser: Rc::new(|w| w.to_ne_bytes().to_vec()),
        }
    }

    #[deprecated(note = "Use `.transform(|store, mut values| { ... })` instead!")]
    pub fn round(self) -> Self {
        // does nothing
        self
    }

    pub fn rescale<T: 'static>(mut self, multiplier: impl Into<f64>) -> Self {
        assert!(self.custom.is_none());

        // precise scale into f64 space
        let scale_f64: f64 = multiplier.into();
        let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();

        self = self.transform(move |_, mut weights| {
            for i in 0..weights.len() {
                let scaled = weights[i] as f64 * scale_f64;
                weights[i] = if is_f32 { scaled as f32 } else { scaled.round() as f32 };
            }
            weights
        });

        self
    }

    pub fn quantise_to_type<T: 'static>(mut self) -> Self {
        assert!(self.custom.is_none());

        let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();

        if is_f32 {
            // float serializer strategy: raw 4 bytes copy
            self.element_serialiser = Rc::new(move |w| {
                return w.to_ne_bytes().to_vec();
            });
        } else {
            // universal integer serializer strategy
            let type_bytes = std::mem::size_of::<T>();
            self.element_serialiser = Rc::new(move |w| {
                let int_val = w as i64;
                let raw_bytes = int_val.to_ne_bytes();
                return raw_bytes[0..type_bytes].to_vec();
            });
        }

        self
    }

    pub fn quantise<T: 'static>(self, multiplier: impl Into<f64>) -> Self {
        self.rescale::<T>(multiplier).quantise_to_type::<T>()
    }

    /// Transpose current values using the shape of the weights from weight `id`.
    /// Panics if this `SavedFormat` was constructed without an associated weight `id`.
    pub fn transpose(self) -> Self {
        let id = self.id.clone().unwrap();
        self.transform(move |graph, weights| Self::transpose_impl(graph.get(&id).shape, &weights))
    }

    /// Transform current values using the provided closure.
    pub fn transform(mut self, f: impl Fn(&GraphWeights, Vec<f32>) -> Vec<f32> + 'static) -> Self {
        assert!(self.custom.is_none());
        self.transforms.push(Rc::new(f));
        self
    }

    #[deprecated(note = "Use `.transform(|store, mut values| { ... })` instead!")]
    pub fn add_transform(mut self, f: impl Fn(&GraphWeights, &str, Vec<f32>) -> Vec<f32> + 'static) -> Self {
        assert!(self.custom.is_none());
        let id = self.get_id().unwrap();
        self.transforms.push(Rc::new(move |store, vals| f(store, &id, vals)));
        self
    }

    pub fn write_to_byte_buffer(&self, graph: &GraphWeights) -> io::Result<Vec<u8>> {
        if let Some(bytes) = &self.custom {
            return Ok(bytes.clone());
        }

        let mut weights = Vec::new();

        for transform in &self.transforms {
            weights = transform(graph, weights);
        }

        if weights.is_empty() {
            return Ok(Vec::new());
        }

        let serialize = self.element_serialiser.as_ref();

        // process the VERY FIRST element to determine exact layout size
        let first_element = serialize(weights[0]);

        let mut buf = Vec::new();
        buf.reserve(weights.len() * first_element.len());
        buf.extend_from_slice(&first_element);

        for i in 1..weights.len() {
            buf.extend_from_slice(&serialize(weights[i]));
        }

        return Ok(buf);
    }

    pub(crate) fn transpose_impl(shape: Shape, weights: &[f32]) -> Vec<f32> {
        assert_eq!(shape.size(), weights.len());

        let rows = shape.rows();
        let cols = shape.cols();
        let mut new_buf = vec![0.0; shape.size()];

        for i in 0..rows {
            for j in 0..cols {
                new_buf[cols * i + j] = weights[rows * j + i];
            }
        }

        new_buf
    }
}
