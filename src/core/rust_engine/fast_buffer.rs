/*
fast_buffer.rs – Yüksek Performanslı Paylaşımlı Bellek (Shared Memory) Tamponu.

Bu Rust modülü, gelen canlı oran verilerini (odds stream) 
aşırı hızda işlemek için dairesel bir tampon (circular buffer) sağlar.
Python tarafı, FFI (Foreign Function Interface) üzerinden bu tampona erişir.
*/

use std::sync::{Arc, RwLock};

pub struct FastOddsBuffer {
    data: Arc<RwLock<Vec<f64>>>,
    size: usize,
    head: usize,
}

impl FastOddsBuffer {
    pub fn new(size: usize) -> Self {
        Self {
            data: Arc::new(RwLock::new(vec![0.0; size])),
            size,
            head: 0,
        }
    }

    pub fn push(&mut self, val: f64) {
        let mut lock = self.data.write().unwrap();
        lock[self.head] = val;
        self.head = (self.head + 1) % self.size;
    }

    pub fn get_avg(&self) -> f64 {
        let lock = self.data.read().unwrap();
        let sum: f64 = lock.iter().sum();
        sum / (self.size as f64)
    }

    pub fn get_max(&self) -> f64 {
        let lock = self.data.read().unwrap();
        *lock.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0)
    }
}

// FFI Köprüsü (C-compatible)
#[no_mangle]
pub extern "C" fn create_buffer(size: usize) -> *mut FastOddsBuffer {
    Box::into_raw(Box::new(FastOddsBuffer::new(size)))
}

#[no_mangle]
pub extern "C" fn push_odds(ptr: *mut FastOddsBuffer, val: f64) {
    let buffer = unsafe { &mut *ptr };
    buffer.push(val);
}

#[no_mangle]
pub extern "C" fn get_avg_odds(ptr: *mut FastOddsBuffer) -> f64 {
    let buffer = unsafe { &*ptr };
    buffer.get_avg()
}

#[no_mangle]
pub extern "C" fn destroy_buffer(ptr: *mut FastOddsBuffer) {
    if !ptr.is_null() {
        unsafe { Box::from_raw(ptr) };
    }
}
