//! Purpose-separated worker RNG streams from `reference/src/random/rng.c`.
use crate::config::{Real, RngAlgorithm};

#[repr(u64)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RngPurpose {
    Model = 1,
    Window = 2,
    Subsample = 3,
    Negative = 4,
}

fn mix_seed(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

pub fn derive_seed(root_seed: u64, worker_id: usize, purpose: RngPurpose) -> u64 {
    let worker_seed = mix_seed((worker_id as u64).wrapping_add(1));
    let purpose_seed = mix_seed(purpose as u64);
    mix_seed(root_seed ^ worker_seed ^ purpose_seed)
}

#[derive(Clone, Copy, Debug)]
pub struct Rng {
    pub state: u64,
    pub algorithm: RngAlgorithm,
}
impl Rng {
    pub fn new(seed: u64, algorithm: RngAlgorithm) -> Self {
        let state = if algorithm == RngAlgorithm::Xorshift && seed == 0 {
            0x6a09e667f3bcc909
        } else {
            seed
        };
        Self { state, algorithm }
    }

    pub fn next_u64(&mut self) -> u64 {
        if self.algorithm == RngAlgorithm::Lcg {
            self.state = self.state.wrapping_mul(25214903917).wrapping_add(11);
            self.state
        } else {
            let mut value = self.state;
            value ^= value >> 12;
            value ^= value << 25;
            value ^= value >> 27;
            self.state = value;
            value.wrapping_mul(2685821657736338717)
        }
    }

    pub fn uniform(&mut self) -> Real {
        if self.algorithm == RngAlgorithm::Lcg {
            (self.next_u64() & 0xffff) as Real / 65536.0
        } else {
            let bits = self.next_u64() >> 40;
            (bits as f64 * (1.0 / 16777216.0)) as Real
        }
    }
}
