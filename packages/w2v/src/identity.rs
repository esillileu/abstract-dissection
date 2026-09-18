//! Stable engine-local identity digests for checkpoint compatibility checks.

pub(crate) struct StableDigest(u64);

impl StableDigest {
    pub(crate) const fn new() -> Self {
        Self(0xcbf29ce484222325)
    }

    pub(crate) fn update(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 = (self.0 ^ u64::from(*byte)).wrapping_mul(0x100000001b3);
        }
    }

    pub(crate) fn value(&mut self, value: u64) {
        self.update(&value.to_le_bytes());
    }

    pub(crate) fn finish(self) -> String {
        format!("fnv1a64:{:016x}", self.0)
    }
}
