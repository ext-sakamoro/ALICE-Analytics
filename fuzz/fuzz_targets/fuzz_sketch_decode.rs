//! Fuzz target: any bytes given to the sketches' `decode` return a sketch or
//! an error, never a panic, and a decoded sketch encodes back to the same bytes
//! (every stored field is read back; `DDSketch` derives `gamma` from `alpha`).

#![no_main]

use alice_analytics::sketch::{CountMinSketch1024x5, DDSketch256, HyperLogLog10};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(h) = HyperLogLog10::decode(data) {
        let mut out = vec![0u8; HyperLogLog10::ENCODED_LEN];
        h.encode(&mut out).unwrap();
        assert_eq!(out, data);
        let _ = h.cardinality();
    }
    if let Ok(d) = DDSketch256::decode(data) {
        let mut out = vec![0u8; DDSketch256::ENCODED_LEN];
        d.encode(&mut out).unwrap();
        assert_eq!(out, data);
        let _ = d.quantile(0.5);
    }
    if let Ok(c) = CountMinSketch1024x5::decode(data) {
        let mut out = vec![0u8; CountMinSketch1024x5::ENCODED_LEN];
        c.encode(&mut out).unwrap();
        assert_eq!(out, data);
        let _ = c.estimate_hash(7);
    }
});
