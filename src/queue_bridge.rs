/*
    ALICE-Analytics
    Copyright (C) 2026 Moroya Sakamoto

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

//! Bridge between ALICE-Queue message transport and ALICE-Analytics streaming aggregation.
//!
//! Converts Queue [`Message`] payloads into Analytics [`MetricEvent`]s and provides
//! [`QueueConsumerPipeline`] that combines dequeue + aggregation in one loop.
//!
//! # Payload Format
//!
//! ```text
//! [type:u8][name_hash:u64 LE][value:f64 LE] = 17 bytes per metric
//! ```
//!
//! # Architecture
//!
//! ```text
//! Sensor/App ──► ALICE-Queue (SPSC + WAL)
//!                      ↓  dequeue
//!               QueueConsumerPipeline
//!                      ↓  parse + submit
//!               MetricPipeline (HLL++, DDSketch, streaming)
//! ```

use crate::pipeline::{MetricEvent, MetricPipeline, MetricType};
use alice_queue::{AliceQueue, Message};

/// Payload size: 1 (type) + 8 (name_hash) + 8 (value) = 17 bytes.
pub const METRIC_PAYLOAD_SIZE: usize = 17;

/// Encode a [`MetricEvent`] into a compact 17-byte payload for Queue transport.
///
/// Format: `[metric_type:u8][name_hash:u64 LE][value:f64 LE]`
#[inline]
pub fn encode_metric_payload(event: &MetricEvent) -> [u8; METRIC_PAYLOAD_SIZE] {
    let mut buf = [0u8; METRIC_PAYLOAD_SIZE];
    buf[0] = match event.metric_type {
        MetricType::Counter => 0,
        MetricType::Gauge => 1,
        MetricType::Histogram => 2,
        MetricType::Unique => 3,
    };
    buf[1..9].copy_from_slice(&event.name_hash.to_le_bytes());
    buf[9..17].copy_from_slice(&event.value.to_le_bytes());
    buf
}

/// Decode a Queue [`Message`] payload into a [`MetricEvent`].
///
/// Returns `None` if the payload is too short or contains an invalid metric type.
#[inline]
pub fn parse_metric_event(msg: &Message) -> Option<MetricEvent> {
    let payload = &msg.payload;
    if payload.len() < METRIC_PAYLOAD_SIZE {
        return None;
    }

    let metric_type = match payload[0] {
        0 => MetricType::Counter,
        1 => MetricType::Gauge,
        2 => MetricType::Histogram,
        3 => MetricType::Unique,
        _ => return None,
    };

    let name_hash = u64::from_le_bytes([
        payload[1], payload[2], payload[3], payload[4],
        payload[5], payload[6], payload[7], payload[8],
    ]);
    let value = f64::from_le_bytes([
        payload[9], payload[10], payload[11], payload[12],
        payload[13], payload[14], payload[15], payload[16],
    ]);

    Some(MetricEvent {
        name_hash,
        metric_type,
        value,
        timestamp: 0,
    })
}

/// Combined Queue consumer + Analytics pipeline.
///
/// Drains messages from an [`AliceQueue`], parses metric payloads,
/// and feeds them into a [`MetricPipeline`] for streaming aggregation.
///
/// # Type Parameters
///
/// - `SLOTS`: Number of metric slots in the pipeline (power of 2 recommended)
/// - `PIPE_QUEUE`: Internal ring buffer size for the pipeline
/// - `Q`: Queue ring buffer capacity
pub struct QueueConsumerPipeline<const SLOTS: usize, const PIPE_QUEUE: usize, const Q: usize> {
    /// The message queue (SPSC + WAL)
    pub queue: AliceQueue<Q>,
    /// The streaming analytics pipeline
    pub pipeline: MetricPipeline<SLOTS, PIPE_QUEUE>,
    /// Total messages consumed
    consumed: u64,
    /// Messages with invalid/unparseable payloads
    parse_errors: u64,
}

impl<const SLOTS: usize, const PIPE_QUEUE: usize, const Q: usize>
    QueueConsumerPipeline<SLOTS, PIPE_QUEUE, Q>
{
    /// Create a new consumer pipeline.
    ///
    /// - `alpha`: DDSketch relative error parameter (e.g., 0.05 for 5%)
    pub fn new(alpha: f64) -> Self {
        Self {
            queue: AliceQueue::new(),
            pipeline: MetricPipeline::new(alpha),
            consumed: 0,
            parse_errors: 0,
        }
    }

    /// Drain all available messages from the queue into the pipeline.
    ///
    /// Returns the number of successfully processed messages.
    pub fn drain(&mut self) -> usize {
        let mut count = 0;
        while let Some((msg, _gap)) = self.queue.dequeue() {
            self.consumed += 1;
            if let Some(event) = parse_metric_event(&msg) {
                self.pipeline.submit(event);
                count += 1;
            } else {
                self.parse_errors += 1;
            }
        }
        if count > 0 {
            self.pipeline.flush();
        }
        count
    }

    /// Drain up to `max` messages from the queue.
    ///
    /// Useful for rate-limiting or cooperative scheduling.
    pub fn drain_batch(&mut self, max: usize) -> usize {
        let mut count = 0;
        for _ in 0..max {
            match self.queue.dequeue() {
                Some((msg, _gap)) => {
                    self.consumed += 1;
                    if let Some(event) = parse_metric_event(&msg) {
                        self.pipeline.submit(event);
                        count += 1;
                    } else {
                        self.parse_errors += 1;
                    }
                }
                None => break,
            }
        }
        if count > 0 {
            self.pipeline.flush();
        }
        count
    }

    /// Total messages consumed from the queue.
    #[inline]
    pub fn total_consumed(&self) -> u64 {
        self.consumed
    }

    /// Messages that failed to parse.
    #[inline]
    pub fn total_parse_errors(&self) -> u64 {
        self.parse_errors
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sketch::FnvHasher;

    fn test_sender() -> [u8; 32] {
        [0u8; 32]
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let events = [
            MetricEvent::counter(0xDEADBEEF, 42.0),
            MetricEvent::gauge(0xCAFEBABE, -3.14),
            MetricEvent::histogram(123456, 99.9),
            MetricEvent::unique(0xFF, 12345),
        ];

        for original in &events {
            let payload = encode_metric_payload(original);
            assert_eq!(payload.len(), METRIC_PAYLOAD_SIZE);

            let msg = Message::new(test_sender(), 1, payload.to_vec());
            let decoded = parse_metric_event(&msg).unwrap();

            assert_eq!(decoded.name_hash, original.name_hash);
            assert_eq!(decoded.metric_type, original.metric_type);
            assert_eq!(decoded.value, original.value);
        }
    }

    #[test]
    fn test_parse_invalid_payload() {
        // Too short
        let msg = Message::new(test_sender(), 1, vec![0, 1, 2]);
        assert!(parse_metric_event(&msg).is_none());

        // Invalid type byte
        let mut payload = [0u8; METRIC_PAYLOAD_SIZE];
        payload[0] = 255;
        let msg = Message::new(test_sender(), 1, payload.to_vec());
        assert!(parse_metric_event(&msg).is_none());
    }

    #[test]
    fn test_consumer_pipeline_drain() {
        let mut consumer = QueueConsumerPipeline::<64, 256, 128>::new(0.05);

        let req_hash = FnvHasher::hash_bytes(b"http.requests");

        // Enqueue 10 counter events
        for i in 0..10u64 {
            let event = MetricEvent::counter(req_hash, 1.0);
            let payload = encode_metric_payload(&event);
            let msg = Message::new(test_sender(), i + 1, payload.to_vec());
            consumer.queue.enqueue(msg).unwrap();
        }

        // Drain
        let processed = consumer.drain();
        assert_eq!(processed, 10);
        assert_eq!(consumer.total_consumed(), 10);
        assert_eq!(consumer.total_parse_errors(), 0);

        // Check pipeline
        let slot = consumer.pipeline.get_slot(req_hash).unwrap();
        assert_eq!(slot.counter, 10.0);
    }

    #[test]
    fn test_consumer_pipeline_mixed_metrics() {
        let mut consumer = QueueConsumerPipeline::<64, 256, 128>::new(0.05);

        let counter_hash = FnvHasher::hash_bytes(b"requests");
        let gauge_hash = FnvHasher::hash_bytes(b"cpu_usage");
        let hist_hash = FnvHasher::hash_bytes(b"latency");

        let mut seq = 1u64;

        // 5 counters
        for _ in 0..5 {
            let payload = encode_metric_payload(&MetricEvent::counter(counter_hash, 1.0));
            consumer.queue.enqueue(Message::new(test_sender(), seq, payload.to_vec())).unwrap();
            seq += 1;
        }

        // 3 gauge updates
        for v in [50.0, 60.0, 70.0] {
            let payload = encode_metric_payload(&MetricEvent::gauge(gauge_hash, v));
            consumer.queue.enqueue(Message::new(test_sender(), seq, payload.to_vec())).unwrap();
            seq += 1;
        }

        // 4 histogram observations
        for v in [10.0, 20.0, 30.0, 100.0] {
            let payload = encode_metric_payload(&MetricEvent::histogram(hist_hash, v));
            consumer.queue.enqueue(Message::new(test_sender(), seq, payload.to_vec())).unwrap();
            seq += 1;
        }

        let processed = consumer.drain();
        assert_eq!(processed, 12);

        // Verify counter
        let slot = consumer.pipeline.get_slot(counter_hash).unwrap();
        assert_eq!(slot.counter, 5.0);

        // Verify gauge (last value)
        let slot = consumer.pipeline.get_slot(gauge_hash).unwrap();
        assert_eq!(slot.gauge, 70.0);

        // Verify histogram
        let slot = consumer.pipeline.get_slot(hist_hash).unwrap();
        assert_eq!(slot.ddsketch.count(), 4);
    }

    #[test]
    fn test_drain_batch_limit() {
        let mut consumer = QueueConsumerPipeline::<64, 256, 128>::new(0.05);

        let hash = FnvHasher::hash_bytes(b"metric");

        // Enqueue 20 events
        for i in 0..20u64 {
            let payload = encode_metric_payload(&MetricEvent::counter(hash, 1.0));
            consumer.queue.enqueue(Message::new(test_sender(), i + 1, payload.to_vec())).unwrap();
        }

        // Drain only 5
        let processed = consumer.drain_batch(5);
        assert_eq!(processed, 5);
        assert_eq!(consumer.queue.len(), 15);

        // Drain remaining
        let processed = consumer.drain();
        assert_eq!(processed, 15);
        assert!(consumer.queue.is_empty());

        // Total counter should be 20
        let slot = consumer.pipeline.get_slot(hash).unwrap();
        assert_eq!(slot.counter, 20.0);
    }
}
