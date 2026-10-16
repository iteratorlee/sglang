//! Incremental terminal-event detection for byte-preserving SSE forwarding.

/// Recognize an event whose complete data is `[DONE]`, including its blank
/// line terminator. Network chunks need not align with lines or events.
///
/// Only a short line prefix is retained: arbitrary JSON payloads are neither
/// buffered nor parsed. LF and CRLF are supported, including split CRLF pairs.
#[derive(Default)]
pub(crate) struct SseDoneDetector {
    // Longest matching line, excluding LF: "data: [DONE]\r".
    line: [u8; 13],
    // Saturates at line.len() + 1, distinguishing overflow from an exact match.
    line_len: usize,
    has_data: bool,
    data_is_done: bool,
}

impl SseDoneDetector {
    pub(crate) fn feed(&mut self, chunk: &[u8]) -> bool {
        for &byte in chunk {
            if byte != b'\n' {
                if self.line_len < self.line.len() {
                    self.line[self.line_len] = byte;
                }
                self.line_len = (self.line_len + 1).min(self.line.len() + 1);
                continue;
            }

            let overflow = self.line_len > self.line.len();
            let line = &self.line[..self.line_len.min(self.line.len())];
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            self.line_len = 0;

            if line.is_empty() && !overflow {
                let done = self.has_data && self.data_is_done;
                self.has_data = false;
                self.data_is_done = false;
                if done {
                    return true;
                }
            } else if line.starts_with(b"data:") || line == b"data" {
                // Multiple data fields join with '\n', so cannot equal [DONE].
                let data = line.strip_prefix(b"data:").unwrap_or(b"");
                let data = data.strip_prefix(b" ").unwrap_or(data);
                self.data_is_done = !self.has_data && !overflow && data == b"[DONE]";
                self.has_data = true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::SseDoneDetector;

    #[test]
    fn done_requires_complete_event_at_every_pair_of_chunk_boundaries() {
        for event in [
            &b"data: [DONE]\n\n"[..],
            &b"data:[DONE]\r\n\r\n"[..],
            &b"event: message\r\ndata: [DONE]\r\n: comment\r\n\r\n"[..],
        ] {
            for first in 0..event.len() {
                for second in first..event.len() {
                    let mut detector = SseDoneDetector::default();
                    assert!(!detector.feed(&event[..first]));
                    assert!(!detector.feed(&event[first..second]));
                    assert!(detector.feed(&event[second..]));
                }
            }
        }
    }

    #[test]
    fn bytewise_done_and_multiple_events() {
        let input =
            b"data: {\"text\":\"####4\",\"finish_reason\":\"stop\"}\n\ndata: [DONE]\r\n\r\n";
        let mut detector = SseDoneDetector::default();
        for (i, byte) in input.iter().enumerate() {
            assert_eq!(detector.feed(&[*byte]), i == input.len() - 1);
        }
        assert!(SseDoneDetector::default().feed(input));
    }

    #[test]
    fn literals_comments_and_nonterminal_data_do_not_finish() {
        for event in [
            &b"data: {\"text\":\"data: [DONE] and data:[DONE]\"}\n\n"[..],
            &b": data: [DONE]\n\nevent: data: [DONE]\n\n"[..],
            &b"data: [DONE]extra\n\n"[..],
            &b"data:  [DONE]\n\n"[..],
            &b"data: [DONE]\ndata: extra\n\n"[..],
            &b"data: [DONE]\ndata\n\n"[..],
            &b"data: earlier\ndata: [DONE]\n\n"[..],
        ] {
            for split in 0..=event.len() {
                let mut detector = SseDoneDetector::default();
                assert!(!detector.feed(&event[..split]));
                assert!(!detector.feed(&event[split..]));
                assert!(detector.feed(b"data: [DONE]\n\n"));
            }
        }
    }

    #[test]
    fn oversized_lines_do_not_match_or_accumulate() {
        let mut detector = SseDoneDetector::default();
        assert!(!detector.feed(b"data: [DONE]"));
        for _ in 0..1024 {
            assert!(!detector.feed(&[b'x'; 1024]));
        }
        assert_eq!(detector.line_len, detector.line.len() + 1);
        assert!(!detector.feed(b"\n\n"));
        assert!(detector.feed(b"data: [DONE]\n\n"));
    }
}
