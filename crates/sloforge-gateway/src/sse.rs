use bytes::Bytes;

#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum SseError {
    #[error("SSE frame exceeded {0} bytes")]
    FrameTooLarge(usize),
    #[error("SSE stream ended before a [DONE] event")]
    MissingDone,
    #[error("SSE data was not UTF-8")]
    InvalidUtf8,
    #[error("SSE data event was malformed JSON: {0}")]
    MalformedJson(String),
}

#[derive(Debug, PartialEq)]
pub enum SseItem {
    Data(serde_json::Value),
    Done,
}

/// Incremental bounded parser for OpenAI-compatible SSE. It accepts arbitrary transport chunking,
/// CRLF, comments, and multi-line data fields.
#[derive(Debug)]
pub struct SseParser {
    bytes: Vec<u8>,
    data_lines: Vec<String>,
    event_bytes: usize,
    max_bytes: usize,
    saw_done: bool,
}

impl SseParser {
    #[must_use]
    pub fn new(max_bytes: usize) -> Self {
        Self {
            bytes: Vec::new(),
            data_lines: Vec::new(),
            event_bytes: 0,
            max_bytes,
            saw_done: false,
        }
    }

    pub fn push(&mut self, chunk: &Bytes) -> Result<Vec<SseItem>, SseError> {
        let mut items = Vec::new();
        for byte in chunk {
            if *byte == b'\n' {
                let mut line = std::mem::take(&mut self.bytes);
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
                if let Some(item) = self.process_line(line)? {
                    items.push(item);
                }
            } else {
                if self.bytes.len() >= self.max_bytes {
                    return Err(SseError::FrameTooLarge(self.max_bytes));
                }
                self.bytes.push(*byte);
            }
        }
        Ok(items)
    }

    pub fn finish(&mut self) -> Result<Vec<SseItem>, SseError> {
        if !self.bytes.is_empty() {
            let line = std::mem::take(&mut self.bytes);
            let _ = self.process_line(line)?;
        }
        let mut items = Vec::new();
        if let Some(item) = self.finish_event()? {
            items.push(item);
        }
        if self.saw_done {
            Ok(items)
        } else {
            Err(SseError::MissingDone)
        }
    }

    fn finish_event(&mut self) -> Result<Option<SseItem>, SseError> {
        if self.data_lines.is_empty() {
            return Ok(None);
        }
        let data = std::mem::take(&mut self.data_lines).join("\n");
        self.event_bytes = 0;
        if data.trim() == "[DONE]" {
            self.saw_done = true;
            return Ok(Some(SseItem::Done));
        }
        serde_json::from_str(&data)
            .map(SseItem::Data)
            .map(Some)
            .map_err(|error| SseError::MalformedJson(error.to_string()))
    }

    fn push_data_line(&mut self, data: &str) -> Result<(), SseError> {
        self.event_bytes = self
            .event_bytes
            .saturating_add(data.len())
            .saturating_add(1);
        if self.event_bytes > self.max_bytes {
            return Err(SseError::FrameTooLarge(self.max_bytes));
        }
        self.data_lines.push(data.to_owned());
        Ok(())
    }

    fn process_line(&mut self, line: Vec<u8>) -> Result<Option<SseItem>, SseError> {
        let line = String::from_utf8(line).map_err(|_| SseError::InvalidUtf8)?;
        if line.is_empty() {
            self.finish_event()
        } else if line.starts_with(':') {
            Ok(None)
        } else if let Some(data) = line.strip_prefix("data:") {
            self.push_data_line(data.strip_prefix(' ').unwrap_or(data))?;
            Ok(None)
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_split_crlf_and_done() -> Result<(), SseError> {
        let mut parser = SseParser::new(1024);
        assert!(
            parser
                .push(&Bytes::from_static(b"data: {\"x\":"))?
                .is_empty()
        );
        let events = parser.push(&Bytes::from_static(b"1}\r\n\r\ndata: [DONE]\n\n"))?;
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], SseItem::Data(serde_json::json!({"x": 1})));
        assert_eq!(events[1], SseItem::Done);
        assert!(parser.finish()?.is_empty());
        Ok(())
    }

    #[test]
    fn rejects_unbounded_or_incomplete_streams() {
        let mut parser = SseParser::new(4);
        assert_eq!(
            parser.push(&Bytes::from_static(b"12345")),
            Err(SseError::FrameTooLarge(4))
        );
        let mut parser = SseParser::new(64);
        assert!(parser.push(&Bytes::from_static(b"data: {}\n\n")).is_ok());
        assert_eq!(parser.finish(), Err(SseError::MissingDone));
    }

    #[test]
    fn bounds_multiline_events_not_only_transport_lines() {
        let mut parser = SseParser::new(16);
        assert!(
            parser
                .push(&Bytes::from_static(b"data: 12345678\n"))
                .is_ok()
        );
        let result = parser.push(&Bytes::from_static(b"data: 12345678\n\n"));
        assert_eq!(result, Err(SseError::FrameTooLarge(16)));
    }

    #[test]
    fn transport_chunk_may_exceed_event_bound_when_events_do_not() -> Result<(), SseError> {
        let mut parser = SseParser::new(16);
        let events = parser.push(&Bytes::from_static(
            b"data: {\"x\":1}\n\ndata: {\"x\":2}\n\ndata: [DONE]\n\n",
        ))?;
        assert_eq!(events.len(), 3);
        Ok(())
    }
}
