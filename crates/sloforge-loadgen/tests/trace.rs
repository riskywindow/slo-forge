#![allow(clippy::expect_used)]

use proptest::prelude::*;
use sloforge_loadgen::{
    ArrivalProcess, BurstWindow, ReplayConfig, ReplayStatus, RequestClass, TokenDistribution,
    WorkloadConfig, generate, read_trace, replay, summarize_trace, write_trace,
};
use std::io::Cursor;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

fn config(seed: u64, count: usize) -> WorkloadConfig {
    WorkloadConfig {
        schema_version: "1.0".into(),
        seed,
        request_count: count,
        max_duration_ms: None,
        arrival_process: ArrivalProcess::BurstyPoisson {
            base_rate_per_second: 10.0,
            bursts: vec![BurstWindow {
                start_ms: 1_000,
                duration_ms: 2_000,
                multiplier: 8.0,
            }],
        },
        classes: vec![
            RequestClass {
                name: "interactive".into(),
                weight: 0.8,
                prompt_tokens: TokenDistribution::Uniform { min: 8, max: 64 },
                output_tokens: TokenDistribution::Uniform { min: 4, max: 32 },
                priority: 0,
                deadline_ms: Some(250),
                canary_eligible: true,
                adapter_ids: vec!["code-lora".into()],
                prefix_groups: vec!["system-code".into()],
            },
            RequestClass {
                name: "long_context".into(),
                weight: 0.2,
                prompt_tokens: TokenDistribution::Empirical {
                    values: vec![2_048, 4_096],
                },
                output_tokens: TokenDistribution::Constant { value: 128 },
                priority: 2,
                deadline_ms: Some(5_000),
                canary_eligible: false,
                adapter_ids: Vec::new(),
                prefix_groups: Vec::new(),
            },
        ],
    }
}

#[test]
fn mixed_bursty_trace_is_deterministic_and_sorted() {
    let left = generate(&config(42, 1_000)).expect("trace");
    let right = generate(&config(42, 1_000)).expect("trace");
    assert_eq!(left, right);
    assert_eq!(left.len(), 1_000);
    assert!(
        left.windows(2)
            .all(|pair| pair[0].arrival_ms <= pair[1].arrival_ms)
    );
    assert!(
        left.iter()
            .any(|request| request.request_class == "interactive")
    );
    assert!(
        left.iter()
            .any(|request| request.request_class == "long_context")
    );
}

#[test]
fn jsonl_round_trip_and_summary_preserve_classes() {
    let records = generate(&config(8, 50)).expect("trace");
    let mut jsonl = Vec::new();
    write_trace(&mut jsonl, &records).expect("write");
    let parsed = read_trace(Cursor::new(jsonl)).expect("parse");
    let mut canonical_expected = records;
    for request in &mut canonical_expected {
        request.canary_eligible = true;
    }
    assert_eq!(canonical_expected, parsed);
    let summary = summarize_trace(&parsed);
    assert_eq!(summary.record_count, 50);
    assert_eq!(summary.priorities, vec![0, 2]);
}

#[test]
fn validator_rejects_unsorted_and_duplicate_records() {
    let mut records = generate(&config(9, 2)).expect("trace");
    records[0].arrival_ms = 100;
    records[1].arrival_ms = 10;
    let unsorted = records
        .iter()
        .map(|record| serde_json::to_string(record).expect("json"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(read_trace(Cursor::new(unsorted)).is_err());

    records[1].arrival_ms = 100;
    let duplicate_id = records[0].id.clone();
    records[1].id = duplicate_id;
    let duplicate = records
        .iter()
        .map(|record| serde_json::to_string(record).expect("json"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(read_trace(Cursor::new(duplicate)).is_err());
}

#[test]
fn fixed_interval_has_exact_expected_arrivals() {
    let mut input = config(1, 5);
    input.arrival_process = ArrivalProcess::FixedInterval { interval_ms: 2.5 };
    let trace = generate(&input).expect("trace");
    let arrivals: Vec<_> = trace.iter().map(|request| request.arrival_ms).collect();
    assert_eq!(arrivals, vec![0, 3, 5, 8, 10]);
}

#[tokio::test]
async fn replay_measures_streaming_ttft_from_real_http() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("address");
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.expect("accept");
        let mut request = vec![0_u8; 16 * 1024];
        let _bytes = stream.read(&mut request).await.expect("read request");
        let body = "data: {\"choices\":[{\"text\":\"x\"}]}\n\ndata: [DONE]\n\n";
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );
        stream
            .write_all(response.as_bytes())
            .await
            .expect("response");
    });
    let records = generate(&config(10, 1)).expect("trace");
    let measurements = replay(
        &records,
        &ReplayConfig {
            target: format!("http://{address}/v1/completions"),
            model: "mock".into(),
            time_scale: 100.0,
            max_concurrency: 1,
            request_timeout_ms: 1_000,
        },
    )
    .await
    .expect("replay");
    server.await.expect("server task");
    assert_eq!(measurements.len(), 1);
    assert_eq!(measurements[0].status, ReplayStatus::Completed);
    assert!(measurements[0].ttft_ms.is_some());
    assert_eq!(measurements[0].stream_events, 1);
}

#[tokio::test]
async fn replay_rejects_oversized_sse_frames() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("address");
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.expect("accept");
        let mut request = vec![0_u8; 16 * 1024];
        let _bytes = stream.read(&mut request).await.expect("read request");
        let body = format!("data: {}", "x".repeat(1024 * 1024));
        let headers = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
            body.len()
        );
        stream.write_all(headers.as_bytes()).await.expect("headers");
        stream.write_all(body.as_bytes()).await.expect("body");
    });
    let records = generate(&config(11, 1)).expect("trace");
    let measurements = replay(
        &records,
        &ReplayConfig {
            target: format!("http://{address}/v1/completions"),
            model: "mock".into(),
            time_scale: 100.0,
            max_concurrency: 1,
            request_timeout_ms: 2_000,
        },
    )
    .await
    .expect("replay");
    server.await.expect("server task");
    assert_eq!(measurements[0].status, ReplayStatus::MalformedStream);
    assert!(
        measurements[0]
            .error
            .as_deref()
            .is_some_and(|error| error.contains("1 MiB"))
    );
}

proptest! {
    #[test]
    fn generated_trace_preserves_count_and_positive_tokens(seed in any::<u64>(), count in 1_usize..2_000) {
        let trace = generate(&config(seed, count)).expect("valid config");
        prop_assert_eq!(trace.len(), count);
        prop_assert!(trace.iter().all(|request| request.prompt_tokens > 0 && request.output_tokens > 0));
    }
}
