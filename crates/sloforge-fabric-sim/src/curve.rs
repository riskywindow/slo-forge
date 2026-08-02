use crate::{CounterfactualModifier, CurvePoint, PhysicalResource, ServiceCurve};

#[derive(Clone, Copy, Debug)]
pub(crate) struct CurveEstimate {
    pub duration_us: f64,
    pub uncertainty_fraction: f64,
}

pub(crate) fn estimate(curve: &ServiceCurve, bytes: u64) -> CurveEstimate {
    let (left, right) = bracket(&curve.points, bytes);
    let ratio = if right.message_bytes == left.message_bytes {
        0.0
    } else {
        exact_f64(bytes.saturating_sub(left.message_bytes))
            / exact_f64(right.message_bytes - left.message_bytes)
    };
    let latency_us = interpolate(left.latency_us, right.latency_us, ratio);
    let bandwidth_gbps = interpolate(left.bandwidth_gbps, right.bandwidth_gbps, ratio);
    let uncertainty_fraction =
        interpolate(left.uncertainty_fraction, right.uncertainty_fraction, ratio);
    // Decimal gigabits per second: bytes * 8 / (Gb/s * 1e3) = microseconds.
    let serialization_us = exact_f64(bytes) * 8.0 / (bandwidth_gbps * 1_000.0);
    CurveEstimate {
        duration_us: latency_us + serialization_us,
        uncertainty_fraction,
    }
}

fn exact_f64(value: u64) -> f64 {
    // Validation limits byte counts to 2^53, where every integer is exactly
    // representable. The conversion is isolated so precision assumptions remain explicit.
    #[allow(clippy::cast_precision_loss)]
    {
        value as f64
    }
}

fn bracket(points: &[CurvePoint], bytes: u64) -> (&CurvePoint, &CurvePoint) {
    if bytes <= points[0].message_bytes {
        return (&points[0], &points[0]);
    }
    for pair in points.windows(2) {
        if bytes <= pair[1].message_bytes {
            return (&pair[0], &pair[1]);
        }
    }
    let last = &points[points.len() - 1];
    (last, last)
}

fn interpolate(left: f64, right: f64, ratio: f64) -> f64 {
    left + (right - left) * ratio.clamp(0.0, 1.0)
}

pub(crate) fn apply_curve_modifiers(
    resources: &mut [PhysicalResource],
    modifiers: &[CounterfactualModifier],
) {
    for modifier in modifiers {
        let CounterfactualModifier::ScaleResourceCurve {
            resource_id,
            latency_multiplier,
            bandwidth_multiplier,
        } = modifier
        else {
            continue;
        };
        if let Some(resource) = resources.iter_mut().find(|item| item.id == *resource_id) {
            for point in &mut resource.curve.points {
                point.latency_us *= latency_multiplier;
                point.bandwidth_gbps *= bandwidth_multiplier;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::estimate;
    use crate::{CalibrationProvenance, CurvePoint, ProvenanceKind, ServiceCurve};

    fn curve() -> ServiceCurve {
        ServiceCurve {
            id: "curve".into(),
            points: vec![
                CurvePoint {
                    message_bytes: 1_000,
                    latency_us: 2.0,
                    bandwidth_gbps: 8.0,
                    uncertainty_fraction: 0.1,
                },
                CurvePoint {
                    message_bytes: 3_000,
                    latency_us: 4.0,
                    bandwidth_gbps: 16.0,
                    uncertainty_fraction: 0.2,
                },
            ],
            provenance: CalibrationProvenance {
                kind: ProvenanceKind::Synthetic,
                artifact_uri: "fixture://curve".into(),
                artifact_sha256: "a".repeat(64),
                environment_fingerprint: "fixture".into(),
                collected_at: "2026-01-01T00:00:00Z".into(),
            },
        }
    }

    #[test]
    fn interpolates_curve_and_clamps_outside_range() {
        let middle = estimate(&curve(), 2_000);
        assert!((middle.duration_us - (3.0 + 16_000.0 / 12_000.0)).abs() < 1e-9);
        assert!((middle.uncertainty_fraction - 0.15).abs() < 1e-9);
        assert!(estimate(&curve(), 5_000).duration_us > 4.0);
    }
}
