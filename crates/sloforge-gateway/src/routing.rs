use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RoutingPolicy {
    #[default]
    RoundRobin,
    LeastOutstanding,
    EstimatedEarliestFinish,
    SloSlackAware,
}

impl RoutingPolicy {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RoundRobin => "round_robin",
            Self::LeastOutstanding => "least_outstanding",
            Self::EstimatedEarliestFinish => "estimated_earliest_finish",
            Self::SloSlackAware => "slo_slack_aware",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RouteCandidate {
    pub index: usize,
    pub name: String,
    pub outstanding: usize,
    pub capacity: usize,
    pub estimated_service_ms: f64,
    pub price_per_hour_usd: f64,
    pub weight: u32,
    pub available: bool,
}

#[derive(Debug, Default)]
pub struct RoutingState {
    cursor: AtomicUsize,
}

impl RoutingState {
    #[must_use]
    pub fn select(
        &self,
        policy: RoutingPolicy,
        candidates: &[RouteCandidate],
        deadline_slack_ms: Option<f64>,
        priority: u8,
    ) -> Option<usize> {
        let available = candidates
            .iter()
            .filter(|candidate| candidate.available)
            .collect::<Vec<_>>();
        if available.is_empty() {
            return None;
        }
        match policy {
            RoutingPolicy::RoundRobin => {
                let total_weight = available.iter().fold(0_usize, |total, candidate| {
                    total.saturating_add(candidate.weight as usize)
                });
                let position = self.cursor.fetch_add(1, Ordering::Relaxed) % total_weight.max(1);
                let mut cumulative = 0_usize;
                available
                    .iter()
                    .find(|candidate| {
                        cumulative = cumulative.saturating_add(candidate.weight as usize);
                        position < cumulative
                    })
                    .map(|candidate| candidate.index)
            }
            RoutingPolicy::LeastOutstanding => available
                .iter()
                .min_by(|left, right| {
                    left.outstanding
                        .cmp(&right.outstanding)
                        .then_with(|| left.name.cmp(&right.name))
                })
                .map(|candidate| candidate.index),
            RoutingPolicy::EstimatedEarliestFinish => available
                .iter()
                .min_by(|left, right| {
                    predicted_finish_ms(left)
                        .total_cmp(&predicted_finish_ms(right))
                        .then_with(|| left.name.cmp(&right.name))
                })
                .map(|candidate| candidate.index),
            RoutingPolicy::SloSlackAware => {
                let slack = deadline_slack_ms.unwrap_or(f64::INFINITY);
                available
                    .iter()
                    .min_by(|left, right| {
                        slack_score(left, slack, priority)
                            .total_cmp(&slack_score(right, slack, priority))
                            .then_with(|| left.name.cmp(&right.name))
                    })
                    .map(|candidate| candidate.index)
            }
        }
    }
}

fn predicted_finish_ms(candidate: &RouteCandidate) -> f64 {
    let waves = (candidate.outstanding + 1).div_ceil(candidate.capacity.max(1));
    candidate.estimated_service_ms * f64::from(u32::try_from(waves).unwrap_or(u32::MAX))
}

fn slack_score(candidate: &RouteCandidate, slack_ms: f64, priority: u8) -> f64 {
    let finish = predicted_finish_ms(candidate);
    if finish <= slack_ms {
        // When both alternatives satisfy the deadline, prefer lower operating cost while retaining
        // a latency tie-breaker. Higher-priority requests discount cost more aggressively.
        let price_weight = 1_000.0 / f64::from(priority).mul_add(1.0, 1.0);
        candidate.price_per_hour_usd.mul_add(price_weight, finish)
    } else {
        // Missing the SLO dominates price.
        1_000_000.0 + (finish - slack_ms) * 10_000.0 + finish
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(index: usize, outstanding: usize, service: f64, price: f64) -> RouteCandidate {
        RouteCandidate {
            index,
            name: format!("backend-{index}"),
            outstanding,
            capacity: 1,
            estimated_service_ms: service,
            price_per_hour_usd: price,
            weight: 1,
            available: true,
        }
    }

    #[test]
    fn routing_algorithms_select_expected_backend() {
        let state = RoutingState::default();
        let candidates = vec![candidate(0, 2, 10.0, 0.1), candidate(1, 0, 20.0, 0.2)];
        assert_eq!(
            state.select(RoutingPolicy::LeastOutstanding, &candidates, None, 0),
            Some(1)
        );
        assert_eq!(
            state.select(RoutingPolicy::EstimatedEarliestFinish, &candidates, None, 0),
            Some(1)
        );
        assert_eq!(
            state.select(RoutingPolicy::SloSlackAware, &candidates, Some(100.0), 0),
            Some(0)
        );
    }

    #[test]
    fn unavailable_backends_are_never_selected() {
        let mut candidates = vec![candidate(0, 0, 1.0, 0.0), candidate(1, 0, 2.0, 0.0)];
        candidates[0].available = false;
        assert_eq!(
            RoutingState::default().select(RoutingPolicy::RoundRobin, &candidates, None, 0),
            Some(1)
        );
    }

    #[test]
    fn high_priority_discounts_cost_when_both_routes_meet_slo() {
        let state = RoutingState::default();
        let candidates = vec![candidate(0, 0, 10.0, 1.0), candidate(1, 0, 50.0, 0.1)];
        assert_eq!(
            state.select(RoutingPolicy::SloSlackAware, &candidates, Some(100.0), 0),
            Some(1)
        );
        assert_eq!(
            state.select(
                RoutingPolicy::SloSlackAware,
                &candidates,
                Some(100.0),
                u8::MAX
            ),
            Some(0)
        );
    }
}
