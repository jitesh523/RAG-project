import json
import time
from typing import Dict, Any, Tuple

from prometheus_client import Histogram, Counter

POLICY_EVAL_DURATION = Histogram(
    "policy_eval_duration_seconds",
    "Policy evaluation latency",
)
POLICY_DENIES_TOTAL = Counter(
    "policy_denies_total",
    "Policy deny events",
    labelnames=["reason"],
)
POLICY_OVERRIDES_TOTAL = Counter(
    "policy_overrides_total",
    "Break-glass overrides",
)

DEFAULT_POLICY = {
    "allowed_sources": [],  # empty => any
    "allowed_doc_types": [],  # empty => any
    "residency": [],  # e.g., ["us", "eu"]. empty => any
    "pii_tier": "redact",  # allow|mask|redact|deny
    "break_glass": False,
}


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    x = dict(base)
    for k, v in (override or {}).items():
        x[k] = v
    return x


def evaluate_pre(
    query_ctx: Dict[str, Any], policy: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Return (filter_hints, decision). filter_hints merges into AskFilters.
    decision may contain {deny: bool, reason: str}.
    """
    t0 = time.time()
    try:
        filt = {}
        deny = False
        reason = ""
        allowed_sources = set(policy.get("allowed_sources") or [])
        if allowed_sources:
            filt["sources"] = list(allowed_sources)
        allowed_doc_types = set(policy.get("allowed_doc_types") or [])
        if allowed_doc_types:
            # for now choose single doc_type if provided one; otherwise leave list for UI/reference
            filt["doc_type"] = query_ctx.get("doc_type") or next(
                iter(allowed_doc_types)
            )
        residency = policy.get("residency") or []
        if residency:
            filt["region"] = query_ctx.get("region") or residency[0]
        decision = {"deny": deny, "reason": reason}
        return (filt, decision)
    finally:
        POLICY_EVAL_DURATION.observe(time.time() - t0)


def evaluate_post(
    answer: str, sources: list, policy: Dict[str, Any], override: bool = False
) -> Tuple[str, list, Dict[str, Any]]:
    """
    Apply PII tiering and citation restrictions per policy.
    Returns (answer, sources, decision)
    """
    t0 = time.time()
    try:
        tier = (policy.get("pii_tier") or "redact").lower()
        if override and policy.get("break_glass"):
            try:
                POLICY_OVERRIDES_TOTAL.inc()
            except Exception:
                pass
            return (answer, sources, {"overridden": True})
        if tier == "deny":
            try:
                POLICY_DENIES_TOTAL.labels(reason="pii_deny").inc()
            except Exception:
                pass
            return ("", [], {"denied": True, "reason": "pii_deny"})
        if tier == "mask":
            # coarse masking for emails and phone-like patterns
            import re

            ans = re.sub(
                r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
                "[MASKED_EMAIL]",
                answer or "",
            )
            ans = re.sub(r"\b(?:\+?\d[\s-]?){7,15}\b", "[MASKED_PHONE]", ans)
            # mask sources by hiding filename
            masked_sources = []
            for s in sources or []:
                obj = dict(s)
                if obj.get("source"):
                    obj["source"] = "[MASKED_SOURCE]"
                masked_sources.append(obj)
            return (ans, masked_sources, {"masked": True})
        if tier == "redact":
            # reuse same masking but keep source names
            import re

            ans = re.sub(
                r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
                "[REDACTED_EMAIL]",
                answer or "",
            )
            ans = re.sub(r"\b(?:\+?\d[\s-]?){7,15}\b", "[REDACTED_PHONE]", ans)
            return (ans, sources, {"redacted": True})
        # allow
        return (answer or "", sources or [], {"allowed": True})
    finally:
        POLICY_EVAL_DURATION.observe(time.time() - t0)


def load_policy(redis_client, tenant: str) -> Dict[str, Any]:
    # returns merged policy: default + tenant override
    pol = DEFAULT_POLICY
    try:
        if redis_client is not None:
            raw = redis_client.get(f"policy:{tenant}")
            if raw:
                tenant_pol = json.loads(raw)
                pol = _merge(DEFAULT_POLICY, tenant_pol)
    except Exception:
        pass
    return pol
