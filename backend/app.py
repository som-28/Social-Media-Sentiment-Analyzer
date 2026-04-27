from __future__ import annotations

import base64
import io
import json
import os
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from collections import Counter
from collections import deque
from datetime import datetime, timezone

import pandas as pd
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

from services.sentiment import SentimentEngine
from services.preprocess import STOPWORDS

app = Flask(__name__)
CORS(app)
engine = SentimentEngine()
PREDICTION_HISTORY = deque(maxlen=500)
SNAPSHOT_STORE_PATH = os.environ.get(
    "SNAPSHOT_STORE_PATH",
    os.path.join(os.path.dirname(__file__), "data", "snapshots.json"),
)
WATCHLIST_STORE_PATH = os.environ.get(
    "WATCHLIST_STORE_PATH",
    os.path.join(os.path.dirname(__file__), "data", "watchlists.json"),
)
SCHEDULE_STORE_PATH = os.environ.get(
    "SCHEDULE_STORE_PATH",
    os.path.join(os.path.dirname(__file__), "data", "report_schedules.json"),
)
DELIVERY_LOG_STORE_PATH = os.environ.get(
    "DELIVERY_LOG_STORE_PATH",
    os.path.join(os.path.dirname(__file__), "data", "delivery_log.json"),
)
ALERT_RULE_STORE_PATH = os.environ.get(
    "ALERT_RULE_STORE_PATH",
    os.path.join(os.path.dirname(__file__), "data", "alert_rules.json"),
)
ALERT_EVENT_STORE_PATH = os.environ.get(
    "ALERT_EVENT_STORE_PATH",
    os.path.join(os.path.dirname(__file__), "data", "alert_events.json"),
)
HISTORY_STORE_PATH = os.environ.get(
    "HISTORY_STORE_PATH",
    os.path.join(os.path.dirname(__file__), "data", "historical_events.json"),
)
SNAPSHOTS: dict[str, dict] = {}
WATCHLISTS: dict[str, dict] = {}
REPORT_SCHEDULES: dict[str, dict] = {}
DELIVERY_LOG: list[dict] = []
ALERT_RULES: dict[str, dict] = {}
ALERT_EVENTS: list[dict] = []
HISTORICAL_EVENTS: list[dict] = []
SCHEDULE_LOCK = threading.Lock()
SCHEDULE_WORKER_STARTED = False

CSV_TEXT_COLUMN_ALIASES = {
    "text",
    "tweet",
    "tweets",
    "content",
    "message",
    "review",
    "comment",
    "sentence",
}

CSV_LOCATION_COLUMN_ALIASES = {
    "location",
    "city",
    "country",
    "region",
    "state",
    "place",
}

CSV_BRAND_COLUMN_ALIASES = {
    "brand",
    "company",
    "product",
    "competitor",
}

CSV_DEMOGRAPHIC_COLUMN_ALIASES = {
    "demographic",
    "group",
    "segment",
    "audience",
    "gender",
    "age_group",
    "region_group",
}

CSV_TIMESTAMP_COLUMN_ALIASES = {
    "timestamp",
    "date",
    "created_at",
    "datetime",
    "time",
}

CONTEXT_TOPIC_RULES = {
    "Customer Support": {"support", "service", "agent", "response", "refund", "help", "care"},
    "Delivery & Logistics": {"delivery", "shipping", "arrived", "late", "delay", "courier", "shipment"},
    "Product Quality": {"quality", "build", "durable", "broken", "defect", "performance", "reliable"},
    "Price & Value": {"price", "cost", "expensive", "cheap", "affordable", "value", "worth"},
    "User Experience": {"app", "update", "feature", "ui", "ux", "bug", "crash", "slow"},
    "Trust & Safety": {"safe", "safety", "risk", "privacy", "secure", "fraud", "scam"},
    "Brand Perception": {"brand", "reputation", "image", "trust", "loyal", "recommend"},
}

TOPIC_NOISE_TOKENS = {
    "amp",
    "http",
    "https",
    "www",
    "com",
    "tco",
    "rt",
    "just",
    "from",
    "what",
    "have",
    "can",
    "out",
    "all",
    "new",
    "see",
}

LIVE_SOURCE_TIMEOUT_SECONDS = 12
LIVE_SOURCE_HEADERS = {"User-Agent": "SocialMediaSentimentAnalyzer/1.0"}


def resolve_column(normalized_columns: dict[str, str], aliases: set[str]) -> str | None:
    return next((normalized_columns[name] for name in aliases if name in normalized_columns), None)


def extract_context_topics(predictions: list[dict]) -> list[dict]:
    topic_stats: dict[str, Counter] = {}
    topic_keywords_seen: dict[str, Counter] = {}

    for item in predictions:
        normalized = str(item.get("normalized_text", "")).strip().lower()
        if not normalized:
            continue
        tokens = {
            token
            for token in normalized.split()
            if token and token not in STOPWORDS and token not in TOPIC_NOISE_TOKENS and len(token) > 2
        }
        if not tokens:
            continue

        label = str(item.get("label", "neutral"))
        for topic, keywords in CONTEXT_TOPIC_RULES.items():
            matched = sorted(tokens.intersection(keywords))
            if not matched:
                continue
            if topic not in topic_stats:
                topic_stats[topic] = Counter()
                topic_keywords_seen[topic] = Counter()

            topic_stats[topic]["count"] += 1
            topic_stats[topic][label] += 1
            topic_keywords_seen[topic].update(matched)

    topics: list[dict] = []
    for topic, counts in topic_stats.items():
        total = int(counts.get("count", 0))
        if total <= 0:
            continue
        top_keywords = [word for word, _ in topic_keywords_seen[topic].most_common(4)]
        topics.append(
            {
                "topic": topic,
                "count": total,
                "positive": int(counts.get("positive", 0)),
                "neutral": int(counts.get("neutral", 0)),
                "negative": int(counts.get("negative", 0)),
                "keywords": top_keywords,
            }
        )

    topics.sort(key=lambda item: item["count"], reverse=True)
    return topics[:10]


def load_snapshots() -> None:
    global SNAPSHOTS
    try:
        os.makedirs(os.path.dirname(SNAPSHOT_STORE_PATH), exist_ok=True)
        if not os.path.exists(SNAPSHOT_STORE_PATH):
            SNAPSHOTS = {}
            return
        with open(SNAPSHOT_STORE_PATH, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, dict):
            SNAPSHOTS = payload
        else:
            SNAPSHOTS = {}
    except (OSError, json.JSONDecodeError):
        SNAPSHOTS = {}


def save_snapshots() -> None:
    os.makedirs(os.path.dirname(SNAPSHOT_STORE_PATH), exist_ok=True)
    with open(SNAPSHOT_STORE_PATH, "w", encoding="utf-8") as file:
        json.dump(SNAPSHOTS, file, ensure_ascii=True, indent=2)


def load_watchlists() -> None:
    global WATCHLISTS
    try:
        os.makedirs(os.path.dirname(WATCHLIST_STORE_PATH), exist_ok=True)
        if not os.path.exists(WATCHLIST_STORE_PATH):
            WATCHLISTS = {}
            return
        with open(WATCHLIST_STORE_PATH, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, dict):
            WATCHLISTS = payload
        else:
            WATCHLISTS = {}
    except (OSError, json.JSONDecodeError):
        WATCHLISTS = {}


def save_watchlists() -> None:
    os.makedirs(os.path.dirname(WATCHLIST_STORE_PATH), exist_ok=True)
    with open(WATCHLIST_STORE_PATH, "w", encoding="utf-8") as file:
        json.dump(WATCHLISTS, file, ensure_ascii=True, indent=2)


def load_report_schedules() -> None:
    global REPORT_SCHEDULES
    try:
        os.makedirs(os.path.dirname(SCHEDULE_STORE_PATH), exist_ok=True)
        if not os.path.exists(SCHEDULE_STORE_PATH):
            REPORT_SCHEDULES = {}
            return
        with open(SCHEDULE_STORE_PATH, "r", encoding="utf-8") as file:
            payload = json.load(file)
        REPORT_SCHEDULES = payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        REPORT_SCHEDULES = {}


def save_report_schedules() -> None:
    os.makedirs(os.path.dirname(SCHEDULE_STORE_PATH), exist_ok=True)
    with open(SCHEDULE_STORE_PATH, "w", encoding="utf-8") as file:
        json.dump(REPORT_SCHEDULES, file, ensure_ascii=True, indent=2)


def load_delivery_log() -> None:
    global DELIVERY_LOG
    try:
        os.makedirs(os.path.dirname(DELIVERY_LOG_STORE_PATH), exist_ok=True)
        if not os.path.exists(DELIVERY_LOG_STORE_PATH):
            DELIVERY_LOG = []
            return
        with open(DELIVERY_LOG_STORE_PATH, "r", encoding="utf-8") as file:
            payload = json.load(file)
        DELIVERY_LOG = payload if isinstance(payload, list) else []
    except (OSError, json.JSONDecodeError):
        DELIVERY_LOG = []


def save_delivery_log() -> None:
    os.makedirs(os.path.dirname(DELIVERY_LOG_STORE_PATH), exist_ok=True)
    with open(DELIVERY_LOG_STORE_PATH, "w", encoding="utf-8") as file:
        json.dump(DELIVERY_LOG, file, ensure_ascii=True, indent=2)


def load_alert_rules() -> None:
    global ALERT_RULES
    try:
        os.makedirs(os.path.dirname(ALERT_RULE_STORE_PATH), exist_ok=True)
        if not os.path.exists(ALERT_RULE_STORE_PATH):
            ALERT_RULES = {}
            return
        with open(ALERT_RULE_STORE_PATH, "r", encoding="utf-8") as file:
            payload = json.load(file)
        ALERT_RULES = payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        ALERT_RULES = {}


def save_alert_rules() -> None:
    os.makedirs(os.path.dirname(ALERT_RULE_STORE_PATH), exist_ok=True)
    with open(ALERT_RULE_STORE_PATH, "w", encoding="utf-8") as file:
        json.dump(ALERT_RULES, file, ensure_ascii=True, indent=2)


def load_alert_events() -> None:
    global ALERT_EVENTS
    try:
        os.makedirs(os.path.dirname(ALERT_EVENT_STORE_PATH), exist_ok=True)
        if not os.path.exists(ALERT_EVENT_STORE_PATH):
            ALERT_EVENTS = []
            return
        with open(ALERT_EVENT_STORE_PATH, "r", encoding="utf-8") as file:
            payload = json.load(file)
        ALERT_EVENTS = payload if isinstance(payload, list) else []
    except (OSError, json.JSONDecodeError):
        ALERT_EVENTS = []


def save_alert_events() -> None:
    os.makedirs(os.path.dirname(ALERT_EVENT_STORE_PATH), exist_ok=True)
    with open(ALERT_EVENT_STORE_PATH, "w", encoding="utf-8") as file:
        json.dump(ALERT_EVENTS, file, ensure_ascii=True, indent=2)


def load_historical_events() -> None:
    global HISTORICAL_EVENTS
    try:
        os.makedirs(os.path.dirname(HISTORY_STORE_PATH), exist_ok=True)
        if not os.path.exists(HISTORY_STORE_PATH):
            HISTORICAL_EVENTS = []
            return
        with open(HISTORY_STORE_PATH, "r", encoding="utf-8") as file:
            payload = json.load(file)
        HISTORICAL_EVENTS = payload if isinstance(payload, list) else []
    except (OSError, json.JSONDecodeError):
        HISTORICAL_EVENTS = []


def save_historical_events() -> None:
    os.makedirs(os.path.dirname(HISTORY_STORE_PATH), exist_ok=True)
    with open(HISTORY_STORE_PATH, "w", encoding="utf-8") as file:
        json.dump(HISTORICAL_EVENTS, file, ensure_ascii=True, indent=2)


def build_actionable_recommendations(distribution: dict, topics: list[dict], alerts: list[dict]) -> list[str]:
    recommendations: list[str] = []
    positive = int(distribution.get("positive", 0))
    neutral = int(distribution.get("neutral", 0))
    negative = int(distribution.get("negative", 0))
    total = max(1, positive + neutral + negative)
    negative_ratio = negative / total

    if negative_ratio >= 0.45:
        recommendations.append("Negative sentiment is elevated; prioritize incident response and stakeholder communication.")
    elif negative_ratio >= 0.3:
        recommendations.append("Negative sentiment is rising; monitor issue clusters and publish clarifications.")
    else:
        recommendations.append("Sentiment baseline is stable; maintain routine monitoring cadence.")

    if topics:
        top_topic = topics[0]
        topic_name = str(top_topic.get("topic", "Key topic"))
        topic_negative = int(top_topic.get("negative", 0))
        topic_count = max(1, int(top_topic.get("count", 1)))
        if (topic_negative / topic_count) >= 0.4:
            recommendations.append(f"Top concern area is {topic_name}; assign owner and mitigation SLA for this theme.")
        else:
            recommendations.append(f"Keep amplifying positive momentum around {topic_name} through content and support playbooks.")

    if alerts:
        recommendations.append("At least one alert rule triggered recently; review alert timeline and validate response actions.")

    return recommendations[:4]


def build_distribution_summary(distribution: dict) -> str:
    positive = int(distribution.get("positive", 0))
    neutral = int(distribution.get("neutral", 0))
    negative = int(distribution.get("negative", 0))
    total = max(1, positive + neutral + negative)
    dominant = max(
        [("positive", positive), ("neutral", neutral), ("negative", negative)],
        key=lambda pair: pair[1],
    )[0]
    return (
        f"Dominant sentiment is {dominant}. "
        f"Positive {round((positive / total) * 100, 1)}%, "
        f"Neutral {round((neutral / total) * 100, 1)}%, "
        f"Negative {round((negative / total) * 100, 1)}%."
    )


def queue_scheduled_delivery(schedule: dict) -> dict:
    distribution = schedule.get("distribution") or {}
    summary = build_distribution_summary(distribution)
    delivery = {
        "id": str(uuid.uuid4()),
        "schedule_id": schedule.get("id"),
        "title": schedule.get("title", "Scheduled Sentiment Report"),
        "destination": schedule.get("destination", "unknown"),
        "channel": schedule.get("channel", "email"),
        "summary": summary,
        "status": "sent_simulated",
        "delivered_at": datetime.now(timezone.utc).isoformat(),
    }
    DELIVERY_LOG.append(delivery)
    if len(DELIVERY_LOG) > 200:
        del DELIVERY_LOG[:-200]
    save_delivery_log()
    return delivery


def evaluate_alert_rules(now: datetime) -> list[dict]:
    triggered: list[dict] = []
    for rule in ALERT_RULES.values():
        if not bool(rule.get("enabled", True)):
            continue

        window = max(5, int(rule.get("window", 30) or 30))
        threshold = float(rule.get("threshold", 0.5) or 0.5)
        cooldown_minutes = max(1, int(rule.get("cooldown_minutes", 30) or 30))
        history_window = list(PREDICTION_HISTORY)[-window:] if len(PREDICTION_HISTORY) else []
        if not history_window:
            continue

        negative_count = sum(1 for item in history_window if str(item.get("label", "")) == "negative")
        ratio = negative_count / len(history_window)

        last_triggered_at = rule.get("last_triggered_at")
        if last_triggered_at:
            last_dt = pd.to_datetime(last_triggered_at, errors="coerce", utc=True)
            if not pd.isna(last_dt) and now < (last_dt + pd.Timedelta(minutes=cooldown_minutes)):
                continue

        if ratio >= threshold:
            event = {
                "id": str(uuid.uuid4()),
                "rule_id": rule.get("id"),
                "rule_name": rule.get("name", "Negative Spike Rule"),
                "triggered_at": now.isoformat(),
                "negative_ratio": round(ratio, 4),
                "window": len(history_window),
                "message": f"Negative ratio {round(ratio * 100, 1)}% exceeded threshold {round(threshold * 100, 1)}%.",
            }
            ALERT_EVENTS.append(event)
            if len(ALERT_EVENTS) > 300:
                del ALERT_EVENTS[:-300]
            rule["last_triggered_at"] = now.isoformat()
            triggered.append(event)

    if triggered:
        save_alert_rules()
        save_alert_events()
    return triggered


def schedule_dispatch_worker() -> None:
    while True:
        try:
            now = datetime.now(timezone.utc)
            with SCHEDULE_LOCK:
                dirty = False
                for schedule in REPORT_SCHEDULES.values():
                    if not bool(schedule.get("enabled", True)):
                        continue
                    interval_minutes = max(1, int(schedule.get("interval_minutes", 60)))
                    next_run_iso = schedule.get("next_run_at")
                    if not next_run_iso:
                        schedule["next_run_at"] = now.isoformat()
                        dirty = True
                        continue
                    next_run = pd.to_datetime(next_run_iso, errors="coerce", utc=True)
                    if pd.isna(next_run):
                        schedule["next_run_at"] = now.isoformat()
                        dirty = True
                        continue
                    if now >= next_run:
                        queue_scheduled_delivery(schedule)
                        schedule["last_run_at"] = now.isoformat()
                        schedule["next_run_at"] = (now + pd.Timedelta(minutes=interval_minutes)).isoformat()
                        dirty = True
                if dirty:
                    save_report_schedules()
                evaluate_alert_rules(now)
        except Exception:
            pass
        time.sleep(30)


def start_schedule_worker_once() -> None:
    global SCHEDULE_WORKER_STARTED
    if SCHEDULE_WORKER_STARTED:
        return
    SCHEDULE_WORKER_STARTED = True
    worker = threading.Thread(target=schedule_dispatch_worker, daemon=True)
    worker.start()


def record_prediction(
    label: str,
    model: str,
    emotion: str,
    confidence: float,
    normalized_text: str = "",
    source: str = "runtime",
) -> None:
    event = {
        "id": str(uuid.uuid4()),
        "label": str(label),
        "model": str(model),
        "emotion": str(emotion),
        "confidence": float(confidence),
        "normalized_text": str(normalized_text),
        "source": str(source),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    PREDICTION_HISTORY.append(event)
    HISTORICAL_EVENTS.append(event)
    if len(HISTORICAL_EVENTS) > 10000:
        del HISTORICAL_EVENTS[:-10000]
    save_historical_events()


def analyze_text_entries(text_entries: list[str], model: str) -> dict:
    total_rows = len(text_entries)
    empty_text_rows = 0
    duplicate_text_rows = 0
    seen_normalized: set[str] = set()

    predictions = []
    language_distribution_counter: Counter = Counter()
    for raw_value in text_entries:
        value = str(raw_value or "").strip()
        if not value:
            empty_text_rows += 1
            continue

        prediction = engine.predict(text=value, model=model)
        normalized = str(prediction.get("normalized_text", "")).strip()
        if normalized in seen_normalized:
            duplicate_text_rows += 1
            continue

        seen_normalized.add(normalized)
        predictions.append(prediction)
        emotion = str((prediction.get("emotion") or {}).get("label", "neutral"))
        confidence = float(prediction.get("confidence", 0.5))
        record_prediction(
            label=prediction.get("label", "neutral"),
            model=model,
            emotion=emotion,
            confidence=confidence,
            normalized_text=str(prediction.get("normalized_text", "")),
            source="ingest_live",
        )
        language_distribution_counter[str((prediction.get("language") or {}).get("label", "unknown"))] += 1

    distribution = dict(Counter([item["label"] for item in predictions]))
    emotion_distribution = dict(
        Counter([str((item.get("emotion") or {}).get("label", "neutral")) for item in predictions])
    )
    topics = extract_context_topics(predictions)
    sarcasm_count = sum(1 for item in predictions if bool((item.get("sarcasm") or {}).get("detected", False)))
    sarcasm_rate = round((sarcasm_count / len(predictions)), 4) if predictions else 0.0
    toxic_rows = [item for item in predictions if str((item.get("toxicity") or {}).get("level", "none")) != "none"]
    toxicity_count = len(toxic_rows)
    avg_toxicity_score = (
        round(sum(float((item.get("toxicity") or {}).get("score", 0.0)) for item in predictions) / len(predictions), 4)
        if predictions
        else 0.0
    )
    uncertainty_avg = (
        round(sum(float((item.get("uncertainty") or {}).get("score", 0.0)) for item in predictions) / len(predictions), 4)
        if predictions
        else 0.0
    )

    quality_score = 1.0
    if total_rows > 0:
        penalty = (empty_text_rows + duplicate_text_rows) / total_rows
        quality_score = max(0.0, 1.0 - penalty)

    return {
        "count": len(predictions),
        "distribution": distribution,
        "emotion_distribution": emotion_distribution,
        "language_distribution": dict(language_distribution_counter),
        "geo_summary": [],
        "geo_source_column": None,
        "topics": topics,
        "sarcasm_count": sarcasm_count,
        "sarcasm_rate": sarcasm_rate,
        "toxicity_count": toxicity_count,
        "avg_toxicity_score": avg_toxicity_score,
        "uncertainty_average": uncertainty_avg,
        "competitor_benchmark": [],
        "campaign_impact": None,
        "fairness_audit": None,
        "data_quality": {
            "total_rows": total_rows,
            "retained_rows": len(predictions),
            "empty_text_rows": empty_text_rows,
            "duplicate_text_rows": duplicate_text_rows,
            "quality_score": round(quality_score, 4),
        },
        "cleaning_actions": {
            "trimmed_whitespace": True,
            "deduplicated_by_normalized_text": True,
        },
        "predictions": predictions,
    }


def fetch_live_posts(
    source_type: str,
    source_url: str,
    limit: int,
    text_field: str,
    manual_texts: list[str] | None,
) -> list[str]:
    source_type = str(source_type).strip().lower()
    text_field = str(text_field or "text").strip() or "text"
    limit = max(1, min(int(limit or 20), 200))

    if source_type == "manual":
        values = [str(item).strip() for item in (manual_texts or []) if str(item).strip()]
        return values[:limit]

    if not source_url:
        raise ValueError("Field 'source_url' is required for this source type.")

    response = requests.get(
        source_url,
        timeout=LIVE_SOURCE_TIMEOUT_SECONDS,
        headers=LIVE_SOURCE_HEADERS,
    )
    response.raise_for_status()

    if source_type == "reddit_json":
        payload = response.json()
        children = ((payload or {}).get("data") or {}).get("children") or []
        values = []
        for child in children:
            data = child.get("data") or {}
            title = str(data.get("title", "")).strip()
            body = str(data.get("selftext", "")).strip()
            merged = f"{title} {body}".strip()
            if merged:
                values.append(merged)
        return values[:limit]

    if source_type == "rss":
        root = ET.fromstring(response.text)
        values = []
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            description = (item.findtext("description") or "").strip()
            merged = f"{title} {description}".strip()
            if merged:
                values.append(merged)
        return values[:limit]

    if source_type == "json_feed":
        payload = response.json()
        values = []
        if isinstance(payload, list):
            entries = payload
        elif isinstance(payload, dict):
            entries = payload.get("items") or payload.get("data") or []
        else:
            entries = []

        for entry in entries:
            if isinstance(entry, dict):
                value = str(entry.get(text_field, "")).strip()
                if value:
                    values.append(value)
        return values[:limit]

    raise ValueError("Unsupported source_type. Use manual, reddit_json, rss, or json_feed.")


@app.get("/")
def index():
    return jsonify(
        {
            "message": "Social Media Sentiment Analyzer API is running.",
            "health": "/api/health",
            "predict": "/api/predict",
            "predict_csv": "/api/predict-csv",
            "live_summary": "/api/live-summary",
            "aspect_analysis": "Included in /api/predict response as aspects",
            "snapshot_storage": SNAPSHOT_STORE_PATH,
            "watchlist_storage": WATCHLIST_STORE_PATH,
            "schedule_storage": SCHEDULE_STORE_PATH,
        }
    )


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/ingest-live")
def ingest_live_source():
    payload = request.get_json(silent=True) or {}
    source_type = str(payload.get("source_type", "manual"))
    source_url = str(payload.get("source_url", "")).strip()
    model = str(payload.get("model", "distilbert"))
    text_field = str(payload.get("text_field", "text"))
    limit = int(payload.get("limit", 30) or 30)
    manual_texts = payload.get("manual_texts")

    try:
        posts = fetch_live_posts(
            source_type=source_type,
            source_url=source_url,
            limit=limit,
            text_field=text_field,
            manual_texts=manual_texts if isinstance(manual_texts, list) else None,
        )
        analysis = analyze_text_entries(posts, model=model)
        return jsonify(
            {
                "source": {
                    "source_type": source_type,
                    "source_url": source_url,
                    "fetched_posts": len(posts),
                },
                **analysis,
            }
        )
    except requests.RequestException as exc:
        return jsonify({"error": f"Failed to fetch source: {exc}"}), 400
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/watchlists")
def list_watchlists():
    return jsonify({"watchlists": list(WATCHLISTS.values())})


@app.post("/api/watchlists")
def create_watchlist():
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name", "")).strip()
    source_type = str(payload.get("source_type", "manual")).strip()
    source_url = str(payload.get("source_url", "")).strip()
    model = str(payload.get("model", "distilbert")).strip()
    text_field = str(payload.get("text_field", "text")).strip() or "text"
    limit = int(payload.get("limit", 30) or 30)
    manual_texts = payload.get("manual_texts")

    if not name:
        return jsonify({"error": "Field 'name' is required."}), 400

    watchlist_id = str(uuid.uuid4())
    WATCHLISTS[watchlist_id] = {
        "id": watchlist_id,
        "name": name,
        "source_type": source_type,
        "source_url": source_url,
        "model": model,
        "text_field": text_field,
        "limit": max(1, min(limit, 200)),
        "manual_texts": manual_texts if isinstance(manual_texts, list) else [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_watchlists()
    return jsonify({"watchlist": WATCHLISTS[watchlist_id]})


@app.delete("/api/watchlists/<watchlist_id>")
def delete_watchlist(watchlist_id: str):
    if watchlist_id not in WATCHLISTS:
        return jsonify({"error": "Watchlist not found."}), 404
    del WATCHLISTS[watchlist_id]
    save_watchlists()
    return jsonify({"deleted": True, "watchlist_id": watchlist_id})


@app.post("/api/watchlists/<watchlist_id>/run")
def run_watchlist(watchlist_id: str):
    watchlist = WATCHLISTS.get(watchlist_id)
    if not watchlist:
        return jsonify({"error": "Watchlist not found."}), 404

    try:
        posts = fetch_live_posts(
            source_type=watchlist.get("source_type", "manual"),
            source_url=watchlist.get("source_url", ""),
            limit=int(watchlist.get("limit", 30) or 30),
            text_field=watchlist.get("text_field", "text"),
            manual_texts=watchlist.get("manual_texts", []),
        )
        analysis = analyze_text_entries(posts, model=str(watchlist.get("model", "distilbert")))
        return jsonify({"watchlist": watchlist, **analysis})
    except requests.RequestException as exc:
        return jsonify({"error": f"Failed to fetch source: {exc}"}), 400
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/report-schedules")
def list_report_schedules():
    return jsonify({"schedules": list(REPORT_SCHEDULES.values())})


@app.post("/api/report-schedules")
def create_report_schedule():
    payload = request.get_json(silent=True) or {}
    title = str(payload.get("title", "Scheduled Sentiment Report")).strip()
    destination = str(payload.get("destination", "")).strip()
    channel = str(payload.get("channel", "email")).strip() or "email"
    interval_minutes = int(payload.get("interval_minutes", 60) or 60)
    distribution = payload.get("distribution") or {}
    topics = payload.get("topics") or []

    if not destination:
        return jsonify({"error": "Field 'destination' is required."}), 400

    schedule_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    schedule = {
        "id": schedule_id,
        "title": title,
        "destination": destination,
        "channel": channel,
        "interval_minutes": max(1, min(interval_minutes, 10080)),
        "distribution": distribution,
        "topics": topics if isinstance(topics, list) else [],
        "enabled": True,
        "created_at": now.isoformat(),
        "next_run_at": (now + pd.Timedelta(minutes=max(1, min(interval_minutes, 10080)))).isoformat(),
        "last_run_at": None,
    }
    with SCHEDULE_LOCK:
        REPORT_SCHEDULES[schedule_id] = schedule
        save_report_schedules()

    return jsonify({"schedule": schedule})


@app.post("/api/report-schedules/<schedule_id>/run-now")
def run_report_schedule_now(schedule_id: str):
    schedule = REPORT_SCHEDULES.get(schedule_id)
    if not schedule:
        return jsonify({"error": "Schedule not found."}), 404

    with SCHEDULE_LOCK:
        delivery = queue_scheduled_delivery(schedule)
        schedule["last_run_at"] = datetime.now(timezone.utc).isoformat()
        save_report_schedules()

    return jsonify({"delivery": delivery, "schedule": schedule})


@app.patch("/api/report-schedules/<schedule_id>")
def update_report_schedule(schedule_id: str):
    schedule = REPORT_SCHEDULES.get(schedule_id)
    if not schedule:
        return jsonify({"error": "Schedule not found."}), 404

    payload = request.get_json(silent=True) or {}
    if "enabled" in payload:
        schedule["enabled"] = bool(payload.get("enabled"))
    if "interval_minutes" in payload:
        schedule["interval_minutes"] = max(1, min(int(payload.get("interval_minutes") or 60), 10080))
    if "distribution" in payload and isinstance(payload.get("distribution"), dict):
        schedule["distribution"] = payload.get("distribution")
    if "topics" in payload and isinstance(payload.get("topics"), list):
        schedule["topics"] = payload.get("topics")
    if "destination" in payload:
        schedule["destination"] = str(payload.get("destination") or schedule.get("destination"))
    if "title" in payload:
        schedule["title"] = str(payload.get("title") or schedule.get("title"))

    save_report_schedules()
    return jsonify({"schedule": schedule})


@app.delete("/api/report-schedules/<schedule_id>")
def delete_report_schedule(schedule_id: str):
    if schedule_id not in REPORT_SCHEDULES:
        return jsonify({"error": "Schedule not found."}), 404
    del REPORT_SCHEDULES[schedule_id]
    save_report_schedules()
    return jsonify({"deleted": True, "schedule_id": schedule_id})


@app.get("/api/report-deliveries")
def list_report_deliveries():
    return jsonify({"deliveries": DELIVERY_LOG[-100:]})


@app.get("/api/alert-rules")
def list_alert_rules():
    return jsonify({"rules": list(ALERT_RULES.values())})


@app.post("/api/alert-rules")
def create_alert_rule():
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name", "Negative Spike Alert")).strip()
    threshold = float(payload.get("threshold", 0.5) or 0.5)
    window = int(payload.get("window", 30) or 30)
    cooldown_minutes = int(payload.get("cooldown_minutes", 30) or 30)

    rule_id = str(uuid.uuid4())
    rule = {
        "id": rule_id,
        "name": name,
        "threshold": max(0.05, min(threshold, 1.0)),
        "window": max(5, min(window, 500)),
        "cooldown_minutes": max(1, min(cooldown_minutes, 10080)),
        "enabled": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_triggered_at": None,
    }
    ALERT_RULES[rule_id] = rule
    save_alert_rules()
    return jsonify({"rule": rule})


@app.delete("/api/alert-rules/<rule_id>")
def delete_alert_rule(rule_id: str):
    if rule_id not in ALERT_RULES:
        return jsonify({"error": "Rule not found."}), 404
    del ALERT_RULES[rule_id]
    save_alert_rules()
    return jsonify({"deleted": True, "rule_id": rule_id})


@app.post("/api/alert-rules/evaluate")
def evaluate_alert_rules_now():
    now = datetime.now(timezone.utc)
    with SCHEDULE_LOCK:
        events = evaluate_alert_rules(now)
    return jsonify({"triggered": events, "total_events": len(ALERT_EVENTS)})


@app.get("/api/alerts")
def list_alert_events():
    return jsonify({"alerts": ALERT_EVENTS[-100:]})


@app.get("/api/live-summary")
def live_summary():
    window_size = request.args.get("window", default=100, type=int)
    if window_size is None or window_size <= 0:
        window_size = 100

    window_size = min(window_size, len(PREDICTION_HISTORY))
    window = list(PREDICTION_HISTORY)[-window_size:] if window_size else []

    distribution = Counter([item["label"] for item in window])
    model_usage = Counter([item["model"] for item in window])
    emotion_distribution = Counter([item.get("emotion", "neutral") for item in window])
    recent_window = window[-20:] if len(window) >= 20 else window
    recent_negative = sum(1 for item in recent_window if item.get("label") == "negative")
    recent_total = len(recent_window)
    negative_ratio = (recent_negative / recent_total) if recent_total else 0.0
    negative_spike_alert = {
        "triggered": bool(recent_total >= 10 and negative_ratio >= 0.5),
        "negative_ratio": round(negative_ratio, 4),
        "window": recent_total,
        "message": (
            "Negative sentiment spike detected in recent predictions."
            if recent_total >= 10 and negative_ratio >= 0.5
            else "No negative spike detected."
        ),
    }

    return jsonify(
        {
            "window_size": window_size,
            "available_history": len(PREDICTION_HISTORY),
            "distribution": {
                "positive": distribution.get("positive", 0),
                "neutral": distribution.get("neutral", 0),
                "negative": distribution.get("negative", 0),
            },
            "model_usage": dict(model_usage),
            "emotion_distribution": dict(emotion_distribution),
            "negative_spike_alert": negative_spike_alert,
            "latest_event": window[-1] if window else None,
        }
    )


@app.get("/api/trend-series")
def trend_series():
    window_size = request.args.get("window", default=150, type=int)
    if window_size is None or window_size <= 0:
        window_size = 150

    window_size = min(window_size, len(PREDICTION_HISTORY))
    window = list(PREDICTION_HISTORY)[-window_size:] if window_size else []

    points = []
    running = {"positive": 0, "neutral": 0, "negative": 0}
    for idx, event in enumerate(window, start=1):
        label = str(event.get("label", "neutral"))
        if label not in running:
            label = "neutral"
        running[label] += 1
        points.append(
            {
                "step": idx,
                "positive": running["positive"],
                "neutral": running["neutral"],
                "negative": running["negative"],
            }
        )

    return jsonify(
        {
            "window_size": window_size,
            "points": points,
        }
    )


def aggregate_historical_points(events: list[dict], bucket: str) -> list[dict]:
    counts: dict[str, dict[str, int]] = {}
    for event in events:
        timestamp_raw = str(event.get("timestamp", ""))
        parsed = pd.to_datetime(timestamp_raw, errors="coerce", utc=True)
        if pd.isna(parsed):
            continue
        key = parsed.strftime("%Y-%m-%d") if bucket == "day" else parsed.strftime("%Y-%m-%d %H:00")
        if key not in counts:
            counts[key] = {"positive": 0, "neutral": 0, "negative": 0}
        label = str(event.get("label", "neutral"))
        if label not in counts[key]:
            label = "neutral"
        counts[key][label] += 1

    points = []
    for key in sorted(counts.keys()):
        row = counts[key]
        points.append(
            {
                "bucket": key,
                "positive": row.get("positive", 0),
                "neutral": row.get("neutral", 0),
                "negative": row.get("negative", 0),
            }
        )
    return points


@app.get("/api/recommendations")
def recommendations():
    window_size = request.args.get("window", default=200, type=int)
    if window_size is None or window_size <= 0:
        window_size = 200
    window = HISTORICAL_EVENTS[-window_size:] if HISTORICAL_EVENTS else []
    distribution = Counter([str(item.get("label", "neutral")) for item in window])
    topics = extract_context_topics(window)
    recent_alerts = ALERT_EVENTS[-20:]
    actions = build_actionable_recommendations(
        {
            "positive": distribution.get("positive", 0),
            "neutral": distribution.get("neutral", 0),
            "negative": distribution.get("negative", 0),
        },
        topics,
        recent_alerts,
    )
    return jsonify({"recommendations": actions, "window_size": len(window)})


@app.get("/api/today-dashboard")
def today_dashboard():
    window_size = request.args.get("window", default=200, type=int)
    if window_size is None or window_size <= 0:
        window_size = 200
    window = HISTORICAL_EVENTS[-window_size:] if HISTORICAL_EVENTS else []
    distribution = Counter([str(item.get("label", "neutral")) for item in window])
    model_usage = Counter([str(item.get("model", "unknown")) for item in window])
    emotion_distribution = Counter([str(item.get("emotion", "neutral")) for item in window])
    topics = extract_context_topics(window)
    recent_alerts = ALERT_EVENTS[-10:]

    movers = []
    for topic in topics[:5]:
        count = int(topic.get("count", 0))
        negative = int(topic.get("negative", 0))
        sentiment_pressure = round((negative / max(1, count)) * 100, 1)
        movers.append(
            {
                "topic": str(topic.get("topic", "General")),
                "count": count,
                "sentiment_pressure": sentiment_pressure,
            }
        )

    risks = []
    negative_count = distribution.get("negative", 0)
    total = max(1, distribution.get("positive", 0) + distribution.get("neutral", 0) + negative_count)
    if (negative_count / total) >= 0.35:
        risks.append("Negative share is elevated in the latest analysis window.")
    if recent_alerts:
        risks.append("Recent alert rule triggers require review.")
    if not risks:
        risks.append("No high-risk signal detected in the current window.")

    cards = [
        {
            "title": "Total analyzed today",
            "value": len(window),
            "subtitle": "Recent predictions captured in history",
        },
        {
            "title": "Negative ratio",
            "value": f"{round((negative_count / total) * 100, 1)}%",
            "subtitle": "Portion of recent signals labeled negative",
        },
        {
            "title": "Active alerts",
            "value": len(recent_alerts),
            "subtitle": "Triggered events in latest alert log",
        },
    ]

    actions = build_actionable_recommendations(
        {
            "positive": distribution.get("positive", 0),
            "neutral": distribution.get("neutral", 0),
            "negative": distribution.get("negative", 0),
        },
        topics,
        recent_alerts,
    )

    return jsonify(
        {
            "window_size": len(window),
            "distribution": {
                "positive": distribution.get("positive", 0),
                "neutral": distribution.get("neutral", 0),
                "negative": distribution.get("negative", 0),
            },
            "model_usage": dict(model_usage),
            "emotion_distribution": dict(emotion_distribution),
            "cards": cards,
            "top_movers": movers,
            "risk_indicators": risks,
            "recommendations": actions,
        }
    )


@app.get("/api/historical-trends")
def historical_trends():
    bucket = str(request.args.get("bucket", "day")).strip().lower()
    if bucket not in {"day", "hour"}:
        bucket = "day"
    window_size = request.args.get("window", default=1000, type=int)
    if window_size is None or window_size <= 0:
        window_size = 1000

    window = HISTORICAL_EVENTS[-window_size:] if HISTORICAL_EVENTS else []
    points = aggregate_historical_points(window, bucket)
    return jsonify({"bucket": bucket, "window_size": len(window), "points": points})


@app.get("/api/drilldown")
def drilldown():
    label = str(request.args.get("label", "")).strip().lower()
    topic = str(request.args.get("topic", "")).strip().lower()
    limit = request.args.get("limit", default=50, type=int)
    if limit is None or limit <= 0:
        limit = 50
    limit = min(limit, 200)

    filtered: list[dict] = []
    for event in reversed(HISTORICAL_EVENTS):
        event_label = str(event.get("label", "neutral")).lower()
        event_text = str(event.get("normalized_text", "")).lower()
        if label and event_label != label:
            continue
        if topic and topic not in event_text:
            continue
        filtered.append(
            {
                "timestamp": event.get("timestamp"),
                "label": event.get("label"),
                "model": event.get("model"),
                "emotion": event.get("emotion"),
                "confidence": event.get("confidence"),
                "text": event.get("normalized_text"),
                "source": event.get("source"),
            }
        )
        if len(filtered) >= limit:
            break

    return jsonify({"count": len(filtered), "results": filtered})


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    model = str(payload.get("model", "distilbert"))
    custom_lexicon = payload.get("custom_lexicon")

    if not text:
        return jsonify({"error": "Field 'text' is required."}), 400

    try:
        result = engine.predict(text=text, model=model, custom_lexicon=custom_lexicon)
        emotion = str((result.get("emotion") or {}).get("label", "neutral"))
        confidence = float(result.get("confidence", 0.5))
        record_prediction(
            label=result.get("label", "neutral"),
            model=model,
            emotion=emotion,
            confidence=confidence,
            normalized_text=str(result.get("normalized_text", "")),
            source="predict_text",
        )
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/compare")
def compare_models():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    custom_lexicon = payload.get("custom_lexicon")
    if not text:
        return jsonify({"error": "Field 'text' is required."}), 400

    models = ["tfidf_lr", "word2vec_svm", "distilbert"]
    results = [engine.predict(text=text, model=model, custom_lexicon=custom_lexicon) for model in models]
    best = max(results, key=lambda item: float(item.get("confidence", 0.0)))
    return jsonify({"text": text, "best_model": best.get("model"), "results": results})


@app.post("/api/simulate-scenario")
def simulate_scenario():
    payload = request.get_json(silent=True) or {}
    base_distribution = payload.get("distribution") or {}
    positive = float(base_distribution.get("positive", 0))
    neutral = float(base_distribution.get("neutral", 0))
    negative = float(base_distribution.get("negative", 0))
    delta_negative_pct = float(payload.get("delta_negative_pct", 0))

    adjusted_negative = max(0.0, negative * (1.0 + (delta_negative_pct / 100.0)))
    total_before = max(1.0, positive + neutral + negative)
    total_after = max(1.0, positive + neutral + adjusted_negative)

    return jsonify(
        {
            "before": {
                "positive": round(positive / total_before, 4),
                "neutral": round(neutral / total_before, 4),
                "negative": round(negative / total_before, 4),
            },
            "after": {
                "positive": round(positive / total_after, 4),
                "neutral": round(neutral / total_after, 4),
                "negative": round(adjusted_negative / total_after, 4),
            },
            "delta_negative_pct": delta_negative_pct,
        }
    )


@app.post("/api/predict-csv")
def predict_csv():
    model = request.form.get("model", "distilbert")
    campaign_start = request.form.get("campaign_start")
    if "file" not in request.files:
        return jsonify({"error": "CSV file is required in form-data with key 'file'."}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Please upload a .csv file."}), 400

    raw = file.read()
    dataframe = pd.read_csv(io.BytesIO(raw))

    normalized_columns = {
        str(column).strip().lstrip("\ufeff").lower(): column for column in dataframe.columns
    }
    text_column = resolve_column(normalized_columns, CSV_TEXT_COLUMN_ALIASES)
    location_column = resolve_column(normalized_columns, CSV_LOCATION_COLUMN_ALIASES)
    brand_column = resolve_column(normalized_columns, CSV_BRAND_COLUMN_ALIASES)
    demographic_column = resolve_column(normalized_columns, CSV_DEMOGRAPHIC_COLUMN_ALIASES)
    timestamp_column = resolve_column(normalized_columns, CSV_TIMESTAMP_COLUMN_ALIASES)

    if text_column is None:
        return (
            jsonify(
                {
                    "error": "CSV must include a text column.",
                    "accepted_column_names": sorted(CSV_TEXT_COLUMN_ALIASES),
                    "detected_columns": [str(column) for column in dataframe.columns],
                }
            ),
            400,
        )

    total_rows = len(dataframe)
    empty_text_rows = 0
    duplicate_text_rows = 0

    working = dataframe.copy()
    working[text_column] = working[text_column].fillna("").astype(str)
    deduped_rows = []
    seen = set()
    for _, row in working.iterrows():
        raw_text = str(row[text_column]).strip()
        if not raw_text:
            empty_text_rows += 1
            continue
        normalized_text = engine.predict(text=raw_text, model=model).get("normalized_text", "")
        if normalized_text in seen:
            duplicate_text_rows += 1
            continue
        seen.add(normalized_text)
        deduped_rows.append(row)

    predictions = []
    language_distribution_counter: Counter = Counter()
    for row in deduped_rows:
        value = str(row[text_column]).strip()
        if not value:
            continue
        prediction = engine.predict(text=value, model=model)
        predictions.append(prediction)
        emotion = str((prediction.get("emotion") or {}).get("label", "neutral"))
        confidence = float(prediction.get("confidence", 0.5))
        record_prediction(
            label=prediction.get("label", "neutral"),
            model=model,
            emotion=emotion,
            confidence=confidence,
            normalized_text=str(prediction.get("normalized_text", "")),
            source="predict_csv",
        )
        language_distribution_counter[str((prediction.get("language") or {}).get("label", "unknown"))] += 1

    distribution = dict(Counter([item["label"] for item in predictions]))
    emotion_distribution = dict(
        Counter([str((item.get("emotion") or {}).get("label", "neutral")) for item in predictions])
    )

    geo_summary = []
    if location_column is not None:
        geo_counter: dict[str, Counter] = {}
        non_empty_rows = dataframe[[text_column, location_column]].fillna("").astype(str)
        for _, row in non_empty_rows.iterrows():
            text_value = str(row[text_column]).strip()
            location_value = str(row[location_column]).strip()
            if not text_value or not location_value:
                continue

            prediction = engine.predict(text=text_value, model=model)
            location_key = location_value.title()
            if location_key not in geo_counter:
                geo_counter[location_key] = Counter()
            geo_counter[location_key][str(prediction.get("label", "neutral"))] += 1

        for place, counts in geo_counter.items():
            total = counts.get("positive", 0) + counts.get("neutral", 0) + counts.get("negative", 0)
            if total == 0:
                continue
            net_score = (counts.get("positive", 0) - counts.get("negative", 0)) / total
            geo_summary.append(
                {
                    "place": place,
                    "count": total,
                    "positive": counts.get("positive", 0),
                    "neutral": counts.get("neutral", 0),
                    "negative": counts.get("negative", 0),
                    "net_score": round(net_score, 3),
                }
            )

        geo_summary.sort(key=lambda item: item["count"], reverse=True)
        geo_summary = geo_summary[:20]

    topics = extract_context_topics(predictions)
    sarcasm_count = sum(1 for item in predictions if bool((item.get("sarcasm") or {}).get("detected", False)))
    sarcasm_rate = round((sarcasm_count / len(predictions)), 4) if predictions else 0.0
    toxic_rows = [item for item in predictions if str((item.get("toxicity") or {}).get("level", "none")) != "none"]
    toxicity_count = len(toxic_rows)
    avg_toxicity_score = (
        round(sum(float((item.get("toxicity") or {}).get("score", 0.0)) for item in predictions) / len(predictions), 4)
        if predictions
        else 0.0
    )

    competitor_benchmark = []
    if brand_column is not None:
        brand_counter: dict[str, Counter] = {}
        brand_rows = dataframe[[text_column, brand_column]].fillna("").astype(str)
        for _, row in brand_rows.iterrows():
            text_value = str(row[text_column]).strip()
            brand_value = str(row[brand_column]).strip()
            if not text_value or not brand_value:
                continue
            prediction = engine.predict(text=text_value, model=model)
            key = brand_value.title()
            if key not in brand_counter:
                brand_counter[key] = Counter()
            brand_counter[key][str(prediction.get("label", "neutral"))] += 1

        for brand, counts in brand_counter.items():
            total = counts.get("positive", 0) + counts.get("neutral", 0) + counts.get("negative", 0)
            if total <= 0:
                continue
            competitor_benchmark.append(
                {
                    "brand": brand,
                    "count": total,
                    "positive_rate": round(counts.get("positive", 0) / total, 4),
                    "negative_rate": round(counts.get("negative", 0) / total, 4),
                    "net_score": round((counts.get("positive", 0) - counts.get("negative", 0)) / total, 4),
                }
            )
        competitor_benchmark.sort(key=lambda item: item["count"], reverse=True)
        competitor_benchmark = competitor_benchmark[:10]

    campaign_impact = None
    if timestamp_column is not None:
        parsed = dataframe[[text_column, timestamp_column]].copy()
        parsed[timestamp_column] = pd.to_datetime(parsed[timestamp_column], errors="coerce", utc=True)
        split_time = pd.to_datetime(campaign_start, errors="coerce", utc=True) if campaign_start else None
        if split_time is not None and not pd.isna(split_time):
            before_counter: Counter = Counter()
            after_counter: Counter = Counter()
            for _, row in parsed.iterrows():
                text_value = str(row[text_column]).strip()
                timestamp_value = row[timestamp_column]
                if not text_value or pd.isna(timestamp_value):
                    continue
                prediction = engine.predict(text=text_value, model=model)
                if timestamp_value < split_time:
                    before_counter[str(prediction.get("label", "neutral"))] += 1
                else:
                    after_counter[str(prediction.get("label", "neutral"))] += 1

            before_total = sum(before_counter.values())
            after_total = sum(after_counter.values())
            campaign_impact = {
                "campaign_start": str(split_time),
                "before_count": before_total,
                "after_count": after_total,
                "before_positive_rate": round(before_counter.get("positive", 0) / before_total, 4) if before_total else 0.0,
                "after_positive_rate": round(after_counter.get("positive", 0) / after_total, 4) if after_total else 0.0,
                "before_negative_rate": round(before_counter.get("negative", 0) / before_total, 4) if before_total else 0.0,
                "after_negative_rate": round(after_counter.get("negative", 0) / after_total, 4) if after_total else 0.0,
            }

    fairness_audit = None
    if demographic_column is not None:
        group_rows = dataframe[[text_column, demographic_column]].fillna("").astype(str)
        group_stats: dict[str, dict[str, float]] = {}
        for _, row in group_rows.iterrows():
            text_value = str(row[text_column]).strip()
            group_value = str(row[demographic_column]).strip()
            if not text_value or not group_value:
                continue
            prediction = engine.predict(text=text_value, model=model)
            key = group_value.title()
            if key not in group_stats:
                group_stats[key] = {"count": 0, "positive": 0, "negative": 0, "confidence_sum": 0.0}
            group_stats[key]["count"] += 1
            if prediction.get("label") == "positive":
                group_stats[key]["positive"] += 1
            if prediction.get("label") == "negative":
                group_stats[key]["negative"] += 1
            group_stats[key]["confidence_sum"] += float(prediction.get("confidence", 0.0))

        groups = []
        positive_rates = []
        for key, stats in group_stats.items():
            count = max(1, int(stats["count"]))
            positive_rate = float(stats["positive"]) / count
            positive_rates.append(positive_rate)
            groups.append(
                {
                    "group": key,
                    "count": int(stats["count"]),
                    "positive_rate": round(positive_rate, 4),
                    "negative_rate": round(float(stats["negative"]) / count, 4),
                    "avg_confidence": round(float(stats["confidence_sum"]) / count, 4),
                }
            )
        disparity = (max(positive_rates) - min(positive_rates)) if positive_rates else 0.0
        fairness_audit = {"groups": groups, "positive_rate_disparity": round(disparity, 4)}

    quality_score = 1.0
    if total_rows > 0:
        penalty = (empty_text_rows + duplicate_text_rows) / total_rows
        quality_score = max(0.0, 1.0 - penalty)

    uncertainty_avg = (
        round(sum(float((item.get("uncertainty") or {}).get("score", 0.0)) for item in predictions) / len(predictions), 4)
        if predictions
        else 0.0
    )

    return jsonify(
        {
            "count": len(predictions),
            "distribution": distribution,
            "emotion_distribution": emotion_distribution,
            "language_distribution": dict(language_distribution_counter),
            "geo_summary": geo_summary,
            "geo_source_column": str(location_column) if location_column is not None else None,
            "topics": topics,
            "sarcasm_count": sarcasm_count,
            "sarcasm_rate": sarcasm_rate,
            "toxicity_count": toxicity_count,
            "avg_toxicity_score": avg_toxicity_score,
            "uncertainty_average": uncertainty_avg,
            "competitor_benchmark": competitor_benchmark,
            "campaign_impact": campaign_impact,
            "fairness_audit": fairness_audit,
            "data_quality": {
                "total_rows": total_rows,
                "retained_rows": len(predictions),
                "empty_text_rows": empty_text_rows,
                "duplicate_text_rows": duplicate_text_rows,
                "quality_score": round(quality_score, 4),
            },
            "cleaning_actions": {
                "trimmed_whitespace": True,
                "deduplicated_by_normalized_text": True,
            },
            "predictions": predictions,
        }
    )


@app.post("/api/generate-summary")
def generate_summary():
    payload = request.get_json(silent=True) or {}
    distribution = payload.get("distribution") or {}
    positive = int(distribution.get("positive", 0))
    neutral = int(distribution.get("neutral", 0))
    negative = int(distribution.get("negative", 0))
    total = max(1, positive + neutral + negative)

    top_label = max(
        [("positive", positive), ("neutral", neutral), ("negative", negative)],
        key=lambda pair: pair[1],
    )[0]
    summary = (
        f"Out of {total} analyzed posts, the dominant sentiment is {top_label}. "
        f"Positive share is {round((positive / total) * 100, 1)}%, "
        f"neutral share is {round((neutral / total) * 100, 1)}%, "
        f"and negative share is {round((negative / total) * 100, 1)}%."
    )
    return jsonify({"summary": summary})


@app.post("/api/save-snapshot")
def save_snapshot():
    payload = request.get_json(silent=True) or {}
    snapshot_id = str(uuid.uuid4())
    share_token = str(uuid.uuid4())[:8]
    SNAPSHOTS[snapshot_id] = {
        "id": snapshot_id,
        "role": str(payload.get("role", "analyst")),
        "title": str(payload.get("title", "Untitled Snapshot")),
        "data": payload.get("data") or {},
        "share_token": share_token,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_snapshots()
    return jsonify({"snapshot_id": snapshot_id, "share_token": share_token})


@app.get("/api/snapshots")
def list_snapshots():
    return jsonify({"snapshots": list(SNAPSHOTS.values())})


@app.get("/api/public-dashboard/<share_token>")
def public_dashboard(share_token: str):
    for snapshot in SNAPSHOTS.values():
        if snapshot.get("share_token") == share_token:
            return jsonify({"snapshot": snapshot})
    return jsonify({"error": "Share token not found."}), 404


@app.post("/api/export-report")
def export_report():
    payload = request.get_json(silent=True) or {}
    title = str(payload.get("title", "Sentiment Report"))
    distribution = payload.get("distribution") or {}
    topics = payload.get("topics") or []
    positive = int(distribution.get("positive", 0))
    neutral = int(distribution.get("neutral", 0))
    negative = int(distribution.get("negative", 0))
    total = max(1, positive + neutral + negative)

    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    pdf_buffer = io.BytesIO()
    pdf = canvas.Canvas(pdf_buffer, pagesize=letter)
    width, height = letter

    y = height - 72
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(72, y, title)
    y -= 32

    pdf.setFont("Helvetica", 12)
    pdf.drawString(72, y, f"Generated: {datetime.now(timezone.utc).isoformat()}")
    y -= 24

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(72, y, "Sentiment Distribution")
    y -= 20

    pdf.setFont("Helvetica", 12)
    pdf.drawString(72, y, f"Positive: {positive} ({round((positive / total) * 100, 1)}%)")
    y -= 18
    pdf.drawString(72, y, f"Neutral: {neutral} ({round((neutral / total) * 100, 1)}%)")
    y -= 18
    pdf.drawString(72, y, f"Negative: {negative} ({round((negative / total) * 100, 1)}%)")
    y -= 26

    # Distribution bar chart
    chart_x = 72
    chart_y = y - 120
    chart_w = 300
    chart_h = 95
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(chart_x, chart_y + chart_h + 18, "Sentiment Bar Chart")
    pdf.line(chart_x, chart_y, chart_x + chart_w, chart_y)
    bars = [
        ("Positive", positive, (0.12, 0.62, 0.4)),
        ("Neutral", neutral, (0.69, 0.54, 0.0)),
        ("Negative", negative, (0.76, 0.2, 0.2)),
    ]
    max_value = max(1, positive, neutral, negative)
    bar_width = 58
    gap = 34
    for index, (label, value, color) in enumerate(bars):
        x = chart_x + 20 + index * (bar_width + gap)
        bar_height = int((value / max_value) * (chart_h - 10))
        pdf.setFillColorRGB(*color)
        pdf.rect(x, chart_y, bar_width, bar_height, fill=1, stroke=0)
        pdf.setFillColorRGB(0.15, 0.15, 0.15)
        pdf.setFont("Helvetica", 10)
        pdf.drawCentredString(x + (bar_width / 2), chart_y - 14, label)
        pdf.drawCentredString(x + (bar_width / 2), chart_y + bar_height + 4, str(value))

    y = chart_y - 36

    top_label = max(
        [("positive", positive), ("neutral", neutral), ("negative", negative)],
        key=lambda item: item[1],
    )[0]
    insight = f"Dominant sentiment is {top_label}."
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(72, y, "Executive Insight")
    y -= 20
    pdf.setFont("Helvetica", 12)
    pdf.drawString(72, y, insight)
    y -= 26

    # Topic chart
    clean_topics = [item for item in topics if isinstance(item, dict) and item.get("topic")][:5]
    if clean_topics:
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(72, y, "Top Context Topics")
        y -= 14
        topic_max = max(int(item.get("count", 0)) for item in clean_topics) if clean_topics else 1
        for item in clean_topics:
            topic_label = str(item.get("topic", ""))
            topic_count = int(item.get("count", 0))
            bar_len = int((topic_count / max(1, topic_max)) * 220)
            pdf.setFont("Helvetica", 10)
            pdf.setFillColorRGB(0.2, 0.2, 0.2)
            pdf.drawString(72, y, f"{topic_label} ({topic_count})")
            y -= 10
            pdf.setFillColorRGB(0.24, 0.47, 0.7)
            pdf.rect(72, y, bar_len, 6, fill=1, stroke=0)
            y -= 12

    pdf.showPage()
    pdf.save()
    pdf_bytes = pdf_buffer.getvalue()
    encoded = base64.b64encode(pdf_bytes).decode("ascii")

    return jsonify(
        {
            "format": "pdf-base64",
            "filename": "sentiment-report.pdf",
            "content_base64": encoded,
            "size_bytes": len(pdf_bytes),
        }
    )


load_snapshots()
load_watchlists()
load_report_schedules()
load_delivery_log()
load_alert_rules()
load_alert_events()
load_historical_events()
start_schedule_worker_once()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
