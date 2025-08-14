import os
import logging
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, abort
from werkzeug.exceptions import BadRequest

import matcher  # the matching engine

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
WORKERS = int(os.getenv("JOB_WORKERS", "2"))

# Keep a small pool so we don't overwhelm APIs
EXECUTOR = ThreadPoolExecutor(max_workers=WORKERS)

app = Flask(__name__)

@app.get("/")
def health():
    return {"ok": True, "service": "zoho-matcher"}, 200

def _run_job(record_id: str):
    try:
        matcher.process_results(result_id=record_id)
    except Exception:
        log.exception(f"Job failed for record {record_id}")

@app.post("/webhooks/zoho/results-created")
def results_created():
    # Token check (querystring ?token=, header, or body)
    token = (
        request.args.get("token")
        or request.headers.get("X-Webhook-Token")
        or ((request.get_json(silent=True) or {}).get("token"))
        or (request.form.get("token"))
    )
    if WEBHOOK_SECRET and token != WEBHOOK_SECRET:
        abort(401, description="Invalid token")

    payload = request.get_json(silent=True) or request.form.to_dict() or {}
    record_id = payload.get("id") or payload.get("record_id") or payload.get("EntityId")
    if not record_id:
        raise BadRequest("Missing record id (expected 'id', 'record_id', or 'EntityId')")

    EXECUTOR.submit(_run_job, record_id)
    return jsonify({"ok": True, "received_id": record_id})
