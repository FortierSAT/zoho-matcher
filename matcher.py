import os
import sys
import csv
import logging
import requests
import datetime
import argparse
import re
import json, time, random
from contextlib import contextmanager
from difflib import SequenceMatcher
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter

load_dotenv()

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
def _clean_base(v: str, default: str) -> str:
    return (v or default).strip().strip('"').strip("'").rstrip('/')

ZOHO_CLIENT_ID        = os.getenv("ZOHO_CLIENT_ID")
ZOHO_CLIENT_SECRET    = os.getenv("ZOHO_CLIENT_SECRET")
ZOHO_REFRESH_TOKEN    = os.getenv("ZOHO_REFRESH_TOKEN")

ACCOUNTS_BASE         = _clean_base(os.getenv("ZOHO_ACCOUNTS_BASE"), "https://accounts.zoho.com")
API_BASE              = _clean_base(os.getenv("ZOHO_API_BASE"),      "https://www.zohoapis.com")

RESULTS_MODULE        = os.getenv("RESULTS_MODULE",        "Results_2025")
SELECTIONS_MODULE     = os.getenv("SELECTIONS_MODULE",     "Random_Selections1")
SELECTION_PROFILE_MOD = os.getenv("SELECTION_PROFILE_MOD", "Random_Selection_Profiles")
PARTICIPANTS_MODULE   = os.getenv("PARTICIPANTS_MODULE",   "Participants")

SUBFORM_NAME          = os.getenv("SUBFORM_NAME", "Selected_Participants")
FIELD_BAT             = os.getenv("FIELD_BAT",    "BAT")
FIELD_CCFID           = os.getenv("FIELD_CCFID",  "CCFID")
FIELD_COMPLETE        = os.getenv("FIELD_COMPLETE","Completed")   # ← subform checkbox API name
COMPLETE_SCOPE        = os.getenv("COMPLETE_SCOPE", "subform").strip().lower()  # 'subform' (default) or 'parent'

FUZZY_RATIO           = float(os.getenv("FUZZY_RATIO", "0.80"))
PAGE_SIZE             = int(os.getenv("PAGE_SIZE", "200"))
LOCK_DIR              = os.getenv("LOCK_DIR", "/tmp/matcher-locks")
HTTP_TIMEOUT          = float(os.getenv("HTTP_TIMEOUT", "20"))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ─── TOKEN CACHE / REFRESH SETTINGS ─────────────────────────────────────────────
TOKEN_CACHE_PATH   = os.getenv("ZOHO_TOKEN_CACHE", "/tmp/zoho_crm_token.json")
REFRESH_LOCK_DIR   = os.getenv("ZOHO_REFRESH_LOCK_DIR", "/tmp")
REFRESH_LOCK_PATH  = os.path.join(REFRESH_LOCK_DIR, "zoho_token_refresh.lock")

RETRY_MAX_TRIES     = int(os.getenv("HTTP_RETRY_MAX_TRIES", "6"))
RETRY_BASE_SECONDS  = float(os.getenv("HTTP_RETRY_BASE_SECONDS", "0.5"))
RETRY_MAX_SECONDS   = float(os.getenv("HTTP_RETRY_MAX_SECONDS", "8.0"))
RETRY_STATUSES      = {429, 500, 502, 503, 504, 520, 522, 524}
CLOCK_SKEW_PAD_SECS = 30  # refresh a little early to avoid edge-expiry

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def _now() -> int:
    return int(time.time())

@contextmanager
def _file_lock(path: str, timeout: float = 20.0, poll: float = 0.1):
    """Crude inter-process lock via O_EXCL file creation."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    start = time.time()
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            try:
                yield
            finally:
                try: os.unlink(path)
                except FileNotFoundError: pass
            return
        except FileExistsError:
            if (time.time() - start) > timeout:
                raise TimeoutError(f"Could not acquire lock {path} in {timeout}s")
            time.sleep(poll)

def _read_token_cache():
    try:
        with open(TOKEN_CACHE_PATH, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except FileNotFoundError:
        return None
    except Exception:
        logger.exception("Failed reading token cache")
        return None

def _write_token_cache(token: str, expires_in: int | float | None):
    os.makedirs(os.path.dirname(TOKEN_CACHE_PATH), exist_ok=True)
    ttl = int(expires_in or 3300)  # Zoho often gives 3600; use 55m if absent
    payload = {"access_token": token, "expires_at": _now() + ttl}
    tmp = TOKEN_CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, TOKEN_CACHE_PATH)

def _sleep_backoff(attempt: int):
    base = min(RETRY_MAX_SECONDS, RETRY_BASE_SECONDS * (2 ** (attempt - 1)))
    time.sleep(base * (0.5 + random.random()))  # 0.5x–1.5x jitter

# ─── ZOHO CLIENT (robust) ──────────────────────────────────────────────────────
class ZohoClient:
    def __init__(self):
        for k, v in [("ZOHO_CLIENT_ID", ZOHO_CLIENT_ID), ("ZOHO_CLIENT_SECRET", ZOHO_CLIENT_SECRET), ("ZOHO_REFRESH_TOKEN", ZOHO_REFRESH_TOKEN)]:
            if not v:
                logger.error(f"Missing {k}. Set it in environment variables.")
                sys.exit(1)
        # Session with connection pooling
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=0)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self._token, self._expires_at = self._load_or_refresh_token()

    # ——— Token logic ——————————————————————————————————————————————
    def _load_or_refresh_token(self):
        cached = _read_token_cache()
        if cached and cached.get("access_token") and cached.get("expires_at", 0) > (_now() + CLOCK_SKEW_PAD_SECS):
            return cached["access_token"], cached["expires_at"]

        # Single-flight refresh via lock
        try:
            with _file_lock(REFRESH_LOCK_PATH, timeout=25.0, poll=0.15):
                cached2 = _read_token_cache()
                if cached2 and cached2.get("access_token") and cached2.get("expires_at", 0) > (_now() + CLOCK_SKEW_PAD_SECS):
                    return cached2["access_token"], cached2["expires_at"]
                token, expires_at = self._do_refresh()
                return token, expires_at
        except TimeoutError:
            # Couldn’t get the lock (busy); wait briefly and read cache again
            time.sleep(1.0)
            cached3 = _read_token_cache()
            if cached3 and cached3.get("access_token") and cached3.get("expires_at", 0) > (_now() + CLOCK_SKEW_PAD_SECS):
                return cached3["access_token"], cached3["expires_at"]
            token, expires_at = self._do_refresh()
            return token, expires_at

    def _do_refresh(self):
        url = f"{ACCOUNTS_BASE}/oauth/v2/token"
        data = {
            "refresh_token": ZOHO_REFRESH_TOKEN,
            "client_id":     ZOHO_CLIENT_ID,
            "client_secret": ZOHO_CLIENT_SECRET,
            "grant_type":    "refresh_token",
        }
        for attempt in range(1, RETRY_MAX_TRIES + 1):
            r = self.session.post(url, data=data, timeout=HTTP_TIMEOUT)
            if r.status_code in RETRY_STATUSES:
                logger.warning("Zoho refresh throttled (%s). attempt=%s body=%s", r.status_code, attempt, r.text[:200])
                if attempt == RETRY_MAX_TRIES:
                    r.raise_for_status()
                _sleep_backoff(attempt)
                continue
            try:
                r.raise_for_status()
            except requests.HTTPError:
                logger.error("Zoho token refresh failed %s: %s", r.status_code, r.text[:500])
                raise
            try:
                j = r.json()
            except ValueError:
                logger.error("Non-JSON token response: %s", r.text[:500])
                r.raise_for_status()
                raise
            token = j["access_token"]
            expires_in = j.get("expires_in") or j.get("expires_in_sec")
            _write_token_cache(token, expires_in)
            return token, _read_token_cache()["expires_at"]
        raise RuntimeError("Unreachable: refresh retry loop exhausted")

    def _ensure_fresh(self):
        if self._expires_at <= (_now() + CLOCK_SKEW_PAD_SECS):
            self._token, self._expires_at = self._load_or_refresh_token()

    def headers(self):
        self._ensure_fresh()
        return {"Authorization": f"Zoho-oauthtoken {self._token}"}

    def _ensure_json(self, r, url):
        if r.status_code == 204:
            raise LookupError(f"No content from {url} (204)")
        ct = r.headers.get("Content-Type", "")
        if "json" not in (ct or "").lower():
            raise RuntimeError(f"Unexpected content type {ct} from {url}: {r.text[:200]}")

    # ——— Robust request with retries + auto-refresh on 401 ————————
    def _request(self, method, url, **kw):
        if "headers" not in kw or kw["headers"] is None:
            kw["headers"] = self.headers()
        else:
            # Merge in fresh Authorization without dropping existing headers
            kw["headers"] = {**kw["headers"], **self.headers()}

        for attempt in range(1, RETRY_MAX_TRIES + 1):
            r = self.session.request(method, url, timeout=HTTP_TIMEOUT, **kw)

            # Token expired/invalid: refresh then retry once immediately
            if r.status_code == 401:
                logger.info("401 received; refreshing token and retrying once")
                self._token, self._expires_at = self._load_or_refresh_token()
                current = kw.get("headers") or {}
                kw["headers"] = {**current, **self.headers()}  # preserve Content-Type etc.
                r = self.session.request(method, url, timeout=HTTP_TIMEOUT, **kw)

            if r.status_code in RETRY_STATUSES:
                if attempt == RETRY_MAX_TRIES:
                    return r
                logger.warning("Retryable status %s from %s (attempt %s)", r.status_code, url, attempt)
                _sleep_backoff(attempt)
                continue

            return r

        return r  # last response

    # ——— Public API ————————————————————————————————————————————————
    def get_record(self, module, record_id):
        url = f"{API_BASE}/crm/v2/{module}/{record_id}"
        r = self._request("GET", url)
        if r.status_code == 404:
            raise LookupError(f"{module} record not found: {record_id}")
        r.raise_for_status()
        self._ensure_json(r, url)
        data = r.json()
        if "data" not in data or not data["data"]:
            raise LookupError(f"No data for {module} record {record_id}")
        return data["data"][0]

    def search(self, module, criteria, page=1):
        url    = f"{API_BASE}/crm/v2/{module}/search"
        params = {"criteria": criteria, "page": page, "per_page": PAGE_SIZE}
        r = self._request("GET", url, params=params)
        if r.status_code == 204:
            return []
        r.raise_for_status()
        self._ensure_json(r, url)
        return r.json().get("data", []) or []

    def update(self, module, record_id, payload):
        url  = f"{API_BASE}/crm/v8/{module}/{record_id}"
        body = {"data": [payload]}
        hdrs = {**self.headers(), "Content-Type": "application/json"}
        r = self._request("PUT", url, json=body, headers=hdrs)
        r.raise_for_status()
        self._ensure_json(r, url)
        data = r.json()
        try:
            info = data["data"][0]
            if info.get("status") != "success":
                logger.error("Update failed for %s %s: %s", module, record_id, info)
        except Exception:
            logger.info("Update response (raw): %s", data)
        return data

# ─── UTILITIES ─────────────────────────────────────────────────────────────────
def fuzzy_match(a, b):
    return SequenceMatcher(None, a, b).ratio() if a and b else 0

def get_period_for_date(date: datetime.date, frequency: str):
    m = date.month
    if frequency == "Quarterly":
        return ["Q1","Q2","Q3","Q4"][(m-1)//3]
    if frequency == "Monthly":
        return date.strftime("%b")
    if frequency == "Semi-Annually":
        return "SA1" if m <= 6 else "SA2"
    if frequency == "Annually":
        return "A1"
    return None

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

def normalize_name_part(name: str):
    n = name.lower()
    n = re.sub(r"[.,]", " ", n)
    for suf in SUFFIXES:
        n = re.sub(rf"\b{suf}\b", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    parts = [p for p in n.split(" ") if len(p) > 1]
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return (parts[0], "") if parts else ("", "")

def name_orders(fn, ln):
    return [(fn, ln), (ln, fn)]

def suffix_match(id1, id2):
    if isinstance(id1, dict): id1 = id1.get("id","")
    if isinstance(id2, dict): id2 = id2.get("id","")
    return bool(id1 and id2) and id1[-3:] == id2[-3:]

def _acquire_lock(record_id: str) -> bool:
    try:
        os.makedirs(LOCK_DIR, exist_ok=True)
        lock_path = os.path.join(LOCK_DIR, f"{record_id}.lock")
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False

def _release_lock(record_id: str):
    try:
        os.remove(os.path.join(LOCK_DIR, f"{record_id}.lock"))
    except FileNotFoundError:
        pass

def _as_bool(v):
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return bool(v)
    if isinstance(v, str): return v.strip().lower() in {"1","true","yes","on"}
    return False

# ─── CORE WORK ──────────────────────────────────────────────────────────────────
def process_results(result_id: str = None, accounts=None, limit: int | None = None):
    client = ZohoClient()
    debug_rows = []

    # 1) FETCH results
    if result_id:
        results = [ client.get_record(RESULTS_MODULE, result_id) ]
    else:
        crit = "(Test_Reason:equals:Random)"
        if accounts:
            accs = " or ".join(f"(Company:equals:{a})" for a in accounts)
            crit = f"({crit} and ({accs}))"
        results, page = [], 1
        while True:
            batch = client.search(RESULTS_MODULE, crit, page)
            if not batch:
                break
            results.extend(batch)
            if limit and len(results) >= limit:
                results = results[:limit]
                break
            page += 1

    logger.info(f"Fetched {len(results)} result(s)")

    # 2) PREFETCH selections & participants
    account_to_sels = {}
    for rec in results:
        acct = rec.get("Company", {}).get("id")
        if not acct or acct in account_to_sels:
            continue
        meta = client.search(SELECTIONS_MODULE, f"(Account:equals:{acct})")
        account_to_sels[acct] = [client.get_record(SELECTIONS_MODULE, s["id"]) for s in meta]

    participants_by_account = {}
    for acct, sels in account_to_sels.items():
        crit_p, parts, pg = f"(Account:equals:{acct})", [], 1
        while True:
            batch = client.search(PARTICIPANTS_MODULE, crit_p, pg)
            if not batch:
                break
            parts.extend(batch); pg += 1
        lookup = []
        for p in parts:
            fn, ln = normalize_name_part(f"{p.get('First_Name','')} {p.get('Last_Name','')}")
            lookup.append({"id": p["id"], "first": fn, "last": ln})
        participants_by_account[acct] = lookup

    # 3) BOOTSTRAP selections (unchanged logic, minor hygiene)
    bootstrapped = set()
    for sels in account_to_sels.values():
        for sel in sels:
            sid  = sel["id"]
            acct = sel.get("Account",{}).get("id")
            if sid in bootstrapped:
                continue
            payload, need = {}, False
            pid  = sel.get("Random_Profile",{}).get("id")
            freq = sel.get("Selection_Frequency") or (
                pid and client.get_record(SELECTION_PROFILE_MOD, pid).get("Selection_Frequency") or ""
            )
            freq = (freq or "").strip()
            if freq and not sel.get("Selection_Frequency"):
                payload["Selection_Frequency"] = freq; need = True
            if sel.get("Selection_Date"):
                sd  = sel["Selection_Date"][:10]
                dt  = datetime.datetime.strptime(sd, "%Y-%m-%d").date()
                per = get_period_for_date(dt, payload.get("Selection_Frequency", sel.get("Selection_Frequency")))
                if per and not sel.get("Selection_Period"):
                    payload["Selection_Period"] = per; need = True

            company_text = sel.get("Account",{}).get("name","")
            acct_parts   = participants_by_account.get(acct, [])
            sub          = sel.get(SUBFORM_NAME,[]) or []
            for row in sub:
                if not row.get("Donor_ID"):
                    full = f"{row.get('First_Name','')} {row.get('Last_Name','')}"
                    rfn, rln = normalize_name_part(full)
                    best_score, best_id = 0.0, None
                    for p in acct_parts:
                        for cand_fn, cand_ln in name_orders(p["first"], p["last"]):
                            for test_fn, test_ln in name_orders(rfn, rln):
                                score = (fuzzy_match(test_fn, cand_fn) + fuzzy_match(test_ln, cand_ln)) / 2
                                if score > best_score:
                                    best_score, best_id = score, p["id"]
                    if best_id and best_score >= FUZZY_RATIO:
                        row["Donor_ID"] = best_id; need = True
                        logger.info(f"  → bootstrapped Donor_ID for '{full}' = {best_id} (score {best_score:.2f})")

                # double-check the API name you intend to write here
                if not row.get("Company_Name") and company_text:
                    row["Company"] = company_text; need = True

            if need:
                client.update(SELECTIONS_MODULE, sid, {**payload, SUBFORM_NAME: sub})
            bootstrapped.add(sid)

    # 4) MATCHING pass
    failures, matched = [], set()
    for rec in results:
        rid = rec["id"]
        if not _acquire_lock(rid):
            logger.info(f"Skipping {rid}: already processing")
            continue
        try:
            if rid in matched:
                continue
            acct     = rec.get("Company",{}).get("id")
            sels     = account_to_sels.get(acct, [])
            col_date = datetime.datetime.strptime(rec.get("Collection_Date","")[:10], "%Y-%m-%d").date()
            update_f = FIELD_BAT if "BAT" in (rec.get("Name") or "").upper() else FIELD_CCFID
            rec_pid  = rec.get("Primary_ID","")

            full = []
            for s in sels:
                sd = s.get("Selection_Date","")[:10]
                try:
                    sdt = datetime.datetime.strptime(sd, "%Y-%m-%d").date()
                except Exception:
                    sdt = None
                full.append({"record": s, "date": sdt, "period": s.get("Selection_Period"), "sel_date": sd})

            freq   = sels[0].get("Selection_Frequency","") if sels else ""
            expect = get_period_for_date(col_date, freq)
            mset = [f for f in full if f["period"] == expect]
            fset = [f for f in full if f["period"] != expect]
            mset.sort(key=lambda x: x["date"] or datetime.date.min, reverse=True)
            fset.sort(key=lambda x: x["date"] or datetime.date.min, reverse=True)

            found = False
            for bucket in (mset, fset):
                if found: break
                for item in bucket:
                    sel    = item["record"]
                    sel_dt = item["date"]
                    sel_date_str = item["sel_date"]
                    row_period   = item["period"]

                    if sel_dt and sel_dt > col_date:
                        continue

                    sub = sel.get(SUBFORM_NAME,[]) or []
                    for idx, row in enumerate(sub):
                        row_id   = row.get("id")
                        full_row = f"{row.get('First_Name','')} {row.get('Last_Name','')}"
                        rfn, rln = normalize_name_part(full_row)
                        rec_fn, rec_ln = normalize_name_part(
                            f"{rec.get('First_Name','')} {rec.get('Last_Name','')}"
                        )

                        best = 0.0
                        for a_fn, a_ln in name_orders(rfn, rln):
                            for b_fn, b_ln in name_orders(rec_fn, rec_ln):
                                score = (fuzzy_match(a_fn, b_fn) + fuzzy_match(a_ln, b_ln)) / 2
                                best = max(best, score)

                        donor_field = row.get("Donor_ID","")
                        if isinstance(donor_field, dict):
                            donor_text = donor_field.get("name") or donor_field.get("lookup_label","")
                        else:
                            donor_text = donor_field

                        suffix_flag = bool(best < FUZZY_RATIO and suffix_match(rec_pid, donor_text))
                        ok = (best >= FUZZY_RATIO) or suffix_flag

                        debug_rows.append({
                            "Result_ID": rid, "Selection_ID": sel["id"], "Row_ID": row_id, "Row_Index": idx,
                            "Collection_Date": col_date.isoformat(), "Selection_Date": sel_date_str,
                            "Expected_Period": expect, "Row_Period": row_period,
                            "Primary_ID": rec_pid, "Donor_ID": donor_text,
                            "Fuzzy_Score": f"{best:.3f}", "Suffix_Flag": suffix_flag,
                            "Matched": ok, "Update_Field": update_f, "Subform_Name": full_row,
                        })

                        if not ok:
                            continue

                        # Build per-row payload
                        payload = {"id": row_id, update_f: rid}

                        # determine required fields from Test_For (list or ';' string)
                        raw_tests = row.get("Test_For", "")
                        if isinstance(raw_tests, list):
                            tests = [str(t).strip().lower() for t in raw_tests if t]
                        else:
                            tests = [t.strip().lower() for t in str(raw_tests).split(";") if t.strip()]

                        required = []
                        if "drug" in tests:    required.append(FIELD_CCFID)
                        if "alcohol" in tests: required.append(FIELD_BAT)

                        # simulate after this update to compute completion
                        simulated = {**row, **payload}
                        is_complete = bool(required) and all(simulated.get(f) for f in required)

                        if COMPLETE_SCOPE == "parent":
                            # parent-level checkbox (rare): update on the selection record itself
                            client.update(SELECTIONS_MODULE, sel["id"], {FIELD_COMPLETE: bool(is_complete)})
                        else:
                            # subform-level checkbox (default): set true/false explicitly
                            payload[FIELD_COMPLETE] = bool(is_complete)
                            resp = client.update(SELECTIONS_MODULE, sel["id"], { SUBFORM_NAME: [payload] })
                            # optional: surface per-row errors clearly
                            try:
                                info = resp["data"][0]
                                if info.get("status") != "success":
                                    logger.error("Subform update failed for sel %s row %s: %s", sel["id"], row_id, info)
                            except Exception:
                                pass

                            # keep in-memory row in sync for later checks
                            row.update(payload)

                        matched.add(rid)
                        logger.info(
                            f"→ Matched {rid} → sel {sel['id']} "
                            f"(row_id={row_id}, idx={idx}, score={best:.2f}, complete={bool(is_complete)})"
                        )
                        found = True
                        break
                    if found: break
                if found: break

            if not found:
                failures.append((rid, "no match"))
        finally:
            _release_lock(rid)

# ─── CLI ENTRY ──────────────────────────────────────────────────────────────────
def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--id", help="Only this result_id")
    p.add_argument("--limit", type=int, help="Stop after N results")
    p.add_argument("--accounts", nargs="+", help="Only these Account IDs")
    args = p.parse_args()
    process_results(result_id=args.id, accounts=args.accounts, limit=args.limit)

if __name__ == "__main__":
    try:
        _cli()
    except requests.exceptions.HTTPError as e:
        logger.error(f"API error: {e}")
        sys.exit(1)
    except Exception:
        logger.error("Unexpected error:", exc_info=True)
        sys.exit(1)
