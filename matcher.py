import os
import sys
import csv
import logging
import requests
import datetime
import argparse
import re
from difflib import SequenceMatcher
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
ZOHO_CLIENT_ID        = os.getenv("ZOHO_CLIENT_ID")
ZOHO_CLIENT_SECRET    = os.getenv("ZOHO_CLIENT_SECRET")
ZOHO_REFRESH_TOKEN    = os.getenv("ZOHO_REFRESH_TOKEN")

ACCOUNTS_BASE         = os.getenv("ZOHO_ACCOUNTS_BASE", "https://accounts.zoho.com")
API_BASE              = os.getenv("ZOHO_API_BASE",      "https://www.zohoapis.com")

RESULTS_MODULE        = os.getenv("RESULTS_MODULE",        "Results_2025")
SELECTIONS_MODULE     = os.getenv("SELECTIONS_MODULE",     "Random_Selections1")
SELECTION_PROFILE_MOD = os.getenv("SELECTION_PROFILE_MOD", "Random_Selection_Profiles")
PARTICIPANTS_MODULE   = os.getenv("PARTICIPANTS_MODULE",   "Participants")

SUBFORM_NAME          = os.getenv("SUBFORM_NAME", "Selected_Participants")
FIELD_BAT             = os.getenv("FIELD_BAT",    "BAT")
FIELD_CCFID           = os.getenv("FIELD_CCFID",  "CCFID")
FIELD_COMPLETE        = os.getenv("FIELD_COMPLETE","Completed")  # ← new


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

# ─── ZOHO CLIENT ────────────────────────────────────────────────────────────────
class ZohoClient:
    def __init__(self):
        for k, v in [("ZOHO_CLIENT_ID", ZOHO_CLIENT_ID), ("ZOHO_CLIENT_SECRET", ZOHO_CLIENT_SECRET), ("ZOHO_REFRESH_TOKEN", ZOHO_REFRESH_TOKEN)]:
            if not v:
                logger.error(f"Missing {k}. Set it in environment variables.")
                sys.exit(1)
        self._token = os.getenv("ZCRM_ACCESS_TOKEN")
        if not self._token:
            self._refresh()

    def _refresh(self):
        url = f"{ACCOUNTS_BASE}/oauth/v2/token"
        params = {
            "refresh_token": ZOHO_REFRESH_TOKEN,
            "client_id":     ZOHO_CLIENT_ID,
            "client_secret": ZOHO_CLIENT_SECRET,
            "grant_type":    "refresh_token"
        }
        r = requests.post(url, params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        self._token = data["access_token"]

    def headers(self):
        return {"Authorization": f"Zoho-oauthtoken {self._token}"}

    def _ensure_json(self, r, url):
        if r.status_code == 204:
            raise LookupError(f"No content from {url} (204)")
        ct = r.headers.get("Content-Type", "")
        if "json" not in ct.lower():
            # Raise with a peek of the body for debugging (capped to 200 chars)
            raise RuntimeError(f"Unexpected content type {ct} from {url}: {r.text[:200]}")

    def get_record(self, module, record_id):
        url = f"{API_BASE}/crm/v2/{module}/{record_id}"
        r = requests.get(url, headers=self.headers(), timeout=HTTP_TIMEOUT)
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
        r = requests.get(url, headers=self.headers(), params=params, timeout=HTTP_TIMEOUT)
        if r.status_code == 204:
            return []
        r.raise_for_status()
        self._ensure_json(r, url)
        return r.json().get("data", []) or []

    def update(self, module, record_id, payload):
        url  = f"{API_BASE}/crm/v8/{module}/{record_id}"
        body = {"data": [payload]}
        r = requests.put(
            url, headers={**self.headers(), "Content-Type": "application/json"},
            json=body, timeout=HTTP_TIMEOUT
        )
        r.raise_for_status()
        self._ensure_json(r, url)
        return r.json()

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

    # 3) BOOTSTRAP selections (trimmed; same logic as before)
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
                except:
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

                        payload = {"id": row_id, update_f: rid}

                        raw_tests = row.get("Test_For", "")
                        if isinstance(raw_tests, list):
                            tests = [str(t).strip().lower() for t in raw_tests if t]
                        else:
                            tests = [t.strip().lower() for t in str(raw_tests).split(";") if t.strip()]

                        # which fields must be filled
                        required = []
                        if "drug" in tests:
                            required.append(FIELD_CCFID)
                        if "alcohol" in tests:
                            required.append(FIELD_BAT)

                        # simulate the row after this update
                        simulated = {**row, **payload}

                        # boolean checkbox instead of "State"
                        is_complete = bool(required) and all(simulated.get(f) for f in required)

                        # only send the checkbox if it changes (optional noise reduction)
                        current_complete = bool(row.get(FIELD_COMPLETE))
                        if current_complete != is_complete:
                            payload[FIELD_COMPLETE] = is_complete

                        # push just this subform row
                        client.update(SELECTIONS_MODULE, sel["id"], { SUBFORM_NAME: [payload] })
                        row.update(payload)

                        matched.add(rid)
                        logger.info(
                            f"→ Matched {rid} → sel {sel['id']} "
                            f"(row_id={row_id}, idx={idx}, score={best:.2f})"
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
