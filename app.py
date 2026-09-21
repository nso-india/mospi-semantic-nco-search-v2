from flask import Flask, g, request, jsonify, render_template, make_response, send_file
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
import io
import os, json, re, threading, hashlib
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from datetime import datetime, timedelta
import difflib
import logging
from dotenv import load_dotenv
import csv
from io import StringIO
from metadata_manager import MetadataConflictError, MetadataManager
from analytics import AnalyticsEngine, AnalyticsMongoUnavailableError
from evaluation_runner import EvaluationRunner, EvaluationValidationError
from evaluation_history_manager import EvaluationHistoryManager, EvaluationHistoryUnavailableError
from metadata_job_manager import MetadataJobManager, MetadataJobUnavailableError
from production_migration import (
    ProductionMigrationConfigurationError,
    ProductionMigrationError,
    ProductionMigrationManager,
    ProductionMigrationPermissionError,
)
from auth_manager import (
    AuthenticationError,
    AuthenticationUnavailableError,
    AuthManager,
    AuthorizationError,
    ROLES,
)
from security import require_login, require_roles, require_search_api_key

# Load environment variables
load_dotenv()

# Configure logging (Fix #4: Prevent source code disclosure via tracebacks)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================================
# CONFIG
# ================================
# Qdrant is the primary vector database. FAISS is a fallback using

# still work if Qdrant becomes unavailable.
USE_QDRANT = True
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except Exception as exc:
    USE_QDRANT = False
    QdrantClient = None
    qmodels = None
    logger.warning("Qdrant client is unavailable; FAISS fallback will be used: %s", exc)

# ================================
# LLM (QUERY REWRITER ONLY)
# ================================
from openai import OpenAI

# LLM Configuration from environment variables
LLM_BASE_URL = os.environ.get('LLM_BASE_URL', "http://10.75.8.2:8002/v1")
LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', "Qwen/Qwen3.6-27B-FP8")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "EMPTY")
LLM_TIMEOUT = 30  # seconds

try:
    rewriter_llm = OpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,  # vLLM doesn't require real API key
        timeout=LLM_TIMEOUT
    )

    rewriter_llm.models.list()
    LLM_IS_RUNNING = True
    logger.info("LLM is running: %s", LLM_MODEL_NAME)
except Exception as e:
    LLM_IS_RUNNING = False
    # logger.warning(f"✗ GPT-OSS-20B is not running: {e}")
    logger.warning("LLM is unavailable (%s): %s", LLM_MODEL_NAME, e)


def call_qwen(
    messages,
    *,
    model=LLM_MODEL_NAME,
    max_tokens=4096,
    temperature=0.3,
    top_p=0.8,
    top_k=20,
    **kwargs,
):
    """Call Qwen through the OpenAI API with thinking always disabled.

    Messages are passed through unchanged, so text and multimodal message
    content (including image URLs) are both supported.
    """
    extra_body = dict(kwargs.pop("extra_body", None) or {})
    chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
    chat_template_kwargs["enable_thinking"] = False
    extra_body["chat_template_kwargs"] = chat_template_kwargs
    extra_body.setdefault("top_k", top_k)

    return rewriter_llm.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        extra_body=extra_body,
        **kwargs,
    )


# ================================
# REGEX
# ================================
YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")

# ================================
# HELPERS
# ================================
def clean_text(t):
    """Text ko lowercase karke special chars hatao, sirf a-z 0-9 space rakho. Embedding/search ke liye normalize."""
    t = (t or "").lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def normalize_confidence(scores, min_conf=50, max_conf=95):
    """Scores ko min_conf se max_conf range mein scale karo. Sab indicators ko 50-95% confidence range mein map."""
    if not scores:
        return []
    mn, mx = min(scores), max(scores)
    if mn == mx:
        return [min_conf] * len(scores)
    return [round(min_conf + (s - mn)/(mx - mn)*(max_conf - min_conf), 2) for s in scores]



#########
BASE_YEAR_PATTERN = re.compile(r"(20\d{2})")

def detect_base_year(query):
    """Query mein base year (20xx) detect karo. CPI/CPI2 conflict resolve ke liye use hota hai."""
    q = query.lower()

    if "base year" or " base" in q:
        m = BASE_YEAR_PATTERN.search(q)
        if m:
            return int(m.group(1))

    return None


def resolve_cpi_conflict(results, query):
    """Jab CPI aur CPI2 dono top results mein hon: base year 2024+ → CPI2 rakho, else CPI rakho. Default: CPI2."""
    # Only when CPI and CPI2 both present in top results
    datasets = [r["product"] for r in results]

    if "CPI" not in datasets or "CPI2" not in datasets:
        return results  # kuch mat chhedo

    base_year = detect_base_year(query)

    # ---------- case 1: user ne base year bola ----------
    if base_year:
        if base_year >= 2024:
            # CPI2 rakho
            return [r for r in results if r["product"] != "CPI"]
        else:
            # CPI rakho
            return [r for r in results if r["product"] != "CPI2"]

    # ---------- case 2: base year nahi bola ----------
    return [r for r in results if r["product"] != "CPI"]


# ================================
# LLM QUERY REWRITE
# ================================
def rewrite_query_with_llm(user_query):
    """Qwen3.6-27B-FP8 LLM se query normalize/rewrite karo (spelling, synonyms, dataset full form). Fail → raw query return."""
    system_prompt = """You are a QUERY NORMALIZATION ENGINE for a data analytics system.

Task:
Rewrite the user query safely with controlled semantic normalization.

STRICT RULES:
1. DO NOT add any new information
2. DO NOT infer missing filters
3. DO NOT assume any category
4. DO NOT enrich meaning
5. ONLY rewrite words that already exist in the query
6. NEVER inject new concepts
7. NEVER add sector/gender/state unless explicitly present
8. Output ONLY rewritten query
9. No explanation
10. If the query contains a known dataset short form (CPI, IIP, NAS, PLFS, ASI, HCES, NSS, EC, WPI, UDISE, ASUSE, Gender, AISHE, ESI, CPIALRL, ENVSTAT, NFHS, RBI, NSS79, NSS79C, EC4, EC5, EC6, NSS77, NSS78), append its full form in the rewritten query while keeping the short form unchanged (e.g., "CPI" → "CPI Consumer Price Index", "EC" → "EC Economic Census", "WPI" → "WPI Wholesale Price Index", "UDISE" → "UDISE Unified District Information System for Education Plus"), and do not expand anything not explicitly present.

SPECIAL RULE (VERY IMPORTANT):
If the query contains "IIP" and also contains any month name 
(January–December or short forms like Jan, Feb, etc.), 
then add the word "monthly" to the query.

If query contains both "year" and "base year", clearly separate them:

Examples:
"IIP July data" → "IIP monthly July data"
"IIP for December" → "IIP monthly December"
"IIP Aug 2022" → "IIP monthly Aug 2022"
"gdp for year 2023-24 base year 2022-23" → "gdp year:2023-24 base_year:2022-23"

DO NOT apply this rule to any other dataset.
If query is about CPI, GDP, PLFS etc → do nothing.

ALLOWED OPERATIONS:
- spelling correction
- grammar correction
- casing normalization
- synonym normalization
- semantic mapping ONLY if the word exists explicitly in text

CRITICAL RULE (VERY IMPORTANT):
- If the user query is ONLY a dataset or product name
  (examples: IIP, CPI, CPIALRL, HCES, ASI, NAS, PLFS, CPI2, EC, EC4, EC5, EC6, WPI, UDISE, ASUSE, Gender, AISHE, ESI, ENVSTAT, NFHS, RBI, NSS79, NSS79C, NSS77, NSS78),
  then: RETURN THE QUERY WITH ITS FULL FORM APPENDED (e.g. "EC4" -> "EC4 4th Economic Census").
- Dataset names must NEVER be replaced with normal English words.

STRICT SEMANTIC MAP (ONLY IF WORD EXISTS):
- gao, gaon, village → rural
- shehar, city, metro → urban
- purush, aadmi, mard, man, men → male
- mahila, aurat, lady, women → female
- ladka → male
- ladki → female

❌ FORBIDDEN:
- Do NOT infer urban from city names
- Do NOT infer rural from state names
- Do NOT infer gender from profession
- Do NOT infer sector from geography
- Do NOT add any category automatically

Examples:
RAW: "mens judge in village"
→ "male judge in rural"

RAW: "Gini Coefficient for urban india in 2023-24"
→ "Gini Coefficient for urban in 2023-24"

RAW: "factory output gujrat 2022"
→ "factory output Gujarat 2022"

RAW: "men judges in delhi"
→ "male judges in Delhi"

RAW: "factory output in gujrat for 2022 in gao"
→ "factory output in Gujarat for 2022 in rural"

RAW: "data for mahila workers"
→ "data for female workers"

RAW: "gaon ke factory worker"
→ "rural factory worker"

RAW: "factory output in mumbai"
→ "factory output in Mumbai"
"""
    
    if not LLM_IS_RUNNING:
        return user_query
    
    try:
        response = call_qwen(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User Query:\n{user_query}"}
            ],
            temperature=0.3,
            max_tokens=256,
            top_p=0.95,
        )
        out = response.choices[0].message.content.strip()
        out = out.replace('"', '').replace("\n", " ").strip()
        return out
    except Exception as e:
        logger.error(f"LLM rewrite failed: {e}")
        return user_query

# ================================
# YEAR NORMALIZATION
# ================================
def normalize_year_string(s):
    """String se sirf digits nikalo (e.g. '2023-24' → '202324'). Year matching ke liye."""
    return re.sub(r"[^0-9]", "", str(s))


def map_year_to_option(user_year, options, query=None):
    """User year (e.g. 2023) ko options (2023-24, 2022-23, etc.) mein map karo.
    Also handles CPIALRL 'YYYY-YYYY' format and month-aware fiscal year mapping.
    Match nahi → None."""
    y = int(user_year)
    
    # --- Month-aware fiscal year mapping for CPIALRL ---
    # If query has a month and the options are in "YYYY-YYYY" format (CPIALRL),
    # months Jan-Mar belong to previous fiscal year
    
    month_names = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ]
    q_lower = (query or "").lower()
    is_jan_mar = any(m in q_lower for m in ["january", "february", "march"])
    
    # Check if options use "YYYY-YYYY" format (CPIALRL style)
    has_long_fy = any("-" in str(o.get("option", "")) and len(re.sub(r'[^0-9]', '', str(o.get("option", "")))) >= 8
                      for o in options[:3])
    
    targets = [
        f"{y}{y+1}",            # → "20232024"  (CPIALRL: "2023-2024")
        f"{y}{str(y+1)[-2:]}",  # → "202324"    (IIP: "2023-24")
        f"{y-1}{y}",            # → "20222023"  (CPIALRL: "2022-2023")
        f"{y-1}{str(y)[-2:]}",  # → "202223"    (IIP: "2022-23")
        str(y)                   # → "2023"      (CPI, IIP Monthly plain year)
    ]
    
    # For CPIALRL with Jan-Mar months, prioritize previous fiscal year
    if is_jan_mar and has_long_fy:
        targets = [
            f"{y-1}{y}",            # → "20102011" for Feb 2011
            f"{y-1}{str(y)[-2:]}",  # → "201011"
            f"{y}{y+1}",            # fallback
            f"{y}{str(y+1)[-2:]}",
            str(y)
        ]
    
    norm_options = {normalize_year_string(o["option"]): o for o in options}
    for t in targets:
        if t in norm_options:
            return norm_options[t]
    return None

# ================================
# FILTER ACCURACY & ESSENTIAL FILTERS (Moth criteria)
# Filter Accuracy = 4 filters only: Year, Sector, Gender, State (when present, must appear first)
# Essential Filters Accuracy = CPI: Series, Base Year; IIP: Base Year; ASI: Classification Year;
#                             NAS: Series, Frequency; CPIALRL: Base Year (when present, must appear)
# ================================
# 4 filters - Filter Accuracy basis
MANDATORY_4 = ["Year", "Sector", "Gender", "State"]

# Essential filters per dataset - Essential Filters Accuracy basis
ESSENTIAL_FILTERS_BY_DATASET = {
    "CPI": ["Series", "Base_Year", "Division"],
    "CPI2": ["Series", "Base_Year", "Division"],
    "IIP": ["Base_Year", "Frequency", "Type", "Category"],
    "ASI": ["classification_year"],
    "NAS": ["Series", "Frequency"],
    "CPIALRL": ["Base_Year"],
    "PLFS": ["Frequency"],
    "TUS": ["Age Group", "ICATUS Activity", "Day Of Week"],
    "WPI": ["Base_Year", "Major Group", "Group"],
    "ESI": ["Use of Energy Balance", "Energy Commodities"],
    "ASUSE": ["Frequency", "Sector"],
    "Gender": ["Gender", "State"],
    "AISHE": ["University Type", "State"],
    "NSS77": ["Sector", "State"],
    "NSS78": ["Sector", "State"],
    "HCES": ["Sector", "State"],
    "ENVSTAT": ["Category", "State"],
    "NFHS": ["Indicator Category", "State"],
    "EC4": ["State", "Sector", "Establishment Type"],
    "EC5": ["State", "Sector", "Establishment Type"],
    "EC6": ["State", "Sector", "Establishment Type"],
    "RBI": ["Bank Name", "Frequency"],
    "NSS79": ["Sector", "State"],
    "NSS79C": ["Sector", "State"],
    "UDISE": ["Management", "School Category", "State"],
}

# Datasets where Year/financial_Year filter should NOT be forced
_SKIP_YEAR_FILTER_DATASETS = {"NSS77", "NSS78", "EC4", "EC5", "EC6"}


def _priority_order_for_dataset(parent_code):
    """Filter ka priority order banao: Year,Sector,Gender,State pehle, phir dataset essential (Series,Base_Year,etc.), phir rest."""
    order = ["Year", "financial_Year", "Sector", "Gender", "State"]
    
    #  For NAS, Account filter must come BEFORE State filter
    # This ensures we set Account=Regional first, then select the appropriate State
    if parent_code == "NAS":
        # Insert Account before State
        if "State" in order:
            state_idx = order.index("State")
            order.insert(state_idx, "Account")
        else:
            order.append("Account")
    
    essential = ESSENTIAL_FILTERS_BY_DATASET.get(parent_code, [])
    for e in essential:
        if e not in order:
            order.append(e)
    for k in ["Base_Year", "Series", "classification_year", "Frequency", "Approach"]:
        if k not in order:
            order.append(k)
    return order


def ensure_mandatory_filter_order(best_filters, parent_code):
    """best_filters ko Moth criteria ke hisaab se reorder: 4 filters first, phir essential, phir baaki. Sirf reorder, kuch add/remove nahi."""
    if not best_filters:
        return best_filters
    by_name = {f["filter_name"]: f for f in best_filters}
    ordered = []
    priority = _priority_order_for_dataset(parent_code)
    for key in priority:
        if key in by_name:
            ordered.append(by_name.pop(key))
    for f in best_filters:
        if f["filter_name"] in by_name:
            ordered.append(by_name.pop(f["filter_name"]))
    for v in by_name.values():
        ordered.append(v)
    return ordered


def ensure_required_filters_present(best_filters, parent_code, grouped, query, cross_encoder):
    """Jo required filters (4 + essential) grouped mein hain lekin best_filters mein nahi, unko add karo. Phir sahi order apply karo."""
    required = list(MANDATORY_4)
    required.extend(ESSENTIAL_FILTERS_BY_DATASET.get(parent_code, []))
    required = list(dict.fromkeys(required))
    out_names = {f["filter_name"]: f for f in best_filters}
    for r in required:
        if r in out_names:
            continue
        # Skip Year/financial_Year for datasets that don't have it
        if r in ("Year", "financial_Year") and parent_code in _SKIP_YEAR_FILTER_DATASETS:
            continue
        if r not in grouped:
            continue
        opts = grouped[r]
        if not opts:
            continue
        best_opt = select_best_filter_option(query, r, opts, cross_encoder)
        best_filters.append({"filter_name": r, "option": best_opt["option"]})
        out_names[r] = best_filters[-1]
    # For _SKIP_YEAR_FILTER_DATASETS, REMOVE Year if it was added by generic filter loop
    if parent_code in _SKIP_YEAR_FILTER_DATASETS:
        best_filters = [f for f in best_filters if f["filter_name"] not in ("Year", "financial_Year")]
    best_filters = ensure_mandatory_filter_order(best_filters, parent_code)
    # CPI-only. Series+Base_Year valid combos: (Current,2012), (Back,2010)
    best_filters = ensure_cpi_series_base_year_consistent(best_filters, parent_code, grouped, query)
    # NAS-only. Account=Regional requires State to be set (not "Select All")
    best_filters = ensure_nas_account_state_consistent(best_filters, parent_code, grouped, query, cross_encoder)
    return best_filters


def ensure_cpi_series_base_year_consistent(best_filters, parent_code, grouped, query):
    """CPI/CPI2 ke liye Series+Base_Year valid combo ensure karo: Current↔2012, Back↔2010. Mismatch ho to fix. Baaki datasets pe no-op."""
    if parent_code not in ("CPI", "CPI2"):
        return best_filters
    by_name = {f["filter_name"]: f for f in best_filters}
    if "Series" not in by_name or "Base_Year" not in by_name:
        return best_filters
    q_lower = query.lower()
    series_opt = str(by_name["Series"].get("option", "")).lower()
    base_opt = str(by_name["Base_Year"].get("option", "")).lower()
    base_year = re.search(r"20\d{2}", base_opt)
    base_year = base_year.group(0) if base_year else ""
    target_series, target_base = None, None
    if "back" in q_lower or "2010" in q_lower:
        target_series, target_base = "Back", "2010"
    elif "current" in q_lower or "2012" in q_lower:
        target_series, target_base = "Current", "2012"
    elif series_opt == "back" and base_year and base_year != "2010":
        target_series, target_base = "Back", "2010"
    elif series_opt == "current" and base_year and base_year != "2012":
        target_series, target_base = "Current", "2012"
    elif base_year == "2010" and series_opt != "back":
        target_series, target_base = "Back", "2010"
    elif base_year == "2012" and series_opt != "current":
        target_series, target_base = "Current", "2012"
    elif not base_year and series_opt == "current":
        target_base = "2012"
    elif not base_year and series_opt == "back":
        target_base = "2010"
    if not target_series and not target_base:
        return best_filters
    series_opts = grouped.get("Series", [])
    base_opts = grouped.get("Base_Year", [])
    if target_series:
        for opt in series_opts:
            if str(opt.get("option", "")).lower() == target_series.lower():
                by_name["Series"]["option"] = opt["option"]
                break
    if target_base:
        for opt in base_opts:
            if target_base in str(opt.get("option", "")):
                by_name["Base_Year"]["option"] = opt["option"]
                break
    return best_filters


def ensure_nas_account_state_consistent(best_filters, parent_code, grouped, query, cross_encoder):
    """
    - If Account=Regional, State must be set to a specific state (not "Select All")
    - If a state is detected in query, Account should be Regional
    Baaki datasets pe no-op."""
    if parent_code != "NAS":
        return best_filters
    
    by_name = {f["filter_name"]: f for f in best_filters}
    
    # Check if Account filter exists
    if "Account" not in by_name:
        return best_filters
    
    account_opt = str(by_name["Account"].get("option", "")).lower()
    
    # Case 1: Account is Regional → ensure State is set to a specific state
    if account_opt == "regional":
        # Check if State filter exists
        if "State" in by_name:
            state_opt = str(by_name["State"].get("option", "")).lower()
            
            # If State is "Select All", try to detect a specific state from query
            if state_opt in ["select all", "selectall"]:
                detected_state = detect_state_from_query(query)
                
                if detected_state and "State" in grouped:
                    # Find the matching state option
                    state_opts = grouped["State"]
                    for opt in state_opts:
                        opt_text = str(opt.get("option", "")).strip()
                        if opt_text == detected_state or opt_text.lower() == detected_state.lower():
                            by_name["State"]["option"] = opt["option"]
                            break
                # If still "Select All" and no state detected, leave as is
                
        else:
            # State filter doesn't exist in best_filters but Account=Regional
            # Try to add State filter if it exists in grouped
            if "State" in grouped:
                state_opts = grouped["State"]
                detected_state = detect_state_from_query(query)
                
                if detected_state:
                    # Find matching state
                    for opt in state_opts:
                        opt_text = str(opt.get("option", "")).strip()
                        if opt_text == detected_state or opt_text.lower() == detected_state.lower():
                            best_filters.append({"filter_name": "State", "option": opt["option"]})
                            by_name["State"] = best_filters[-1]
                            break
                
                # If no state detected, add "Select All"
                if "State" not in by_name:
                    for opt in state_opts:
                        if str(opt.get("option", "")).lower().strip() in ["select all", "selectall"]:
                            best_filters.append({"filter_name": "State", "option": opt["option"]})
                            by_name["State"] = best_filters[-1]
                            break
    
    # Case 2: Detected state in query but Account is National → switch to Regional
    detected_state = detect_state_from_query(query)
    if detected_state and account_opt != "regional":
        # Check if Regional is available in Account options
        if "Account" in grouped:
            account_opts = grouped["Account"]
            for opt in account_opts:
                if str(opt.get("option", "")).lower() == "regional":
                    by_name["Account"]["option"] = opt["option"]
                    
                    # Also set the State filter
                    if "State" in by_name:
                        # Update existing State filter
                        if "State" in grouped:
                            state_opts = grouped["State"]
                            for s_opt in state_opts:
                                opt_text = str(s_opt.get("option", "")).strip()
                                if opt_text == detected_state or opt_text.lower() == detected_state.lower():
                                    by_name["State"]["option"] = s_opt["option"]
                                    break
                    else:
                        # Add State filter
                        if "State" in grouped:
                            state_opts = grouped["State"]
                            for s_opt in state_opts:
                                opt_text = str(s_opt.get("option", "")).strip()
                                if opt_text == detected_state or opt_text.lower() == detected_state.lower():
                                    best_filters.append({"filter_name": "State", "option": s_opt["option"]})
                                    by_name["State"] = best_filters[-1]
                                    break
                    break
    
    return best_filters


# ================================
# UNIVERSAL FILTER NORMALIZER
# ================================
def universal_filter_normalizer(product_name, filters_json):
    """Flatten one indicator's nested filters for request-time option selection.

    The temporary ``parent`` field below exists only inside Python because the
    product-specific filter rules need to know whether an option belongs to
    NAS, CPI, IIP, and so on. It is never written to Mongo or Qdrant.
    """
    flat = []
    def recurse(key, value):
        if isinstance(value, list) and all(isinstance(x, str) for x in value):
            for opt in value:
                flat.append({"parent": product_name,"filter_name": key,"option": opt})
        elif isinstance(value, list) and all(isinstance(x, dict) for x in value):
            for item in value:
                for k, v in item.items():
                    if k.lower() in ["name", "title", "label"]:
                        flat.append({"parent": product_name,"filter_name": key,"option": v})
                    else:
                        recurse(k, v)
        elif isinstance(value, dict):
            for k, v in value.items():
                recurse(k, v)

    for f in filters_json:
        if isinstance(f, dict):
            for k, v in f.items():
                recurse(k, v)
    return flat


def apply_iip_filter_hierarchy(best_filters, filters_json, query):
    """Correct IIP filter parents without changing any other product.

    IIP stores Type as a nested list whose Category and Subcategory values
    belong to that Type. The universal normalizer deliberately serves every
    product and flattens those values, so this IIP-only step restores the
    owning Type after the best subcategory/category has been selected.

    NIC codes are not linked to subcategories in the current IIP metadata.
    Consequently, a NIC code is selected only when the query explicitly says
    ``NIC``; a calendar year such as 2014 must never select NIC 14.
    """
    type_nodes = []
    for filter_object in filters_json or []:
        if not isinstance(filter_object, dict):
            continue
        value = filter_object.get("Type")
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            type_nodes = value
            break

    if not type_nodes:
        return best_filters

    by_name = {
        item.get("filter_name"): item
        for item in best_filters
        if isinstance(item, dict) and item.get("filter_name")
    }
    select_all_values = {"select all", "selectall", "all"}

    def clean_strings(value):
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if isinstance(item, str) and item.strip()]

    def selected_value(filter_name):
        item = by_name.get(filter_name)
        return str(item.get("option", "")).strip() if item else ""

    def set_value(filter_name, option):
        if filter_name in by_name:
            by_name[filter_name]["option"] = option

    selected_subcategory = selected_value("Subcategory")
    selected_category = selected_value("Category")
    owner = None

    # A specific matched subcategory is the strongest available ownership
    # signal in the current IIP JSON.
    if selected_subcategory.lower() not in select_all_values:
        owners = [
            node for node in type_nodes
            if selected_subcategory.casefold() in {
                option.casefold() for option in clean_strings(node.get("Subcategory"))
            }
        ]
        if len(owners) == 1:
            owner = owners[0]

   
    if owner is None and selected_category.lower() not in select_all_values:
        owners = [
            node for node in type_nodes
            if selected_category.casefold() in {
                option.casefold() for option in clean_strings(node.get("Category"))
            }
        ]
        if len(owners) == 1:
            owner = owners[0]

    
    if owner is None:
        query_lower = query.lower()
        for node in type_nodes:
            type_name = str(node.get("name", "")).strip()
            if type_name and type_name.lower() in query_lower:
                owner = node
                break

    if owner is not None:
        owner_name = str(owner.get("name", "")).strip()
        if owner_name:
            set_value("Type", owner_name)

        scoped_categories = clean_strings(owner.get("Category"))
        scoped_subcategories = clean_strings(owner.get("Subcategory"))

        
        if (
            "Subcategory" in by_name
            and selected_subcategory
            and selected_subcategory.lower() not in select_all_values
            and selected_subcategory.casefold() not in {v.casefold() for v in scoped_subcategories}
        ):
            fallback = next(
                (value for value in scoped_subcategories if value.lower() in select_all_values),
                scoped_subcategories[0] if scoped_subcategories else "Select All",
            )
            set_value("Subcategory", fallback)
            selected_subcategory = fallback

        category_choice = None
        query_lower = query.lower()
        for category in scoped_categories:
            if category.lower() not in select_all_values and category.lower() in query_lower:
                category_choice = category
                break

        if category_choice is None and len([
            value for value in scoped_categories if value.lower() not in select_all_values
        ]) == 1:
            category_choice = next(
                value for value in scoped_categories if value.lower() not in select_all_values
            )


        if category_choice is None and owner_name.casefold() == "sectoral":
            subcategory_lower = selected_subcategory.lower()
            category_rules = [
                (("water", "sewerage", "waste"), ("water supply",)),
                (("gas supply",), ("electricity & gas", "gas supply")),
                (("electricity", "power"), ("electricity",)),
                (("mining", "quarrying", "mineral", "extraction"), ("mining",)),
            ]
            for subcategory_terms, category_terms in category_rules:
                if any(term in subcategory_lower for term in subcategory_terms):
                    category_choice = next(
                        (
                            category for category in scoped_categories
                            if any(term in category.lower() for term in category_terms)
                        ),
                        None,
                    )
                    if category_choice:
                        break
            if category_choice is None:
                category_choice = next(
                    (category for category in scoped_categories if category.casefold() == "manufacturing"),
                    None,
                )

        if category_choice is None and selected_category.casefold() in {
            value.casefold() for value in scoped_categories
        }:
            category_choice = selected_category
        if category_choice is None and scoped_categories:
            category_choice = next(
                (value for value in scoped_categories if value.lower() in select_all_values),
                scoped_categories[0],
            )
        if category_choice:
            set_value("Category", category_choice)

    # Avoid the generic substring match where NIC 14 matches calendar year
    # 2014. Exact NIC selection remains available through phrases such as
    if "NIC_Code" in by_name:
        nic_options = []
        for filter_object in filters_json or []:
            if isinstance(filter_object, dict) and "NIC_Code" in filter_object:
                nic_options = clean_strings(filter_object["NIC_Code"])
                break
        nic_match = re.search(
            r"\bnic(?:[_\s-]*code)?\s*[:#-]?\s*([0-9]{2,4}(?:-[0-9]+)?)\b",
            query,
            flags=re.IGNORECASE,
        )
        explicit_nic = nic_match.group(1) if nic_match else None
        if explicit_nic and explicit_nic in nic_options:
            set_value("NIC_Code", explicit_nic)
        else:
            fallback = next(
                (value for value in nic_options if value.lower() in select_all_values),
                selected_value("NIC_Code"),
            )
            set_value("NIC_Code", fallback)

    return best_filters


#############LLM 
# ================================
# SMART FILTER ENGINE
# ================================
def detect_state_from_query(query):
    """Detect state name from query for NAS regional data. Returns state name if found, else None."""
    q_lower = query.lower()
    
    # List of all Indian states and UTs (matching products.nas.json)
    states = [
        "Andaman & Nicobar Islands", "Andaman and Nicobar Islands", "Andaman",
        "Andhra Pradesh", "Arunachal Pradesh", 
        "Assam", "Bihar", "Chandigarh", "Chhattisgarh", 
        "Delhi", "Goa", "Gujarat", "Haryana", 
        "Himachal Pradesh", "Jammu & Kashmir", "Jammu and Kashmir",
        "Jharkhand", "Karnataka", "Kerala", "Ladakh",
        "Madhya Pradesh", "Maharashtra", "Manipur", 
        "Meghalaya", "Mizoram", "Nagaland", "Odisha", 
        "Puducherry", "Punjab", "Rajasthan", "Sikkim", 
        "Tamil Nadu", "Telangana", "Tripura", 
        "Uttarakhand", "Uttar Pradesh", "West Bengal"
    ]
    
    # Check for exact matches and common variations
    for state in states:
        state_lower = state.lower()
        if state_lower in q_lower:
            return state
        # Handle "&" vs "and" variations
        if "&" in state:
            state_alt = state.replace("&", "and")
            if state_alt.lower() in q_lower:
                return state
        if "and" in state_lower:
            state_alt = state.replace(" and ", " & ")
            if state_alt.lower() in q_lower:
                return state
    
    # Check for common abbreviations
    abbreviations = {
        "ap": "Andhra Pradesh",
        "hp": "Himachal Pradesh",
        "mp": "Madhya Pradesh",
        "up": "Uttar Pradesh",
        "tn": "Tamil Nadu",
        "wb": "West Bengal",
        "jk": "Jammu & Kashmir",
        "j&k": "Jammu & Kashmir"
    }
    
    words = q_lower.split()
    for word in words:
        if word in abbreviations:
            return abbreviations[word]
    
    return None


ASI_CLASSIFICATION_RULES = (
    (1992, 1997, "1987"),
    (1998, 2003, "1998"),
    (2004, 2007, "2004"),
    (2008, 2023, "2008"),
)


def select_asi_classification_year(query, options):
    """Choose the ASI NIC classification that applies to the requested data year."""
    query_lower = query.lower()

    def get_option(target):
        return next(
            (
                option for option in options
                if str(option.get("option", "")).strip() == target
            ),
            None,
        )

    # An explicit NIC or classification request takes priority over the data year.
    explicit_match = re.search(
        r"(?:nic|classification[\s_-]*year)\s*[:\-]?\s*(1987|1998|2004|2008)",
        query_lower,
    )
    if explicit_match:
        explicit_option = get_option(explicit_match.group(1))
        if explicit_option:
            return explicit_option

    # Supports both financial years such as 2022-23 and plain years such as 2003.
    data_year_match = re.search(r"\b((?:19|20)\d{2})(?:[-/]\d{2,4})?\b", query_lower)
    if data_year_match:
        data_year = int(data_year_match.group(1))
        for start_year, end_year, classification_year in ASI_CLASSIFICATION_RULES:
            if start_year <= data_year <= end_year:
                mapped_option = get_option(classification_year)
                if mapped_option:
                    return mapped_option

    # ASI defaults to the latest NIC classification where no data year is given.
    return get_option("2008") or options[0]


def select_best_filter_option(query, filter_name, options, cross_encoder):
    """Query ke hisaab se sabse sahi filter option pick karo. Year→Select All/latest, Series→Current/Back, State/Sector→match, etc."""
    if not options:
        return {"parent": "", "filter_name": filter_name, "option": "Select All"}
    q_lower = query.lower()
    fname_lower = filter_name.lower()
    parent_code = str(options[0].get("parent", "")).split("_", 1)[0].upper()

    if fname_lower == "classification_year" and parent_code == "ASI":
        return select_asi_classification_year(query, options)
    
    # =========================
    # ACCOUNT FILTER (NAS - National vs Regional)
    #  For NAS, detect state-level queries and set Account to "Regional"
    # =========================
    if fname_lower == "account":
        parent_code = options[0].get("parent", "").split("_")[0] if options else ""
        
        if parent_code == "NAS":
            # Keywords indicating regional/state-level data
            regional_keywords = [
                "state", "states", "regional", "gsdp", "gsva", "nsdp", "nsva",
                "pcnsdp", "pcgsdp", "state-wise", "statewise", "state level",
                "state-level", "gross state", "net state", "per capita state"
            ]
            
            # Check if query mentions regional keywords
            is_regional_query = any(kw in q_lower for kw in regional_keywords)
            
        
            detected_state = detect_state_from_query(query)
            if detected_state:
                is_regional_query = True
            
            # If regional query detected, select "Regional" if available
            if is_regional_query:
                for opt in options:
                    if str(opt.get("option", "")).lower() == "regional":
                        return opt
            
         
            for opt in options:
                if str(opt.get("option", "")).lower() == "national":
                    return opt
        
        # For non-NAS datasets, check for "Select All" or return first
        for opt in options:
            o_lower = str(opt.get("option", "")).lower().strip()
            if o_lower in ["select all", "selectall"]:
                return opt
        return options[0]
    
    # =========================
    # STATE FILTER (NAS Regional data)
    # If Account is Regional, detect and select the correct state
    # =========================
    if fname_lower == "state":
        parent_code = options[0].get("parent", "").split("_")[0] if options else ""
        
        if parent_code == "NAS":
            # Try to detect state from query
            detected_state = detect_state_from_query(query)
            
            if detected_state:
                # Find matching state option
                for opt in options:
                    opt_text = str(opt.get("option", "")).strip()
                    # Exact match or close match (handling & vs and)
                    if opt_text == detected_state:
                        return opt
                    # Handle variations
                    if opt_text.lower() == detected_state.lower():
                        return opt
                    if "&" in opt_text and opt_text.replace("&", "and").lower() == detected_state.lower():
                        return opt
                    if "and" in opt_text.lower() and opt_text.replace(" and ", " & ").lower() == detected_state.lower():
                        return opt
            
            # No state detected → return "Select All" if available
            for opt in options:
                if str(opt.get("option", "")).lower().strip() in ["select all", "selectall"]:
                    return opt
        
        # For non-NAS or no match, use fuzzy matching (existing logic below)
        pass
     
    # =========================
    # FREQUENCY FILTER
    # =========================
    if fname_lower in ["frequency"]:
        # --- Check for explicit mention ---
        for keyword in ["annually", "quarterly", "monthly", "annual"]:
            if keyword in q_lower:
                for opt in options:
                    o = str(opt.get("option", "")).lower()
                    if o.startswith(keyword) or keyword.startswith(o):
                        return opt

        # --- Month names → Monthly (full names only to avoid "may" false positive) ---
        month_names = [
            "january", "february", "march", "april", "june",
            "july", "august", "september", "october", "november", "december"
        ]
        if any(m in q_lower for m in month_names):
            for opt in options:
                o = str(opt.get("option", "")).lower()
                if o in ["monthly", "month"]:
                    return opt

        # --- Quarter keywords → Quarterly ---
        quarter_keywords = ["quarter", "quarterly", "q1", "q2", "q3", "q4",
                            "jul-sep", "oct-dec", "jan-mar", "apr-jun"]
        if any(qk in q_lower for qk in quarter_keywords):
            for opt in options:
                if str(opt.get("option", "")).lower() in ["quarterly"]:
                    return opt

        # --- Year format "2023-24" or standalone year → Annually ---
        if re.search(r"\d{4}[-/]\d{2,4}", q_lower) or YEAR_PATTERN.search(q_lower):
            for opt in options:
                if str(opt.get("option", "")).lower() in ["annually", "annual"]:
                    return opt

        # --- No frequency clue → Select All ---
        return {
            "parent": options[0]["parent"],
            "filter_name": filter_name,
            "option": "Select All"
        }
    # =========================
    # YEAR FILTER (Year, financial_Year)
    # User mention nahi kiya → Select All (agar hai), else latest year
    # User mention kiya → exact year
    #  When query has both "base year XXXX" and data year,
    # pick the data year, not the base year
    # =========================
    if (
        "year" in fname_lower
        and "base" not in fname_lower
        and fname_lower != "classification_year"
    ):
        # Extract all years from query (include 19xx and 20xx for IIP/CPIALRL) ---
        _ANY_YEAR_PAT = re.compile(r"\b((?:19|20)\d{2})\b")
        all_years = _ANY_YEAR_PAT.findall(q_lower)

        # --- Separate base year from data year ---
        # If query contains "base" keyword, the year right after "base" is the base year
        base_year_in_query = None
        data_years = []
        if all_years:
            # Find years that appear near "base" keyword
            base_pattern = re.search(r'base[_ ]?(?:year)?[:\s]*(?:of\s+)?(?:year\s+)?(\d{4})', q_lower)
            if base_pattern:
                base_year_in_query = base_pattern.group(1)
            # Also check for "(Base XXXX-XX)" pattern like "(Base 2011-12)"
            base_paren = re.search(r'\(\s*base\s+(\d{4})', q_lower)
            if base_paren:
                base_year_in_query = base_paren.group(1)
            # Data years = all years minus the base year
            for y in all_years:
                if y != base_year_in_query:
                    data_years.append(y)
            # If no data years found (all years were base year), use all years
            if not data_years:
                data_years = all_years
        
        #  Also handle financial year format "YYYY-YY" in query 
        fy_match = re.search(r'(\d{4})[-/](\d{2,4})', q_lower)
        fy_year = None
        if fy_match:
            fy_str = fy_match.group(0)
            # Make sure this isn't a base year pattern
            before_fy = q_lower[:fy_match.start()].rstrip()
            if not before_fy.endswith('base') and 'base_year' not in before_fy[-15:]:
                fy_year = fy_match.group(1)

        # --- Quarter-aware fiscal year mapping (Golden rule: PLFS/other FY datasets) ---

        quarter_q4 = re.search(r'(?:jan(?:uary)?[\s\-]+mar(?:ch)?|q4)', q_lower)
        if quarter_q4 and data_years and not fy_year:
            qy = int(data_years[0])
            adjusted_year = str(qy - 1)
            mapped = map_year_to_option(adjusted_year, options, query=query)
            if mapped:
                return mapped

        if not all_years and not fy_year:
            for opt in options:
                o = str(opt.get("option", "")).strip().lower()
                if o in ("select all", "selectall"):
                    return opt
          
            def _extract_year_val(o):
                m = re.search(r"20\d{2}", str(o.get("option", "")))
                return int(m.group(0)) if m else 0
            return max(options, key=lambda o: _extract_year_val(o))

       
        if fy_year and fy_year not in (base_year_in_query or ""):
            mapped = map_year_to_option(fy_year, options, query=query)
            if mapped:
                return mapped

       
        user_year = data_years[0] if data_years else (all_years[0] if all_years else None)
        if user_year:
            mapped = map_year_to_option(user_year, options, query=query)
            if mapped:
                return mapped

        pairs = [(query, f"{filter_name} {o['option']}") for o in options]
        scores = cross_encoder.predict(pairs)
        return options[int(np.argmax(scores))]

    # =========================
    # SERIES FILTER (CPI, NAS - Current/Back)
    # =========================
    if fname_lower == "series":
        if "back" in q_lower or "historical" in q_lower:
            for opt in options:
                if str(opt.get("option", "")).lower() == "back":
                    return opt
        if "current" in q_lower:
            for opt in options:
                if str(opt.get("option", "")).lower() == "current":
                    return opt
        for opt in options:
            if str(opt.get("option", "")).lower() == "current":
                return opt
        return options[0] if options else {"parent": "", "filter_name": filter_name, "option": "Select All"}

    # =========================
    # CLASSIFICATION YEAR (ASI)
    #  ASI-only. Options are NIC years: 2008, 2004, 1998, 1987
    # If user mentions "NIC 2004" or "NIC-2004" → pick 2004
    # If user mentions "NIC 2008" → pick 2008
    # Default → latest (2008)
    # =========================
    if fname_lower == "classification_year":
        # Check for explicit NIC year mention like "NIC 2004", "NIC-2004", "NIC2004"
        nic_match = re.search(r'nic[\s\-_]*(\d{4})', q_lower)
        if nic_match:
            nic_year = nic_match.group(1)
            for opt in options:
                if str(opt.get("option", "")).strip() == nic_year:
                    return opt
        # Check for explicit mention of classification year value
        for opt in options:
            opt_text = str(opt.get("option", "")).strip()
            # Only match if the year appears as "classification year XXXX" or standalone mention
            if opt_text in q_lower:
               
                # by checking it's not part of a range
                idx = q_lower.find(opt_text)
                after = q_lower[idx + len(opt_text):idx + len(opt_text) + 1] if idx + len(opt_text) < len(q_lower) else ""
                if after not in ("-", "/"):  # Not a range like "2004-05"
                    return opt
        # Default → latest classification year (2008 for NIC)
        def _extract_year(opt):
            m = re.search(r"\d{4}", str(opt.get("option", "")))
            return int(m.group(0)) if m else 0
        return max(options, key=lambda o: _extract_year(o))

    # =========================
    # BASE YEAR FILTER (FINAL FIX)
    #  Product-specific defaults
    # NAS: default "2011-12" unless data year >= 2023 → "2022-23"
    # IIP: default based on data year range
    #   pre-2005 → "1993-94", 2005-2011 → "2004-05", 2012+ → "2011-12"
    # CPI: handled by ensure_cpi_series_base_year_consistent
    # Others: latest base year
    # =========================
    if "base" in fname_lower and "year" in fname_lower:

        base_explicit = re.search(r'base[_ ]?(?:year)?[:\s]*(\d{4}(?:[-/]\d{2,4})?)', q_lower)
        if base_explicit:
            base_val = base_explicit.group(1)
            for opt in options:
                opt_text = str(opt.get("option", "")).lower().strip()
                if base_val in opt_text or opt_text in base_val:
                    return opt
                # Normalize both and compare
                norm_base = re.sub(r'[^0-9]', '', base_val)
                norm_opt = re.sub(r'[^0-9]', '', opt_text)
                if norm_base and norm_opt and (norm_base.startswith(norm_opt) or norm_opt.startswith(norm_base)):
                    return opt

        # Check for standalone base year value in query
        for opt in options:
            opt_text = str(opt.get("option", "")).lower().strip()
            if opt_text in q_lower:
                return opt

        parent_code = options[0].get("parent", "").split("_")[0] if options else ""

        def extract_start_year(opt):
            m = re.search(r"\d{4}", str(opt.get("option", "")))
            return int(m.group(0)) if m else 0

        if parent_code == "NAS":
            data_year_match = YEAR_PATTERN.search(q_lower)
            data_year = int(data_year_match.group(1)) if data_year_match else 0
            target_base = "2022-23" if data_year >= 2023 else "2011-12"
            for opt in options:
                if target_base in str(opt.get("option", "")):
                    return opt

        if parent_code == "IIP":
            _ANY_YEAR_PAT_IIP = re.compile(r"\b((?:19|20)\d{2})\b")
            # Get all years from query, pick the first non-base-year
            all_q_years = _ANY_YEAR_PAT_IIP.findall(q_lower)
            # Remove years that are part of "base YYYY" pattern
            base_yr_match = re.search(r'base[_ ]?(?:year)?[:\s]*(\d{4})', q_lower)
            base_yr = base_yr_match.group(1) if base_yr_match else None
            data_yr = None
            for yr in all_q_years:
                if yr != base_yr:
                    data_yr = int(yr)
                    break
            if data_yr is None and all_q_years:
                data_yr = int(all_q_years[0])
            if data_yr:
                if data_yr < 2005:
                    target = "1993-94"
                elif data_yr < 2012:
                    target = "2004-05"
                else:
                    target = "2011-12"
                for opt in options:
                    if target in str(opt.get("option", "")):
                        return opt
            for opt in options:
                if "2011" in str(opt.get("option", "")):
                    return opt

        latest = max(options, key=lambda o: extract_start_year(o))
        return latest

    
    if fname_lower == "month":
        month_map = [
            ("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"),
            ("may", "may"), ("june", "jun"), ("july", "jul"), ("august", "aug"),
            ("september", "sep"), ("october", "oct"), ("november", "nov"), ("december", "dec")
        ]
        for full, short in month_map:
            if full in q_lower or short in q_lower:
                for opt in options:
                    if str(opt.get("option", "")).lower() == full or str(opt.get("option", "")).lower().startswith(full[:3]):
                        return opt
        return {
            "parent": options[0]["parent"],
            "filter_name": filter_name,
            "option": "Select All"
        }

    # =========================
    # PRODUCT-SPECIFIC FILTERS (Isolation)
    # =========================
    if fname_lower in ["management", "school category"]:
        # UDISE specific
        for opt in options:
            if str(opt.get("option", "")).lower() in q_lower:
                return opt
    if fname_lower in ["university type", "name of univ type"]:
        # AISHE specific
        for opt in options:
            if str(opt.get("option", "")).lower() in q_lower:
                return opt
    if fname_lower == "indicator category":
        # ENVSTAT, NFHS specific
        for opt in options:
            if str(opt.get("option", "")).lower() in q_lower:
                return opt
    if fname_lower in ["major group", "group"]:
        # WPI specific
        for opt in options:
            if str(opt.get("option", "")).lower() in q_lower:
                return opt

    # =========================
    # AGE GROUP FILTER
    # =========================
    # Metadata stores ranges such as "18-30 years", while people naturally
    # ask "aged 18 to 30" or "18–30". Match the two numeric bounds exactly
    # before generic text matching can incorrectly fall back to "Select All".
    if fname_lower in ["age_group", "age group"]:
        age_match = re.search(
            r"\b(?:aged?\s*)?(\d+)\s*(?:to|[-–—])\s*(\d+)(?:\s*years?)?\b",
            q_lower,
        )
        if age_match:
            requested_range = age_match.groups()
            for opt in options:
                option_text = str(opt.get("option", "")).lower()
                option_match = re.search(
                    r"\b(\d+)\s*[-–—]\s*(\d+)\s*years?\b",
                    option_text,
                )
                if option_match and option_match.groups() == requested_range:
                    return opt

    # =========================
    # OTHER FILTERS
    # =========================
    mentioned = []

    for opt in options:
        opt_text = str(opt.get("option", "")).lower().strip()
        if not opt_text:
            continue

        if opt_text in q_lower:
            mentioned.append(opt)
            continue

        for word in q_lower.split():
            if difflib.SequenceMatcher(None, opt_text, word).ratio() > 0.80:
                mentioned.append(opt)
                break

    if mentioned:
        pairs = [(query, f"{filter_name} {o['option']}") for o in mentioned]
        scores = cross_encoder.predict(pairs)
        return mentioned[int(np.argmax(scores))]

    #  Find the best default "All" option if no specific match
    for opt in options:
        o_lower = str(opt.get("option", "")).lower().strip()
        if o_lower in ["select all", "selectall", "all", "person", "combined", "general", "total"] or o_lower.startswith("all "):
            return opt
            
    # As a last resort, just return the first available valid option instead of a fake string
    return options[0]


# ================================
# PRODUCT CATALOG AND SEARCH INDEX
# ================================
# Published product JSON is stored in MongoDB.  products/index.json is used
# only once to preserve legacy product order while the initial Mongo records
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRODUCT_CATALOG_FILE = os.path.join(BASE_DIR, "products", "index.json")
if not os.path.exists(PRODUCT_CATALOG_FILE):
    raise FileNotFoundError(f"Product catalog not found: {PRODUCT_CATALOG_FILE}")

metadata_manager = MetadataManager(PRODUCT_CATALOG_FILE)
# Metadata changes can take time because each product must be re-embedded.
# This manager stores progress in MongoDB so it survives page navigation.
metadata_job_manager = MetadataJobManager()
DATASETS, INDICATORS, FILTERS = [], [], []
SEARCH_INDEX_LOCK = threading.RLock()

# ================================
# MODELS
# ================================

bi_encoder = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")


production_migration_manager = ProductionMigrationManager(metadata_manager, bi_encoder)

# ================================
# VECTOR DB
# ================================
VECTOR_DIM = bi_encoder.get_sentence_embedding_dimension()
# The Mongo and Qdrant collections share this name but are different stores:
# Mongo holds exact source JSON and Qdrant holds generated embedding vectors.
COLLECTION = os.environ.get("QDRANT_COLLECTION", "product_metadata")
PRODUCTION_COLLECTION = os.environ.get(
    "PRODUCTION_QDRANT_COLLECTION",
    "product_metadata_prod",
)
qclient = None
production_qclient = None
faiss_index = None
# FAISS is persisted separately from the Docker image. Mount this directory as
# a Docker volume so rebuilding/recreating the application container reuses the
# same vectors instead of embedding every published indicator again.
FAISS_CACHE_DIR = os.environ.get("FAISS_CACHE_DIR", os.path.join(BASE_DIR, "data", "faiss"))
FAISS_INDEX_FILE = os.path.join(FAISS_CACHE_DIR, "indicators.faiss")
FAISS_MANIFEST_FILE = os.path.join(FAISS_CACHE_DIR, "manifest.json")
# Change this value whenever the embedding model or embedding-text rule changes.
# It deliberately participates in the cache fingerprint.
FAISS_CACHE_VERSION = "mixedbread-mxbai-large_indicator-name-description_v1"

if USE_QDRANT:
    try:
        qclient = QdrantClient(
            host=os.environ.get("QDRANT_HOST", "localhost"),
            port=int(os.environ.get("QDRANT_PORT", "6333")),
            grpc_port=int(os.environ.get("QDRANT_GRPC_PORT", "6334")),
            prefer_grpc=True,
            api_key=os.environ.get("QDRANT_API_KEY"),
            https=False,
        )
        metadata_manager.configure_indexing(bi_encoder, qclient, qmodels, COLLECTION)
        logger.info(
            "Qdrant primary client configured: host=%s, grpc_port=%s, collection=%s",
            os.environ.get("QDRANT_HOST", "localhost"),
            os.environ.get("QDRANT_GRPC_PORT", "6334"),
            COLLECTION,
        )
    except Exception as exc:
       
        USE_QDRANT = False
        logger.warning("Qdrant client setup failed; FAISS fallback will be used: %s", exc)

if USE_QDRANT:
    try:
        production_qclient = QdrantClient(
            host=os.environ.get("PRODUCTION_QDRANT_HOST", "localhost"),
            port=int(os.environ.get("PRODUCTION_QDRANT_PORT", "6333")),
            grpc_port=int(os.environ.get("PRODUCTION_QDRANT_GRPC_PORT", "6334")),
            prefer_grpc=os.environ.get("PRODUCTION_QDRANT_PREFER_GRPC", "true").lower() == "true",
            api_key=os.environ.get("PRODUCTION_QDRANT_API_KEY"),
            https=os.environ.get("PRODUCTION_QDRANT_HTTPS", "false").lower() == "true",
        )
        logger.info(
            "Production Qdrant client configured: host=%s, collection=%s",
            os.environ.get("PRODUCTION_QDRANT_HOST", "localhost"),
            PRODUCTION_COLLECTION,
        )
    except Exception as exc:
        # Public API requests retain the existing FAISS fallback if this
        # optional production Qdrant client is unavailable.
        logger.warning("Production Qdrant client setup failed; public API will use FAISS fallback: %s", exc)


def _normalise_catalog_datasets(product_datasets):
    """Build FAISS fallback records with the same clean indicator schema as Qdrant."""
    datasets, indicators, filters = [], [], []
    for dataset_name, dataset_info in product_datasets.items():
        datasets.append({"code": dataset_name, "name": dataset_name})
        for indicator in dataset_info.get("indicators", []):
            if not isinstance(indicator, dict):
                continue
            indicator_record = {
                "product": dataset_name,
                "product_desc": dataset_info.get("description", ""),
                "name": indicator["name"],
                "description": indicator.get("description", ""),
                "filters": indicator.get("filters", []),
               
                # FAISS is an in-memory fallback; it does not need it to
                # select or format an indicator result.
                "last_updated_on": None,
            }
            indicators.append(indicator_record)
            # Retained only for startup diagnostics. Search responses flatten
            # their selected indicator's own filters directly.
            filters.extend(universal_filter_normalizer(dataset_name, indicator.get("filters", [])))
    return datasets, indicators, filters


def _faiss_cache_fingerprint(indicators):
    """Return a stable fingerprint for the exact text embedded by FAISS.

    Filters and product descriptions are intentionally excluded because they
    are payload/result metadata, not part of the indicator vector. Changing a
    filter therefore does not waste time regenerating identical FAISS vectors.
    """
    embedding_rows = [
        [
            clean_text(indicator.get("name", "")),
            clean_text(indicator.get("description", "")),
        ]
        for indicator in indicators
    ]
    fingerprint_input = json.dumps(
        {
            "cache_version": FAISS_CACHE_VERSION,
            "vector_dimension": VECTOR_DIM,
            "embedding_rows": embedding_rows,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(fingerprint_input).hexdigest()


def _load_persisted_faiss(indicators):
    """Load FAISS only when its manifest matches current Mongo metadata."""
    if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(FAISS_MANIFEST_FILE):
        logger.info("Persisted FAISS cache is not present; a one-time build is required")
        return None
    try:
        with open(FAISS_MANIFEST_FILE, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        expected_fingerprint = _faiss_cache_fingerprint(indicators)
        if manifest.get("fingerprint") != expected_fingerprint:
            logger.info("Persisted FAISS cache is stale because indicator metadata changed")
            return None
        if manifest.get("indicator_count") != len(indicators):
            logger.warning("Persisted FAISS manifest has an unexpected indicator count")
            return None
        loaded_index = faiss.read_index(FAISS_INDEX_FILE)
        if loaded_index.ntotal != len(indicators) or loaded_index.d != VECTOR_DIM:
            logger.warning(
                "Persisted FAISS index shape is invalid: vectors=%s dimension=%s",
                loaded_index.ntotal,
                loaded_index.d,
            )
            return None
        logger.info(
            "Persisted FAISS fallback loaded without embedding: indicators=%s file=%s",
            loaded_index.ntotal,
            FAISS_INDEX_FILE,
        )
        return loaded_index
    except Exception as exc:
        logger.warning("Could not load persisted FAISS cache; it will be rebuilt once: %s", exc)
        return None


def _persist_faiss(index, indicators):
    """Atomically persist the FAISS index and its metadata fingerprint."""
    os.makedirs(FAISS_CACHE_DIR, exist_ok=True)
    index_tmp = f"{FAISS_INDEX_FILE}.{os.getpid()}.tmp"
    manifest_tmp = f"{FAISS_MANIFEST_FILE}.{os.getpid()}.tmp"
    try:
        faiss.write_index(index, index_tmp)
        with open(manifest_tmp, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "fingerprint": _faiss_cache_fingerprint(indicators),
                    "indicator_count": len(indicators),
                    "vector_dimension": VECTOR_DIM,
                    "cache_version": FAISS_CACHE_VERSION,
                    "saved_at": datetime.now().astimezone().isoformat(),
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(index_tmp, FAISS_INDEX_FILE)
        os.replace(manifest_tmp, FAISS_MANIFEST_FILE)
        logger.info("Persisted FAISS fallback saved: file=%s", FAISS_INDEX_FILE)
    finally:
        for temporary_file in (index_tmp, manifest_tmp):
            if os.path.exists(temporary_file):
                try:
                    os.remove(temporary_file)
                except OSError:
                    logger.warning("Could not remove temporary FAISS file: %s", temporary_file)


def _build_and_persist_faiss(indicators):
    """Build FAISS only when no valid persisted copy exists."""
    embedding_texts = [
        " ".join(
            part for part in (
                clean_text(indicator["name"]),
                clean_text(indicator.get("description", "")),
            ) if part
        )
        for indicator in indicators
    ]
    logger.info("Building FAISS fallback because its persisted metadata changed: indicators=%s", len(indicators))
    embeddings = bi_encoder.encode(embedding_texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings /= np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)
    built_index = faiss.IndexFlatL2(embeddings.shape[1])
    built_index.add(embeddings.astype("float32"))
    _persist_faiss(built_index, indicators)
    return built_index


def _load_catalog_metadata():
    """Load exact product/filter response metadata without generating vectors."""
    product_datasets = metadata_manager.load_all_datasets()
    datasets, indicators, filters = _normalise_catalog_datasets(product_datasets)
    if not indicators:
        raise RuntimeError("No published indicators were found in MongoDB product_metadata")
    return datasets, indicators, filters


def initialize_search_runtime():
    """Reuse persisted Qdrant and FAISS indexes during application startup.

    This function never writes to Qdrant. FAISS is rebuilt only once when its
    persistent cache is missing/stale; ordinary image builds and Docker restarts
    load both existing indexes without re-embedding product metadata.
    """
    global DATASETS, INDICATORS, FILTERS, faiss_index, USE_QDRANT

    logger.info("Search runtime startup: loading Mongo metadata without Qdrant re-indexing")
    datasets, indicators, filters = _load_catalog_metadata()
    cached_faiss = _load_persisted_faiss(indicators)
    if cached_faiss is None:
        cached_faiss = _build_and_persist_faiss(indicators)

    qdrant_ready = False
    if qclient is not None:
        try:
            qdrant_ready = qclient.collection_exists(COLLECTION)
            if qdrant_ready:
                logger.info("Existing Qdrant collection reused without embedding: %s", COLLECTION)
            else:
                logger.warning(
                    "Qdrant collection %s does not exist; FAISS FALLBACK ACTIVE until an explicit index operation",
                    COLLECTION,
                )
        except Exception as exc:
            logger.warning("Qdrant startup check failed; FAISS FALLBACK ACTIVE: %s", exc)

    with SEARCH_INDEX_LOCK:
        DATASETS, INDICATORS, FILTERS = datasets, indicators, filters
        faiss_index = cached_faiss
        USE_QDRANT = qdrant_ready

    logger.info(
        "Search runtime ready without Qdrant sync: datasets=%s indicators=%s primary=%s fallback=FAISS",
        len(datasets),
        len(indicators),
        "Qdrant" if qdrant_ready else "FAISS",
    )


def reload_search_index(product_name=None, sync_qdrant=True):
    """Refresh indexes after a real metadata change or explicit index request.

    Both indexes are built from the same published Mongo product JSON. Qdrant
    is always tried first for live search; FAISS is used only when Qdrant is
    unavailable. Application startup uses ``initialize_search_runtime`` and
    therefore never calls the full Qdrant indexing branch in this function.
    """
    global DATASETS, INDICATORS, FILTERS, faiss_index, USE_QDRANT

    operation = f"product {product_name}" if product_name else "all products"
    logger.info("Search-index sync started for %s: loading Mongo product metadata", operation)
    # This triggers the one-time migration of separate product JSON files only
    # if Mongo product_metadata is empty. Normal application reads use Mongo.
    datasets, indicators, filters = _load_catalog_metadata()

    new_faiss_index = _load_persisted_faiss(indicators)
    if new_faiss_index is None:
        new_faiss_index = _build_and_persist_faiss(indicators)

    qdrant_sync_succeeded = False
    if sync_qdrant and qclient is not None:
        try:
            if product_name:
                logger.info("Qdrant primary sync started for product %s", product_name)
                metadata_manager.index_product(product_name)
            else:
                logger.info(
                    "Mongo metadata ready: products=%s, indicators=%s. Starting full Qdrant primary sync.",
                    len(datasets), len(indicators),
                )
                metadata_manager.index_all_products()
            USE_QDRANT = True
            qdrant_sync_succeeded = True
            logger.info("Qdrant primary sync completed; FAISS remains warm as fallback")
        except Exception as exc:
            # A Qdrant problem must not take down user search. The index above
            # was built first, so switching to FAISS is safe and immediate.
            USE_QDRANT = False
            logger.warning(
                "Qdrant primary sync failed; FAISS FALLBACK ACTIVE for semantic search: %s",
                exc,
                exc_info=True,
            )
    elif not sync_qdrant:
        # index_one_product already completed a Qdrant write. Do not duplicate
        # it; simply refresh the fallback/cache data for the same Mongo state.
        qdrant_sync_succeeded = USE_QDRANT
    else:
        USE_QDRANT = False
        logger.warning("Qdrant client is unavailable; FAISS FALLBACK ACTIVE for semantic search")

    with SEARCH_INDEX_LOCK:
        DATASETS, INDICATORS, FILTERS = datasets, indicators, filters
        faiss_index = new_faiss_index

    logger.info(
        "Search-index sync complete for %s: datasets=%s, indicators=%s, filters=%s, primary=%s, fallback=FAISS",
        operation,
        len(datasets),
        len(indicators),
        len(filters),
        "Qdrant" if qdrant_sync_succeeded else "FAISS",
    )


initialize_search_runtime()

# ================================
# SEARCH
# ================================
def select_cpi_indicator_smart(query, cpi_candidates):
    """
    CPI ke liye smart indicator selection: detail level + year + base year ke basis pe.
    
    Priority Logic:
    1. Explicit base year mention (2010/2012/2024) → determines indicator
    2. Detail level (Item > Group > Division) → determines indicator type
    3. Year → determines which base to use
    
    Key Rule: Detail level is MORE important than year for 2011-2024
    """
    q_lower = query.lower()
    
    # Step 1: Detect explicit base year mention
    base_2010 = 'base 2010' in q_lower or 'base year 2010' in q_lower or '(base 2010)' in q_lower
    base_2012 = 'base 2012' in q_lower or 'base year 2012' in q_lower or '(base 2012)' in q_lower
    base_2024 = 'base 2024' in q_lower or 'base year 2024' in q_lower or '(base 2024)' in q_lower
    
    # Detect "back series" explicit mention
    back_series_explicit = 'back series' in q_lower
    
    # Step 2: Detect year
    year_match = re.search(r'\b(20\d{2})\b', q_lower)
    year = int(year_match.group(1)) if year_match else None
    
    # Step 3: Detect detail level
    # Item-level keywords (most specific)
    item_keywords = [
        'rice', 'wheat', 'atta', 'maida', 'bread', 'biscuit', 'noodles',
        'potato', 'onion', 'tomato', 'cabbage', 'cauliflower', 'carrot',
        'banana', 'apple', 'mango', 'grapes', 'orange',
        'milk', 'curd', 'ghee', 'butter', 'egg', 'paneer',
        'chicken', 'mutton', 'fish', 'prawn',
        'mustard oil', 'groundnut oil', 'refined oil', 'edible oil',
        'arhar', 'moong', 'urd', 'gram', 'masur',
        'sugar', 'gur', 'salt', 'turmeric', 'chilli',
        'petrol', 'diesel', 'lpg', 'kerosene', 'electricity',
        'medicine', 'doctor', 'hospital',
        'school fees', 'tuition', 'books',
        'house rent', 'water charges',
        'shirt', 'saree', 'shoes', 'sandals',
        'soap', 'shampoo', 'toothpaste', 'detergent',
        'bicycle', 'mobile phone', 'handset', 'railway fare'
    ]
    

    group_keywords = [
        'cereals', 'pulses', 'oils and fats', 'vegetables', 'fruits',
        'meat and fish', 'milk and products', 'spices', 'sugar and confectionery',
        'non-alcoholic beverages', 'prepared meals', 'snacks', 'sweets',
        'household goods', 'household requisites', 'personal care and effects',
        'medical care', 'transport and communication', 'recreation and amusement',
        'housing', 'fuel and light', 'clothing', 'footwear', 'clothing and footwear',
        'health', 'transport', 'education', 'miscellaneous',
        'pan, tobacco and intoxicants', 'paan tobacco',
        'general-overall', 'consumer food price'
    ]
    

    division_keywords = [
        'food and beverages',
        'furnishings, household equipment',
        'information and communication',
        'recreation, sport and culture',
        'restaurants and accommodation',
        'personal care, social protection',
        'overall inflation', 'combined inflation', 'rural inflation', 'urban inflation',
        'general index'
    ]
    
    has_item = any(kw in q_lower for kw in item_keywords)
    has_group = any(kw in q_lower for kw in group_keywords)
    has_division = any(kw in q_lower for kw in division_keywords)
    
    # Step 4: Decision logic
    
    # Priority 1: Explicit base year mention
    if base_2010 or base_2012:
        # User explicitly wants 2010/2012 base
        if has_item:
            return next((c for c in cpi_candidates if c["name"] == "Item"), cpi_candidates[0])
        else:
            return next((c for c in cpi_candidates if c["name"] == "Group"), cpi_candidates[0])
    
    if base_2024 or back_series_explicit:
        # User explicitly wants 2024 base or back series
        if back_series_explicit:
            return next((c for c in cpi_candidates if c["name"] == "Back"), cpi_candidates[0])
        else:
            return next((c for c in cpi_candidates if c["name"] == "Current"), cpi_candidates[0])
    

    if not year or year >= 2025:
        # Latest years → Current (has Division/Group/Item hierarchy)
        return next((c for c in cpi_candidates if c["name"] == "Current"), cpi_candidates[0])
    
    # For 2024: Special case - defaults to 2012 base unless "base 2024" mentioned
    if year == 2024:
        # Item-level query → Item indicator (2012 base)
        if has_item:
            return next((c for c in cpi_candidates if c["name"] == "Item"), cpi_candidates[0])
        # Group-level or Division-level → Group indicator (2012 base)
        else:
            return next((c for c in cpi_candidates if c["name"] == "Group"), cpi_candidates[0])
    
    # For 2013-2023: Detail level determines indicator
    if year and 2013 <= year <= 2023:
        # Item-level query → Item indicator (most specific)
        if has_item:
            return next((c for c in cpi_candidates if c["name"] == "Item"), cpi_candidates[0])
        
        # Group-level query → Group indicator
        elif has_group:
            return next((c for c in cpi_candidates if c["name"] == "Group"), cpi_candidates[0])
        
        # Division-level or generic → Back series (2024 base)
        else:
            return next((c for c in cpi_candidates if c["name"] == "Back"), cpi_candidates[0])
    
    # For 2011-2012: Detail level determines indicator
    if year and 2011 <= year <= 2012:
        # Item-level query → Item indicator
        if has_item:
            return next((c for c in cpi_candidates if c["name"] == "Item"), cpi_candidates[0])
        # Group-level or generic → Group indicator (most versatile)
        else:
            return next((c for c in cpi_candidates if c["name"] == "Group"), cpi_candidates[0])
    
    # Fallback: Current (latest)
    return next((c for c in cpi_candidates if c["name"] == "Current"), cpi_candidates[0])


def search_indicators(
    query,
    top_k=25,
    max_products=3,
    raw_query=None,
    qdrant_search_client=None,
    qdrant_collection=None,
):
    """Search Qdrant first, or FAISS only when the primary is unavailable."""
    global USE_QDRANT

  
    q_vec = bi_encoder.encode([clean_text(query)], convert_to_numpy=True)
    q_vec /= np.clip(np.linalg.norm(q_vec, axis=1, keepdims=True), 1e-12, None)

    active_qdrant_client = qclient if qdrant_search_client is None else qdrant_search_client
    active_qdrant_collection = qdrant_collection or COLLECTION

    candidates = None
    if USE_QDRANT and active_qdrant_client is not None:
        try:
           
            query_result = active_qdrant_client.query_points(
                collection_name=active_qdrant_collection,
                query=q_vec[0].tolist(),
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            
            candidates = [dict(point.payload or {}) for point in query_result.points]
            logger.info("Semantic search backend=Qdrant collection=%s", active_qdrant_collection)
        except Exception as exc:
         
            if active_qdrant_client is qclient:
                USE_QDRANT = False
            logger.warning(
                "Qdrant search failed for collection=%s; using FAISS fallback: %s",
                active_qdrant_collection,
                exc,
                exc_info=True,
            )

    if candidates is None:
        with SEARCH_INDEX_LOCK:
            active_faiss_index = faiss_index
            fallback_indicators = list(INDICATORS)
        if active_faiss_index is None:
            raise RuntimeError("Semantic search is unavailable: neither Qdrant nor FAISS is ready")
        _, indexes = active_faiss_index.search(q_vec.astype("float32"), top_k)
        
        candidates = [dict(fallback_indicators[index]) for index in indexes[0] if index >= 0]
        logger.warning("Semantic search backend=FAISS FALLBACK qdrant_status=unavailable")

   
    if not candidates:
        logger.warning("Semantic search returned no indicator candidates")
        return []

    scores = cross_encoder.predict([
        (query, c["name"] + " " + c.get("description", ""))
        for c in candidates
    ])
    for i, c in enumerate(candidates):
        c["score"] = float(scores[i])

 
    _intent_src = (raw_query or "").lower()
    _udise_intent = (raw_query is not None) and (
        bool(re.search(r"\budise\b", _intent_src) or "udise+" in _intent_src)
    )
    if _udise_intent:
        for c in candidates:
            if c.get("product") == "UDISE":
                c["score"] = c["score"] + 5.0
    
    if raw_query:
        detected_state = detect_state_from_query(raw_query)
        q_lower = raw_query.lower()
        
        logger.info(f"[NAS Boost Debug] raw_query='{raw_query}', detected_state='{detected_state}'")
        
        # Check for regional keywords
        regional_keywords = [
            "state", "states", "regional", "gsdp", "gsva", "nsdp", "nsva",
            "pcnsdp", "pcgsdp", "state-wise", "statewise", "state level",
            "state-level", "gross state", "net state", "per capita state"
        ]
        has_regional_keyword = any(kw in q_lower for kw in regional_keywords)
        
        logger.info(f"[NAS Boost Debug] has_regional_keyword={has_regional_keyword}, regional_keywords_in_query={[kw for kw in regional_keywords if kw in q_lower]}")
        
        if detected_state or has_regional_keyword:
            logger.info(f"[NAS Boost] TRIGGERED - State: {detected_state}, Regional keyword: {has_regional_keyword}")
            state_indicator_keywords = ["gross state", "net state", "gsdp", "nsdp", "pcgsdp", "pcnsdp", "state domestic"]
            boost_count = 0
            for c in candidates:
                if c.get("product") == "NAS":
                    indicator_lower = c.get("name", "").lower()
                    # Boost if indicator contains state-level keywords
                    if any(kw in indicator_lower for kw in state_indicator_keywords):
                        old_score = c["score"]
                        c["score"] = c["score"] + 10.0  # Strong boost to ensure state indicators rank above national
                        logger.info(f"[NAS Boost] Boosted '{c.get('name')}' from {old_score:.3f} to {c['score']:.3f}")
                        boost_count += 1
            logger.info(f"[NAS Boost] Total indicators boosted: {boost_count}")
        else:
            logger.info(f"[NAS Boost] NOT triggered - No state or regional keywords detected")

    candidates.sort(key=lambda x: x["score"], reverse=True)

    # CPI conflict resolve ONLY if both present
    candidates = resolve_cpi_conflict(candidates, query)

    # NEW: CPI smart indicator selection
    # Group CPI indicators together and pick the right one based on year + detail level
    cpi_candidates = [c for c in candidates if c["product"] == "CPI"]
    if len(cpi_candidates) > 1:
        # Multiple CPI indicators found - use smart selection
        best_cpi = select_cpi_indicator_smart(query, cpi_candidates)
        # Remove all CPI candidates and add back only the best one
        candidates = [c for c in candidates if c["product"] != "CPI"]
        # Insert at position where first CPI was (maintain relative scoring)
        first_cpi_idx = next((i for i, c in enumerate(candidates) if c.get("product") != "CPI"), 0)
        candidates.insert(first_cpi_idx, best_cpi)

    seen, final = set(), []
    for c in candidates:

        if c["product"] not in seen:
            seen.add(c["product"])
            final.append(c)
        if len(final) == max_products:
            break


    return final


def _search_dataset_only(query, product_codes):
    """Sirf given dataset(s) ke indicators mein search karo. Best matching indicator return, nahi mile to None."""
    if isinstance(product_codes, str):
        product_codes = (product_codes,)
    indicators = [i.copy() for i in INDICATORS if i["product"] in product_codes]
    if not indicators:
        return None
    pairs = [(query, c["name"] + " " + c.get("description", "")) for c in indicators]
    scores = cross_encoder.predict(pairs)
    for i, c in enumerate(indicators):
        c["score"] = float(scores[i])
    return max(indicators, key=lambda x: x["score"])


def _search_wpi_only(query):
    """WPI dataset ke andar sirf search karo. Force-include ke liye use hota hai."""
    return _search_dataset_only(query, "WPI")


def _search_ec_only(query):
    """EC4/EC5/EC6 ke andar search karo. Economic Census force-include ke liye."""
    return _search_dataset_only(query, ("EC4", "EC5", "EC6"))


###################query capture 


import uuid
from datetime import datetime

# Legacy JSONL log destination. JSONL writes below are intentionally disabled;
# new completed search interactions are stored in MongoDB instead.

analytics_engines = {
    "dev": AnalyticsEngine(environment="dev"),
    "prod": AnalyticsEngine(
        mongo_uri=os.environ.get("PRODUCTION_MONGO_URI"),
        mongo_database=os.environ.get("PRODUCTION_MONGO_DB_NAME", "semantic_search"),
        mongo_collection=os.environ.get("PRODUCTION_MONGO_COLLECTION", "interactions"),
        product_collection=os.environ.get("PRODUCTION_MONGO_PRODUCT_COLLECTION", "product_metadata"),
        environment="prod",
        use_default_env=False,
    ),
}

EVALUATION_CONTEXT = threading.local()
# The manager creates its two Mongo collections and indexes automatically on
# first connection. A history outage never stops a completed evaluation CSV.
evaluation_history_manager = EvaluationHistoryManager()
try:
    evaluation_history_manager.ensure_collections()
except EvaluationHistoryUnavailableError as exc:
    logger.warning("MongoDB evaluation history is unavailable at startup: %s", exc)
evaluation_runner = EvaluationRunner(
    os.path.join(BASE_DIR, "evaluation_results"),
    history_manager=evaluation_history_manager,
)

def save_query_log(raw_query, rewritten_query, response_json, interaction_engine):
    """Save a successful interaction using the engine explicitly provided by its route."""

    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow(),
        "raw_query": raw_query,
        "rewritten_query": rewritten_query,
        "response": response_json
    }

    try:
        interaction_engine.save_interaction(record)
    except AnalyticsMongoUnavailableError as exc:
        logger.error("Interaction was not saved to MongoDB: %s", exc)


# ================================
# FLASK
# ================================
app = Flask(__name__, template_folder="templates")

# Basic Flask configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(32).hex())
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max request size
app.config['JSON_SORT_KEYS'] = False
app.config['AUTH_SESSION_COOKIE'] = os.environ.get('AUTH_SESSION_COOKIE', 'semantic_search_session')
app.config['AUTH_CSRF_COOKIE'] = os.environ.get('AUTH_CSRF_COOKIE', 'semantic_search_csrf')
app.config['AUTH_COOKIE_SECURE'] = os.environ.get(
    'AUTH_COOKIE_SECURE',
    'true' if os.environ.get('FLASK_ENV', 'production').lower() == 'production' else 'false',
).lower() == 'true'

auth_manager = AuthManager()
app.extensions['auth_manager'] = auth_manager


def _configured_cors_origins():
    """Return a strict origin allow-list suitable for credential cookies."""
    configured = [origin.strip() for origin in os.environ.get('CORS_ORIGINS', '').split(',') if origin.strip()]
    if configured:
        return configured
    return [
        'http://localhost:3000', 'http://localhost:5173', 'http://localhost:5174',
        'http://localhost:5005', 'http://127.0.0.1:5173', 'http://10.75.8.2:5173',
    ]

# Enable CORS for frontend
CORS_ORIGINS = _configured_cors_origins()
CORS(app, resources={
    r"/auth/*": {
        "origins": CORS_ORIGINS,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "X-CSRF-Token"],
    },
    r"/api/*": {
        "origins": CORS_ORIGINS,
        "methods": ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "X-CSRF-Token"]
    },
    r"/search/*": {
        "origins": CORS_ORIGINS,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "X-CSRF-Token"]
    },
    r"/analytics/*": {
        "origins": CORS_ORIGINS,
        "methods": ["GET", "OPTIONS"],
        "allow_headers": ["Content-Type", "X-CSRF-Token"]
    },
    r"/health": {"origins": CORS_ORIGINS}
}, supports_credentials=True)

# Initialize rate limiter (optional - can be disabled)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour"],
    storage_uri="memory://",
    enabled=os.environ.get('RATELIMIT_ENABLED', 'false').lower() == 'true'
)

logger.info("Flask app initialized")


# ================================
# AUTHENTICATION AND USER MANAGEMENT
# ================================
def _request_audit(action, resource_type, resource_id=None, success=True, metadata=None, error_message=None, actor=None):
    """Write a safe audit event without ever including passwords or session tokens."""
    auth_manager.audit_event(
        actor=actor if actor is not None else getattr(g, 'current_user', None),
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        success=success,
        method=request.method,
        path=request.path,
        ip_address=request.remote_addr,
        user_agent=request.headers.get('User-Agent'),
        metadata=metadata,
        error_message=error_message,
    )


def _set_auth_cookies(response, session_token, csrf_token, expires_at):
    """Issue an HttpOnly session cookie and a separate double-submit CSRF cookie."""
    max_age = max(int((expires_at - datetime.now(expires_at.tzinfo)).total_seconds()), 1)
    cookie_kwargs = {
        'max_age': max_age,
        'secure': app.config['AUTH_COOKIE_SECURE'],
        'samesite': 'Lax',
        'path': '/',
    }
    response.set_cookie(app.config['AUTH_SESSION_COOKIE'], session_token, httponly=True, **cookie_kwargs)
    response.set_cookie(app.config['AUTH_CSRF_COOKIE'], csrf_token, httponly=False, **cookie_kwargs)
    return response


def _clear_auth_cookies(response):
    cookie_kwargs = {'secure': app.config['AUTH_COOKIE_SECURE'], 'samesite': 'Lax', 'path': '/'}
    response.delete_cookie(app.config['AUTH_SESSION_COOKIE'], **cookie_kwargs)
    response.delete_cookie(app.config['AUTH_CSRF_COOKIE'], **cookie_kwargs)
    return response


@app.route('/auth/login', methods=['POST'])
@limiter.limit('5 per minute')
def auth_login():
    """Create a MongoDB-backed browser session after verifying Argon2 password."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({'error': 'A JSON request body is required'}), 400
    username = payload.get('username')
    try:
        user, session_token, csrf_token, expires_at = auth_manager.authenticate(username, payload.get('password'))
    except AuthenticationError:
        _request_audit(
            'auth.login', 'user', success=False,
            metadata={'username': str(username or '')[:64]},
            error_message='Invalid username or password', actor=None,
        )
        return jsonify({'error': 'Invalid username or password'}), 401
    except AuthenticationUnavailableError as exc:
        logger.error('Login unavailable: %s', exc)
        return jsonify({'error': 'Authentication service is unavailable'}), 503

    _request_audit('auth.login', 'user', resource_id=user['id'], success=True, actor=user)
    response = jsonify({'user': user, 'expires_at': expires_at.isoformat()})
    return _set_auth_cookies(response, session_token, csrf_token, expires_at)


@app.route('/auth/logout', methods=['POST'])
@require_login
def auth_logout():
    """Revoke the current session server-side and delete both browser cookies."""
    try:
        auth_manager.revoke_session(request.cookies.get(app.config['AUTH_SESSION_COOKIE']))
    except AuthenticationUnavailableError as exc:
        logger.error('Logout unavailable: %s', exc)
        return jsonify({'error': 'Authentication service is unavailable'}), 503
    _request_audit('auth.logout', 'user', resource_id=g.current_user['id'], success=True)
    return _clear_auth_cookies(jsonify({'success': True}))


@app.route('/auth/me', methods=['GET'])
@require_login
def auth_me():
    """Return the signed-in account for frontend startup and route guards."""
    return jsonify({'user': g.current_user})


@app.route('/auth/change-password', methods=['POST'])
@require_login
@limiter.limit('5 per minute')
def auth_change_password():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({'error': 'A JSON request body is required'}), 400
    try:
        auth_manager.change_password(
            g.current_user['id'],
            payload.get('current_password'),
            payload.get('new_password'),
        )
    except AuthenticationError as exc:
        return jsonify({'error': str(exc)}), 400
    except AuthenticationUnavailableError as exc:
        logger.error('Password change unavailable: %s', exc)
        return jsonify({'error': 'Authentication service is unavailable'}), 503
    _request_audit('auth.password_changed', 'user', resource_id=g.current_user['id'], success=True)
    # Password changes revoke every existing session, including this one.  The
    # user returns to the login page with the newly chosen password.
    return _clear_auth_cookies(jsonify({'success': True, 'message': 'Password changed. Please log in again.'}))


@app.route('/api/admin/users', methods=['GET'])
@require_roles('admin')
@limiter.limit('60 per minute')
def admin_list_users():
    try:
        return jsonify({'users': auth_manager.list_users(), 'roles': list(ROLES)})
    except AuthenticationUnavailableError as exc:
        logger.error('Could not list users: %s', exc)
        return jsonify({'error': 'Authentication service is unavailable'}), 503


@app.route('/api/admin/users', methods=['POST'])
@require_roles('admin')
@limiter.limit('10 per minute')
def admin_create_user():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({'error': 'A JSON request body is required'}), 400
    try:
        user = auth_manager.create_user(
            payload.get('username'), payload.get('password'),
            payload.get('roles', payload.get('role', 'viewer')), payload.get('email'),
            actor=g.current_user, must_change_password=True,
        )
        return jsonify({'user': user}), 201
    except AuthenticationError as exc:
        return jsonify({'error': str(exc)}), 400
    except AuthenticationUnavailableError as exc:
        logger.error('Could not create user: %s', exc)
        return jsonify({'error': 'Authentication service is unavailable'}), 503


@app.route('/api/admin/users/<user_id>', methods=['PATCH'])
@require_roles('admin')
@limiter.limit('20 per minute')
def admin_update_user(user_id):
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({'error': 'A JSON request body is required'}), 400
  
    if user_id == g.current_user['id']:
        if payload.get('is_active') is False:
            return jsonify({'error': 'You cannot disable your own account'}), 400
        requested_roles = payload.get('roles', payload.get('role'))
        if requested_roles is not None and not auth_manager.role_has_any(requested_roles, ('admin',)):
            return jsonify({'error': 'You cannot remove your own admin role'}), 400
    try:
        user = auth_manager.update_user(
            user_id,
            roles=payload['roles'] if 'roles' in payload else None,
            role=payload['role'] if 'role' in payload else None,
            email=payload['email'] if 'email' in payload else None,
            is_active=payload['is_active'] if 'is_active' in payload else None,
            actor=g.current_user,
        )
        return jsonify({'user': user})
    except KeyError:
        return jsonify({'error': 'User not found'}), 404
    except AuthenticationError as exc:
        return jsonify({'error': str(exc)}), 400
    except AuthenticationUnavailableError as exc:
        logger.error('Could not update user: %s', exc)
        return jsonify({'error': 'Authentication service is unavailable'}), 503


@app.route('/api/admin/users/<user_id>', methods=['DELETE'])
@require_roles('admin')
@limiter.limit('10 per minute')
def admin_delete_user(user_id):
    try:
        auth_manager.delete_user(user_id, actor=g.current_user)
        return jsonify({'success': True})
    except KeyError:
        return jsonify({'error': 'User not found'}), 404
    except AuthorizationError as exc:
        return jsonify({'error': str(exc)}), 400
    except AuthenticationUnavailableError as exc:
        logger.error('Could not delete user: %s', exc)
        return jsonify({'error': 'Authentication service is unavailable'}), 503


@app.route('/api/admin/users/<user_id>/reset-password', methods=['POST'])
@require_roles('admin')
@limiter.limit('5 per minute')
def admin_reset_password(user_id):
    if user_id == g.current_user['id']:
        return jsonify({'error': 'Use Change password for your own account'}), 400
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({'error': 'A JSON request body is required'}), 400
    try:
        auth_manager.reset_password(user_id, payload.get('new_password'), actor=g.current_user)
        return jsonify({'success': True, 'message': 'Password reset. The user must choose a new password after login.'})
    except KeyError:
        return jsonify({'error': 'User not found'}), 404
    except AuthenticationError as exc:
        return jsonify({'error': str(exc)}), 400
    except AuthenticationUnavailableError as exc:
        logger.error('Could not reset password: %s', exc)
        return jsonify({'error': 'Authentication service is unavailable'}), 503


# ---------------------------------------------------------------------------
# ADMIN-ONLY PRODUCTION MIGRATION API
# ---------------------------------------------------------------------------
@app.route("/api/admin/prod-migration/products", methods=["GET"])
@require_roles("admin")
@limiter.limit("60 per minute")
def list_production_migration_products():
    """Return published dev products for the admin-only migration dropdown."""
    try:
        return jsonify({"products": production_migration_manager.list_products()})
    except Exception as exc:
        logger.error("Could not load production migration products: %s", exc, exc_info=True)
        return jsonify({"error": "Unable to load products for production migration"}), 500


@app.route("/api/admin/prod-migration/products/<product_name>", methods=["GET"])
@require_roles("admin")
@limiter.limit("30 per minute")
def get_production_migration_product_preview(product_name):
    """Show one exact source JSON document in the UI; this endpoint never edits it."""
    try:
        return jsonify({"product": production_migration_manager.get_product_preview(product_name)})
    except KeyError:
        return jsonify({"error": f"Unknown product: {product_name}"}), 404
    except ProductionMigrationError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("Could not load production migration preview %s: %s", product_name, exc, exc_info=True)
        return jsonify({"error": "Unable to load product preview"}), 500


@app.route("/api/admin/prod-migration/request", methods=["POST"])
@require_roles("admin")
@limiter.limit("15 per hour")
def request_production_migration_approval():
    """Snapshot one reviewed product and e-mail the owner-only approval code."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "A JSON request body is required"}), 400
    try:
        promotion = production_migration_manager.request_approval(
            payload.get("product_name"), payload.get("confirmed") is True, g.current_user
        )
        _request_audit(
            "production_migration.approval_requested",
            "product",
            resource_id=promotion["product_name"],
            metadata={"request_id": promotion["request_id"], "source_version": promotion["source_version"]},
        )
        return jsonify({"request": promotion}), 202
    except ProductionMigrationConfigurationError as exc:
        return jsonify({"error": str(exc)}), 503
    except ProductionMigrationPermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    except KeyError:
        return jsonify({"error": "Unknown product"}), 404
    except ProductionMigrationError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("Could not request production migration approval: %s", exc, exc_info=True)
        return jsonify({"error": "Unable to request production migration approval"}), 500


@app.route("/api/admin/prod-migration/<request_id>/verify-code", methods=["POST"])
@require_roles("admin")
@limiter.limit("10 per hour")
def verify_production_migration_code(request_id):
    """Consume a valid one-time owner code and begin background promotion."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "A JSON request body is required"}), 400
    try:
        promotion = production_migration_manager.verify_code_and_start(
            request_id, payload.get("code"), g.current_user
        )
        _request_audit(
            "production_migration.approved",
            "product",
            resource_id=promotion["product_name"],
            metadata={"request_id": promotion["request_id"]},
        )
        return jsonify({"request": promotion}), 202
    except ProductionMigrationConfigurationError as exc:
        return jsonify({"error": str(exc)}), 503
    except ProductionMigrationPermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    except KeyError:
        return jsonify({"error": "Production migration request not found"}), 404
    except ProductionMigrationError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("Could not verify production migration code %s: %s", request_id, exc, exc_info=True)
        return jsonify({"error": "Unable to verify production migration code"}), 500


@app.route("/api/admin/prod-migration/<request_id>/cancel", methods=["POST"])
@require_roles("admin")
@limiter.limit("10 per hour")
def cancel_production_migration_request(request_id):
    """Cancel only the current admin's unapproved migration request."""
    try:
        promotion = production_migration_manager.cancel_request(request_id, g.current_user)
        _request_audit(
            "production_migration.cancelled",
            "product",
            resource_id=promotion["product_name"],
            metadata={"request_id": promotion["request_id"]},
        )
        return jsonify({"request": promotion})
    except ProductionMigrationPermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    except KeyError:
        return jsonify({"error": "Production migration request not found"}), 404
    except ProductionMigrationError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("Could not cancel production migration request %s: %s", request_id, exc, exc_info=True)
        return jsonify({"error": "Unable to cancel production migration request"}), 500


@app.route("/api/admin/prod-migration/<request_id>/status", methods=["GET"])
@require_roles("admin")
@limiter.limit("120 per minute")
def get_production_migration_status(request_id):
    """Return durable promotion progress after navigation or page refresh."""
    try:
        return jsonify({"request": production_migration_manager.get_request(request_id, g.current_user)})
    except ProductionMigrationPermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    except KeyError:
        return jsonify({"error": "Production migration request not found"}), 404
    except ProductionMigrationError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("Could not load production migration status %s: %s", request_id, exc, exc_info=True)
        return jsonify({"error": "Unable to load production migration status"}), 500


@app.route("/api/metadata/list", methods=["GET"])
@require_roles('metadata_editor', 'metadata_publisher', 'admin')
@limiter.limit("60 per minute")
def list_metadata_products():
    """Return published Mongo product codes; the UI loads one selected product."""
    try:
        return jsonify({"datasets": metadata_manager.list_datasets()})
    except Exception as exc:
        logger.error("Could not list metadata products: %s", exc, exc_info=True)
        return jsonify({"error": "Unable to load product catalog"}), 500


@app.route("/api/metadata/dataset/<dataset_name>", methods=["GET"])
@require_roles('metadata_editor', 'metadata_publisher', 'admin')
@limiter.limit("60 per minute")
def get_metadata_dataset(dataset_name):
    """Load exactly one published Mongo product for the metadata editor."""
    try:
        return jsonify(metadata_manager.get_dataset(dataset_name))
    except KeyError:
        return jsonify({"error": f"Unknown product: {dataset_name}"}), 404
    except Exception as exc:
        logger.error("Could not load metadata product %s: %s", dataset_name, exc, exc_info=True)
        return jsonify({"error": "Unable to load product metadata"}), 500


@app.route("/api/metadata/dataset/<dataset_name>/download", methods=["GET"])
@require_roles('metadata_editor', 'metadata_publisher', 'admin')
@limiter.limit("20 per minute")
def download_metadata_dataset(dataset_name):
    """Download exactly one clean product JSON document for offline editing."""
    try:
        exported = metadata_manager.download_dataset_source(dataset_name)
    except KeyError:
        return jsonify({"error": f"Unknown product: {dataset_name}"}), 404
    except Exception as exc:
        logger.error("Could not export metadata product %s: %s", dataset_name, exc, exc_info=True)
        return jsonify({"error": "Unable to export product metadata"}), 500

    response = make_response(exported["source_json"])
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    response.headers["Content-Disposition"] = f'attachment; filename="{exported["product_name"]}.json"'
    # The editor retains this as optimistic-lock context when it later uploads.
    response.headers["X-Metadata-Version"] = exported["version"]
    return response


@app.route("/api/metadata/dataset/<dataset_name>/replace", methods=["POST"])
@require_roles('metadata_publisher', 'admin')
@limiter.limit("10 per minute")
def replace_metadata_dataset(dataset_name):
    """Queue a confirmed whole-product JSON replacement and re-index it."""
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "Choose a JSON file to upload"}), 400
    if not upload.filename.lower().endswith(".json"):
        return jsonify({"error": "Upload a .json metadata file"}), 400
    try:
        source_json = upload.read().decode("utf-8")
    except UnicodeDecodeError:
        return jsonify({"error": "The uploaded JSON file must be UTF-8"}), 400
    if not source_json.strip():
        return jsonify({"error": "The uploaded JSON file is empty"}), 400

    expected_version = request.form.get("expected_version")
    actor = getattr(g, "current_user", None)
    user_ip = request.remote_addr
    try:
        current = metadata_manager.get_dataset(dataset_name)
        parsed = json.loads(source_json)
        if not isinstance(parsed, dict):
            raise ValueError("Uploaded JSON must contain an object")
        product_name = current["product_name"]
        datasets = parsed.get("datasets")
        if not isinstance(datasets, dict) or list(datasets.keys()) != [product_name]:
            raise ValueError(f"Uploaded JSON must contain exactly one product at datasets.{product_name}")
        valid, message = metadata_manager.validate_dataset_structure(datasets[product_name])
        if not valid:
            raise ValueError(message)
        job = _start_metadata_job(
            "replace",
            dataset_name,
            actor,
            lambda: metadata_manager.replace_dataset_from_source_json(
                dataset_name,
                source_json,
                expected_version=expected_version,
                user_ip=user_ip,
                actor=actor,
            ),
        )
        return jsonify({"success": True, "job": job, "job_id": job["job_id"]}), 202
    except json.JSONDecodeError as exc:
        return jsonify({"error": f"Uploaded file is not valid JSON: {exc.msg}"}), 400
    except KeyError:
        return jsonify({"error": f"Unknown product: {dataset_name}"}), 404
    except MetadataJobUnavailableError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("Could not queue JSON replacement for %s: %s", dataset_name, exc, exc_info=True)
        return jsonify({"error": "Unable to start JSON replacement"}), 500


def _start_metadata_job(operation, dataset_name, actor, mongo_save):
    """Start a durable update/publish job with clear UI progress stages.

    ``mongo_save`` performs just the product MongoDB write.  Qdrant/FAISS
    refresh happens after that write, so the UI can accurately say whether a
    product is saved but still waiting for semantic-search embeddings.
    """
    def worker(report):
        report("saving_to_mongo", 20, "Saving the ordered product JSON to MongoDB")
        result = mongo_save()
        report("mongo_saved", 45, "MongoDB saved. Starting semantic-search embeddings")
        try:
            report("embedding", 60, f"Refreshing Qdrant and fallback index for {dataset_name}")
            reload_search_index(dataset_name)
        except Exception as exc:
            raise RuntimeError(
                "Metadata was saved in MongoDB, but semantic-search indexing failed. "
                "Retry indexing for this product; restarting the service does not re-embed products."
            ) from exc
        report("index_ready", 90, "Semantic-search index refreshed")
        result["index_refreshed"] = True
        result["note"] = f"Only {dataset_name} was refreshed in the semantic search index."
        return result

    return metadata_job_manager.start_job(operation, dataset_name, actor, worker)


@app.route("/api/metadata/save", methods=["POST"])
@require_roles('metadata_publisher', 'admin')
@limiter.limit("10 per minute")
def save_metadata_dataset():
    """Queue one product update and return its durable progress-job ID."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "A JSON request body is required"}), 400

    dataset_name = payload.get("changed_dataset")
    dataset_data = payload.get("dataset_data")
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        return jsonify({"error": "changed_dataset is required"}), 400
    if not isinstance(dataset_data, dict):
        return jsonify({"error": "dataset_data must be an object"}), 400
    dataset_name = dataset_name.strip()
    actor = getattr(g, "current_user", None)
    user_ip = request.remote_addr
    valid, message = metadata_manager.validate_dataset_structure(dataset_data)
    if not valid:
        return jsonify({"error": message}), 400
    try:
        # Check before creating a job. An unchanged product must not spend
        # time on Mongo writes, Qdrant embeddings, or FAISS refreshes.
        change_check = metadata_manager.check_dataset_changes(
            dataset_name, dataset_data, payload.get("expected_version")
        )
        if not change_check["changed"]:
            return jsonify({
                "success": True,
                "no_changes": True,
                "message": f"No changes detected for {change_check['product_name']}. Search indexing was not run.",
                "version": change_check["version"],
            })
        job = _start_metadata_job(
            "update",
            dataset_name,
            actor,
            lambda: metadata_manager.update_dataset_preserving_structure(
                dataset_name=dataset_name,
                new_dataset_data=dataset_data,
                change_summary=payload.get("change_summary"),
                user_ip=user_ip,
                expected_version=payload.get("expected_version"),
                actor=actor,
            ),
        )
        return jsonify({"success": True, "job": job, "job_id": job["job_id"]}), 202
    except KeyError:
        return jsonify({"error": f"Unknown product: {dataset_name}"}), 404
    except MetadataJobUnavailableError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("Could not queue metadata update for %s: %s", dataset_name, exc, exc_info=True)
        return jsonify({"error": "Unable to start metadata update"}), 500


@app.route("/api/metadata/jobs/<job_id>", methods=["GET"])
@require_roles('metadata_publisher', 'admin')
@limiter.limit("60 per minute")
def get_metadata_job(job_id):
    """Return a user-authorized metadata job's latest MongoDB progress."""
    try:
        return jsonify({"job": metadata_job_manager.get_job(job_id, getattr(g, "current_user", None))})
    except KeyError:
        return jsonify({"error": "Metadata job not found"}), 404
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    except MetadataJobUnavailableError as exc:
        return jsonify({"error": str(exc)}), 503


@app.route("/api/internal/products/<product_name>/index", methods=["POST"])
@require_roles('admin')
@limiter.limit("10 per minute")
def index_one_product(product_name):
    """Internal transport-only endpoint for product-scoped Mongo to Qdrant indexing."""
    global USE_QDRANT
    try:
        result = metadata_manager.index_product(product_name)
        # A successful manual index confirms that Qdrant has recovered, so
        # normal semantic searches can use it as the primary backend again.
        USE_QDRANT = True
        logger.info("Qdrant primary backend restored by manual product index: %s", product_name)
        # Refresh product/filter response caches without sending the same
        # product to Qdrant a second time in this request.
        reload_search_index(product_name=product_name, sync_qdrant=False)
        return jsonify(result)
    except KeyError:
        return jsonify({"error": f"Unknown product: {product_name}"}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("Could not index product %s: %s", product_name, exc, exc_info=True)
        return jsonify({"error": "Unable to index product"}), 500


@app.route("/api/metadata/history", methods=["GET"])
@require_roles('metadata_editor', 'metadata_publisher', 'admin')
@limiter.limit("60 per minute")
def metadata_history():
    try:
        limit = min(max(int(request.args.get("limit", 50)), 1), 500)
    except ValueError:
        return jsonify({"error": "limit must be an integer"}), 400
    return jsonify({"changes": metadata_manager.get_metadata_history(limit=limit)})


# Analytics is read-only in this phase and reads MongoDB interactions only.
def _selected_analytics_engine():
    environment = request.args.get("environment", "dev").strip().lower()
    if environment not in analytics_engines:
        raise ValueError("environment must be dev or prod")
    return environment, analytics_engines[environment]


@app.route("/analytics/periods", methods=["GET"])
@require_roles('analyst', 'admin')
@limiter.limit("60 per minute")
def analytics_periods():
    """Return the interaction-log months available in 2026."""
    try:
        environment, engine = _selected_analytics_engine()
        return jsonify({
            "environment": environment,
            "year": 2026,
            "periods": engine.get_available_periods(year=2026),
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except AnalyticsMongoUnavailableError as exc:
        return jsonify({"error": str(exc), "source": "mongodb"}), 503


@app.route("/analytics/daily", methods=["GET"])
@require_roles('analyst', 'admin')
@limiter.limit("60 per minute")
def analytics_daily():
    try:
        environment, engine = _selected_analytics_engine()
        result = engine.get_daily_analytics(request.args.get("date"))
        result["environment"] = environment
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except AnalyticsMongoUnavailableError as exc:
        return jsonify({"error": str(exc), "source": "mongodb"}), 503


@app.route("/analytics/weekly", methods=["GET"])
@require_roles('analyst', 'admin')
@limiter.limit("60 per minute")
def analytics_weekly():
    try:
        environment, engine = _selected_analytics_engine()
        result = engine.get_weekly_analytics(request.args.get("start_date"))
        result["environment"] = environment
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except AnalyticsMongoUnavailableError as exc:
        return jsonify({"error": str(exc), "source": "mongodb"}), 503


@app.route("/analytics/monthly", methods=["GET"])
@require_roles('analyst', 'admin')
@limiter.limit("60 per minute")
def analytics_monthly():
    try:
        year = int(request.args["year"]) if request.args.get("year") else None
        month = int(request.args["month"]) if request.args.get("month") else None
    except ValueError:
        return jsonify({"error": "year and month must be integers"}), 400
    if month is not None and not 1 <= month <= 12:
        return jsonify({"error": "month must be between 1 and 12"}), 400
    try:
        environment, engine = _selected_analytics_engine()
        result = engine.get_monthly_analytics(year, month)
        result["environment"] = environment
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except AnalyticsMongoUnavailableError as exc:
        return jsonify({"error": str(exc), "source": "mongodb"}), 503


@app.route("/analytics/custom", methods=["GET"])
@require_roles('analyst', 'admin')
@limiter.limit("60 per minute")
def analytics_custom():
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    if not start_date or not end_date:
        return jsonify({"error": "start_date and end_date are required"}), 400
    try:
        environment, engine = _selected_analytics_engine()
        analytics = engine.get_custom_range_analytics(start_date, end_date)
        analytics["environment"] = environment
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except AnalyticsMongoUnavailableError as exc:
        return jsonify({"error": str(exc), "source": "mongodb"}), 503
    return jsonify(analytics), 400 if "error" in analytics else 200


def _analytics_csv_range():
    """Resolve the date range used by the Analytics UI download button."""
    date_value = request.args.get("date")
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    year_value = request.args.get("year")
    month_value = request.args.get("month")
    try:
        if date_value:
            start = datetime.strptime(date_value, "%Y-%m-%d")
            return start, start + timedelta(days=1), f"daily_{date_value}"
        if year_value and month_value:
            year, month = int(year_value), int(month_value)
            if not 1 <= month <= 12:
                raise ValueError("month must be between 1 and 12")
            start = datetime(year, month, 1)
            end = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)
            return start, end, f"monthly_{year}-{month:02d}"
        if start_date and end_date:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
            if end <= start:
                raise ValueError("end_date must be on or after start_date")
            return start, end, f"range_{start_date}_to_{end_date}"
    except ValueError as exc:
        raise ValueError(f"Invalid analytics export date: {exc}") from exc
    raise ValueError("Provide date, year and month, or start_date and end_date")


def _safe_csv_value(value):
    """Serialize nested response data and neutralize spreadsheet formulas."""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    text = "" if value is None else str(value)
    return f"'{text}" if text.startswith(("=", "+", "-", "@")) else text


@app.route("/analytics/download/csv", methods=["GET"])
@require_roles('analyst', 'admin')
@limiter.limit("20 per minute")
def analytics_download_csv():
    try:
        environment, engine = _selected_analytics_engine()
        start, end, filename_suffix = _analytics_csv_range()
        logs = engine.get_interactions_for_range(start, end)
    except AnalyticsMongoUnavailableError as exc:
        return jsonify({"error": str(exc), "source": "mongodb"}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "timestamp", "raw_query", "rewritten_query", "response"])
    for log in logs:
        writer.writerow([
            _safe_csv_value(log.get("id")),
            _safe_csv_value(log.get("timestamp")),
            _safe_csv_value(log.get("raw_query")),
            _safe_csv_value(log.get("rewritten_query")),
            _safe_csv_value(log.get("response")),
        ])

    logger.info("MongoDB analytics CSV export: %s records for %s", len(logs), filename_suffix)
    response = make_response(output.getvalue())
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = f'attachment; filename="interactions_{environment}_{filename_suffix}.csv"'
    return response


def _product_activity_range():
    """Use the same daily/weekly/monthly/custom range as interaction analytics."""
    has_range = any(request.args.get(key) for key in ("date", "start_date", "end_date", "year", "month"))
    if not has_range:
        return None, None, "all"
    return _analytics_csv_range()


@app.route("/analytics/products", methods=["GET"])
@require_roles('analyst', 'admin')
@limiter.limit("60 per minute")
def analytics_products():
    try:
        environment, engine = _selected_analytics_engine()
        start, end, period = _product_activity_range()
        result = engine.get_product_activity(start, end)
        result["period"] = period
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except AnalyticsMongoUnavailableError as exc:
        return jsonify({"error": str(exc), "source": "mongodb"}), 503


@app.route("/analytics/products/download/csv", methods=["GET"])
@require_roles('analyst', 'admin')
@limiter.limit("20 per minute")
def analytics_products_download_csv():
    try:
        environment, engine = _selected_analytics_engine()
        start, end, period = _product_activity_range()
        products = engine.get_product_activity(start, end)["products"]
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except AnalyticsMongoUnavailableError as exc:
        return jsonify({"error": str(exc), "source": "mongodb"}), 503

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["S.No.", "Product", "Added Date", "Last Updated", "Status", "Environment"])
    for product in products:
        writer.writerow([
            product["serial_number"], _safe_csv_value(product["product"]),
            product["created_at"] or "", product["updated_at"] or "",
            product["status"], environment,
        ])
    response = make_response(output.getvalue())
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = f'attachment; filename="products_{environment}_{period}.csv"'
    return response


# Draft product metadata lives in its own MongoDB collection, scoped to the
# authenticated user. It is never catalog-mapped or indexed until publish.
@app.route("/api/metadata/drafts", methods=["GET"])
@require_roles('metadata_editor', 'metadata_publisher', 'admin')
@limiter.limit("60 per minute")
def list_metadata_drafts():
    try:
        return jsonify({"drafts": metadata_manager.list_drafts(getattr(g, "current_user", None))})
    except Exception as exc:
        logger.error("Could not list metadata drafts: %s", exc, exc_info=True)
        return jsonify({"error": "Unable to load metadata drafts"}), 500


@app.route("/api/metadata/drafts", methods=["POST"])
@require_roles('metadata_editor', 'metadata_publisher', 'admin')
@limiter.limit("20 per minute")
def create_metadata_draft():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "A JSON request body is required"}), 400
    try:
        draft = metadata_manager.create_draft(
            payload.get("product_name"), payload.get("dataset_data"), getattr(g, "current_user", None)
        )
        return jsonify({"success": True, "draft": draft}), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("Could not create metadata draft: %s", exc, exc_info=True)
        return jsonify({"error": "Unable to create metadata draft"}), 500


@app.route("/api/metadata/drafts/<draft_id>", methods=["GET"])
@require_roles('metadata_editor', 'metadata_publisher', 'admin')
@limiter.limit("60 per minute")
def get_metadata_draft(draft_id):
    try:
        return jsonify({"draft": metadata_manager.get_draft(draft_id, getattr(g, "current_user", None))})
    except KeyError:
        return jsonify({"error": "Draft not found"}), 404
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("Could not load metadata draft %s: %s", draft_id, exc, exc_info=True)
        return jsonify({"error": "Unable to load metadata draft"}), 500


@app.route("/api/metadata/drafts/<draft_id>", methods=["PUT"])
@require_roles('metadata_editor', 'metadata_publisher', 'admin')
@limiter.limit("20 per minute")
def update_metadata_draft(draft_id):
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "A JSON request body is required"}), 400
    try:
        draft = metadata_manager.update_draft(
            draft_id=draft_id,
            product_name=payload.get("product_name"),
            dataset_data=payload.get("dataset_data"),
            expected_version=payload.get("expected_version"),
            actor=getattr(g, "current_user", None),
        )
        return jsonify({"success": True, "draft": draft})
    except MetadataConflictError as exc:
        return jsonify({"error": str(exc), "reload_required": True}), 409
    except KeyError:
        return jsonify({"error": "Draft not found"}), 404
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("Could not update metadata draft %s: %s", draft_id, exc, exc_info=True)
        return jsonify({"error": "Unable to save metadata draft"}), 500


@app.route("/api/metadata/drafts/<draft_id>", methods=["DELETE"])
@require_roles('metadata_editor', 'metadata_publisher', 'admin')
@limiter.limit("20 per minute")
def delete_metadata_draft(draft_id):
    try:
        metadata_manager.delete_draft(draft_id, getattr(g, "current_user", None))
        return jsonify({"success": True})
    except KeyError:
        return jsonify({"error": "Draft not found"}), 404
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("Could not delete metadata draft %s: %s", draft_id, exc, exc_info=True)
        return jsonify({"error": "Unable to delete metadata draft"}), 500


@app.route("/api/metadata/drafts/<draft_id>/publish", methods=["POST"])
@require_roles('metadata_publisher', 'admin')
@limiter.limit("10 per minute")
def publish_metadata_draft(draft_id):
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "A JSON request body is required"}), 400
    actor = getattr(g, "current_user", None)
    user_ip = request.remote_addr
    try:
        draft = metadata_manager.get_draft(draft_id, actor)
        product_name = draft["product_name"]
        job = _start_metadata_job(
            "publish",
            product_name,
            actor,
            lambda: metadata_manager.publish_draft(
                draft_id=draft_id,
                expected_version=payload.get("expected_version"),
                user_ip=user_ip,
                actor=actor,
            ),
        )
        return jsonify({"success": True, "job": job, "job_id": job["job_id"]}), 202
    except MetadataConflictError as exc:
        return jsonify({"error": str(exc), "reload_required": True}), 409
    except KeyError:
        return jsonify({"error": "Draft not found"}), 404
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    except MetadataJobUnavailableError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("Could not queue metadata draft publish %s: %s", draft_id, exc, exc_info=True)
        return jsonify({"error": "Unable to start product publish"}), 500

@app.route("/")
@limiter.limit("30 per minute")
def home():
    """Home page - search UI render karo."""
    return render_template("index.html")

def _predict_impl(interaction_engine, qdrant_search_client, qdrant_collection):
    """Main API: query receive karo, LLM rewrite, semantic search, filter selection, results + filters return. Top 3 datasets."""
    try:
        # Input validation
        raw_q = request.json.get("query", "").strip()
        if not raw_q:
            return jsonify({"error": "query required"}), 400
        
        # Basic input sanitization
        raw_q = raw_q[:500]  # Limit to 500 characters
        
        if len(raw_q) < 2:
            return jsonify({"error": "Query too short. Minimum 2 characters required"}), 400

        #  LLM rewrite
        q = rewrite_query_with_llm(raw_q)

        # 1. Fallback Expansions for all 22 products
        dataset_expansions = {
            r'\bplfs\b': "Periodic Labour Force Survey",
            r'\basuse\b': "Annual Survey of Unincorporated Sector Enterprises",
            r'\basi\b': "Annual Survey of Industries",
            r'\btus\b': "Time Use Survey",
            r'\bgender\b': "Gender Statistics",
            r'\baishe\b': "All India Survey on Higher Education",
            r'\bnss77\b': "NSS 77th Round AIDIS",
            r'\bnss78\b': "NSS 78th Round Domestic Tourism",
            r'\besi\b': "Energy Statistics India",
            r'\bcpialrl\b': "Consumer Price Index for Agricultural and Rural Labourers",
            r'\bhces\b': "Household Consumption Expenditure Survey",
            r'\benvstat\b': "Environment Statistics India",
            r'\bnfhs\b': "National Family Health Survey",
            r'\bec4\b': "4th Economic Census",
            r'\bec5\b': "5th Economic Census",
            r'\bec6\b': "6th Economic Census",
            r'\biip\b': "Index of Industrial Production",
            r'\bwpi\b': "Wholesale Price Index",
            r'\bcpi\b': "Consumer Price Index",
            r'\bnas\b': "National Accounts Statistics",
            r'\brbi\b': "Reserve Bank of India Banking Statistics",
            r'\bnss79c?\b': "Comprehensive Annual Modular Survey CAMS",
            r'\budise\b': "Unified District Information System for Education Plus"
        }
        for pat, exp in dataset_expansions.items():
            if re.search(pat, q.lower()) and exp.lower() not in q.lower():
                q = f"{q} {exp}"

        logger.info(f"Query - RAW: {raw_q}, LLM: {q}")

        top_results = search_indicators(
            q,
            raw_query=raw_q,
            qdrant_search_client=qdrant_search_client,
            qdrant_collection=qdrant_collection,
        )

        # 2. Force-Inclusion Logic (Isolation) - STRIcT CONTEXT
        _force_ds_map = {
            r'\bplfs\b|unemployment rate|labour force|lfpr|wpr|\bworker population ratio\b': ["PLFS"],
            r'\basuse\b|unincorporated|unorganized': ["ASUSE"],
            r'\basi\b|annual survey of industries|factory output|fixed capital|gross output|workers in factory': ["ASI"],
            r'\btus\b|time use survey|unpaid caregiving|domestic services': ["TUS"],
            r'\bgender\b|sex ratio': ["Gender"],
            r'\baishe\b|higher education|college|university': ["AISHE"],
            r'\bnss77\b|debt|investment|land|livestock': ["NSS77"],
            r'\bnss78\b|tourism': ["NSS78"],
            r'\besi\b|energy statistics|electricity|power supply': ["ESI"],
            r'\bcpialrl\b|agricultural labo|rural labo': ["CPIALRL"],
            r'\bhces\b|consumption expenditure|mpce': ["HCES"],
            r'\benvstat\b|environment statistics|forest cover|hazardous waste': ["ENVSTAT"],
            r'\bnfhs\b|family health|immunization|fertility|antenatal care|stunted|wasted|anemia': ["NFHS"],
            r'\bec4\b|4th economic census': ["EC4"],
            r'\bec5\b|5th economic census': ["EC5"],
            r'\bec6\b|6th economic census': ["EC6"],
            r'\biip\b|industrial production|mining index|manufacturing index|electricity index': ["IIP"],
            r'\bwpi\b|wholesale price': ["WPI"],
            r'\bcpi\b|consumer price|retail price|retail inflation': ["CPI", "CPI2"],
            r'\bnas\b|national accounts|gdp|gva': ["NAS"],
            r'\brbi\b|reserve bank|lending rate|exchange rate|forex|external debt|rupee vis-a-vis': ["RBI"],
            r'\bnss79c?\b|cams|modular survey': ["NSS79C", "NSS79"],
            r'\budise\b|school education|unified district': ["UDISE"]
        }
        
        _raw_lower = raw_q.lower().strip()
        _forced = None
        # Check for specific codes first
        for pat, codes in _force_ds_map.items():
            if re.search(pat, _raw_lower):
                _forced = codes
                break

        if _forced == ["UDISE"]:
            ds_best = _search_dataset_only(q or raw_q, _forced)
            if ds_best:
                top_results = [ds_best] + [r for r in top_results if r["product"] != ds_best["product"]][:2]
                ds_best["score"] = max((r.get("score", 0) for r in top_results), default=0) + 1

        if _forced and not any(r["product"] in _forced for r in top_results):
            ds_best = _search_dataset_only(q or raw_q, _forced)
            if ds_best:
                top_results = [ds_best] + [r for r in top_results if r["product"] != ds_best["product"]][:2]
                ds_best["score"] = max(r["score"] for r in top_results) + 1  # Boost to 1st

        _ds_priority = None
        for pat, codes in _force_ds_map.items():
            if re.search(pat, _raw_lower):
                _ds_priority = codes
                break
                
        if _ds_priority:
            for i, r in enumerate(top_results):
                if r["product"] in _ds_priority:
                    if i > 0:
                        top_results = [r] + [x for x in top_results if x["product"] != r["product"]][:2]
                    top_results[0]["score"] = max(x["score"] for x in top_results) + 1
                    break
        if _ds_priority:
            for i, r in enumerate(top_results):
                if r["product"] in _ds_priority:
                    if i > 0:
                        top_results = [r] + [x for x in top_results if x["product"] != r["product"]][:2]
                        r = top_results[0]
                    # Boost 95% confidence (whether moved or already 1st)
                    all_scores = [x["score"] for x in top_results]
                    top_results[0]["score"] = max(all_scores) + 1
                    break

        confidences = normalize_confidence([r["score"] for r in top_results])

        results = []

        for ind, conf in zip(top_results, confidences):
            related_filters = universal_filter_normalizer(
                ind["product"],
                ind.get("filters", []),
            )

            grouped = {}
            for f in related_filters:
                grouped.setdefault(f["filter_name"], []).append(f)

            best_filters = []
            for fname, opts in grouped.items():
                best_opt = select_best_filter_option(
                    query=q,
                    filter_name=fname,
                    options=opts,
                    cross_encoder=cross_encoder
                )
                best_filters.append({
                    "filter_name": fname,
                    "option": best_opt["option"]
                })
            best_filters = ensure_required_filters_present(best_filters, ind["product"], grouped, q, cross_encoder)
            if ind["product"] == "IIP":
                best_filters = apply_iip_filter_hierarchy(
                    best_filters,
                    ind.get("filters", []),
                    q,
                )

            results.append({
                "dataset": ind["product"],
                "product": ind["product"].lower(),  # ec4, ec5, ec6 - for URL (macroindicators?product=ec4)
                "indicator": ind["name"],
                "confidence": conf,
                "filters": best_filters
            })
        response = {"results": results}
        if not getattr(EVALUATION_CONTEXT, "active", False):
            save_query_log(
                raw_query=raw_q,
                rewritten_query=q,
                response_json=response,
                interaction_engine=interaction_engine,
            )

        return jsonify({"results": results})
    
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return jsonify({"error": "Invalid request", "message": str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/search/predict", methods=["POST"])
@require_login
@limiter.limit("20 per minute")
def predict():
    """Authenticated semantic-search endpoint available to every active user."""
    return _predict_impl(analytics_engines["dev"], qclient, COLLECTION)


@app.route("/api/v1/search/predict", methods=["POST"])
@limiter.limit("20 per minute")
@require_search_api_key
def api_v1_predict():
    """Static API-key endpoint using the exact browser search implementation.

    Rate limiting runs before key verification to slow invalid-key attacks.
    External clients authenticate each request through ``X-API-Key`` and do
    not need a browser session cookie or CSRF token.
    """
    return _predict_impl(analytics_engines["prod"], production_qclient, PRODUCTION_COLLECTION)


def _predict_for_evaluation(prompt):
    """Run the exact normal prediction handler for one evaluation prompt.

    The Flask request context is intentionally local to the background worker.
    Calling the undecorated handler avoids consuming the browser 20/minute
    rate-limit bucket while preserving all LLM, Qdrant/FAISS, reranking, and
    filter-selection behavior used by ``/search/predict``.
    """
    EVALUATION_CONTEXT.active = True
    try:
        with app.test_request_context(
            "/search/predict",
            method="POST",
            json={"query": prompt},
        ):
          
            response = _predict_impl(analytics_engines["dev"], qclient, COLLECTION)
            status_code = 200
            if isinstance(response, tuple):
                response, status_code = response[0], response[1]
            payload = response.get_json()
            if not 200 <= int(status_code) < 300:
                detail = None
                if isinstance(payload, dict):
                    detail = payload.get("message") or payload.get("error")
                raise RuntimeError(detail or f"Prediction API returned HTTP {status_code}")
            if not isinstance(payload, dict):
                raise RuntimeError("Prediction API returned invalid JSON")
            return payload
    finally:
        # Do not leave the flag set on a worker thread after one failed row.
        EVALUATION_CONTEXT.active = False


@app.route("/api/evaluation/jobs", methods=["POST"])
@require_roles('analyst', 'admin')
@limiter.limit("5 per minute")
def start_evaluation_job():
    """Upload a CSV/XLSX sheet and start a sequential background evaluation."""
    uploaded_file = request.files.get("file")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "Upload a CSV or XLSX evaluation file in the file field"}), 400
    try:
        job = evaluation_runner.start(
            filename=uploaded_file.filename,
            content=uploaded_file.read(),
            prediction_fn=_predict_for_evaluation,
            actor=g.current_user,
        )
        logger.info(
            "Evaluation job started: job_id=%s, file=%s, rows=%s",
            job["job_id"],
            job["file_name"],
            job["total_rows"],
        )
        return jsonify(job), 202
    except EvaluationValidationError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("Could not start evaluation job: %s", exc, exc_info=True)
        return jsonify({"error": "Could not start evaluation"}), 500


@app.route("/api/evaluation/jobs/<job_id>/stop", methods=["POST"])
@require_roles('analyst', 'admin')
@limiter.limit("10 per minute")
def stop_evaluation_job(job_id):
    """Request a safe stop; the active prompt may finish before the job stops."""
    try:
        job = evaluation_runner.request_cancel(job_id)
        logger.info("Evaluation stop requested: job_id=%s, user=%s", job_id, g.current_user.get("username"))
        return jsonify(job), 202
    except EvaluationValidationError as exc:
        return jsonify({"error": str(exc)}), 409
    except KeyError:
        return jsonify({"error": "Evaluation job not found"}), 404


@app.route("/api/evaluation/jobs/<job_id>", methods=["GET"])
@require_roles('analyst', 'admin')
@limiter.limit("120 per minute")
def get_evaluation_job(job_id):
    """Return one evaluation job summary plus one paginated page of rows."""
    try:
        page = int(request.args.get("page", "1"))
        page_size = int(request.args.get("page_size", "10"))
        return jsonify(evaluation_runner.get_page(job_id, page=page, page_size=page_size))
    except ValueError:
        return jsonify({"error": "page and page_size must be whole numbers"}), 400
    except EvaluationValidationError as exc:
        return jsonify({"error": str(exc)}), 400
    except KeyError:
        return jsonify({"error": "Evaluation job not found"}), 404


@app.route("/api/evaluation/jobs/<job_id>/download", methods=["GET"])
@require_roles('analyst', 'admin')
@limiter.limit("20 per minute")
def download_evaluation_job(job_id):
    """Download the complete completed evaluation as a CSV spreadsheet."""
    try:
        path = evaluation_runner.get_download_path(job_id)
        return send_file(
            path,
            mimetype="text/csv; charset=utf-8",
            as_attachment=True,
            download_name=path.name,
        )
    except EvaluationValidationError as exc:
        return jsonify({"error": str(exc)}), 409
    except KeyError:
        return jsonify({"error": "Evaluation job not found"}), 404


@app.route("/api/evaluation/history", methods=["GET"])
@require_roles('analyst', 'admin')
@limiter.limit("60 per minute")
def list_evaluation_history():
    """List the one latest saved evaluation available for each product."""
    try:
        page = int(request.args.get("page", "1"))
        page_size = int(request.args.get("page_size", "10"))
        search = request.args.get("search", "")
        return jsonify(
            evaluation_history_manager.list_latest_page(
                page=page,
                page_size=page_size,
                search=search,
            )
        )
    except ValueError:
        return jsonify({"error": "page and page_size must be whole numbers; page_size must be between 1 and 100"}), 400
    except EvaluationHistoryUnavailableError as exc:
        return jsonify({"error": str(exc), "source": "mongodb"}), 503


@app.route("/api/evaluation/history/<product_key>/download", methods=["GET"])
@require_roles('analyst', 'admin')
@limiter.limit("20 per minute")
def download_evaluation_history(product_key):
    """Download the latest saved evaluation sheet for one expected product."""
    try:
        content, filename = evaluation_history_manager.latest_csv(product_key)
        return send_file(
            io.BytesIO(content),
            mimetype="text/csv; charset=utf-8",
            as_attachment=True,
            download_name=filename,
        )
    except KeyError:
        return jsonify({"error": "No saved evaluation history for this product"}), 404
    except ValueError:
        return jsonify({"error": "Invalid product key"}), 400
    except EvaluationHistoryUnavailableError as exc:
        return jsonify({"error": str(exc), "source": "mongodb"}), 503


@app.route("/health", methods=["GET"])
@limiter.limit("60 per minute")
def health_check():
    """Expose whether Qdrant primary or FAISS fallback serves searches."""
    global USE_QDRANT
    fallback_ready = faiss_index is not None
    timestamp = datetime.utcnow().isoformat()

    if qclient is not None:
        try:
            collection_info = qclient.get_collection(COLLECTION)
            return jsonify({
                "status": "healthy" if USE_QDRANT else "degraded",
                "semantic_search_backend": "qdrant" if USE_QDRANT else "faiss_fallback",
                "llm_status": "running" if LLM_IS_RUNNING else "unavailable",
                "qdrant_status": "ready" if USE_QDRANT else "standby",
                "qdrant_collection": COLLECTION,
                "qdrant_vectors": getattr(collection_info, "vectors_count", None),
                "timestamp": timestamp,
            }), 200
        except Exception as exc:
            if USE_QDRANT:
                logger.warning(
                    "Health check detected Qdrant failure; FAISS FALLBACK ACTIVE: %s",
                    exc,
                    exc_info=True,
                )
            USE_QDRANT = False

    if fallback_ready:
        return jsonify({
            "status": "degraded",
            "semantic_search_backend": "faiss_fallback",
            "qdrant_status": "unavailable",
            "qdrant_collection": COLLECTION,
            "timestamp": timestamp,
        }), 200

    logger.error("Health check failed: Qdrant and FAISS semantic indexes are unavailable")
    return jsonify({
        "status": "unhealthy",
        "semantic_search_backend": "unavailable",
        "qdrant_status": "unavailable",
        "qdrant_collection": COLLECTION,
        "timestamp": timestamp,
    }), 503


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    
    logger.warning("=" * 60)
    logger.warning("WARNING: Running Flask development server")
    logger.warning("For production, use: gunicorn -w 4 -b 0.0.0.0:5000 app:app")
    logger.warning("=" * 60)
    
    app.run(
        debug=False,  
        host="0.0.0.0",
        port=port,
        threaded=True
    )
