"""Churn risk desk — Streamlit front end for the churn prediction API (notebook §11).

Run:  streamlit run churn_app.py        (the API must be running; API_URL defaults to http://127.0.0.1:8000)
"""
import json
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from pydantic import ValidationError
from sklearn.metrics import roc_auc_score

from churn_api import Customer          # the API's own schema: identical validation in the UI and the service
from churn_features import PREDICTORS
import whatif as W

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_DIR = os.getenv("MODEL_DIR")       # Docker: tables come from the exported model folder
MAX_ROWS, API_BATCH = 10_000, 1_000
ORANGE, TEAL, BLUE, INK, MUTED = "#EB6834", "#1BAF7A", "#2A78D6", "#14213D", "#5B6575"

st.set_page_config(page_title="Churn risk desk", page_icon="📶", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, .stApp *:not([data-testid="stIconMaterial"]):not(.material-symbols-rounded):not(code) { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { font-variant-numeric: tabular-nums; }
h1 { font-weight: 600; letter-spacing: -0.02em; font-size: 2.1rem !important; margin-bottom: 0 !important; }
h2, h3 { font-weight: 600; letter-spacing: -0.01em; }
.lede { color: #5B6575; font-size: 1.02rem; max-width: 64ch; margin: .2rem 0 1.2rem; }
.readout { display: flex; align-items: baseline; gap: .8rem; margin: .2rem 0 .1rem; }
.readout .pct { font-size: 4.2rem; font-weight: 600; line-height: 1; letter-spacing: -0.04em;
                font-variant-numeric: tabular-nums; }
.readout .what { color: #5B6575; font-size: 1rem; max-width: 18ch; line-height: 1.3; }
.rail { position: relative; margin: 1.1rem 0 .35rem; }
.rail-track { position: relative; height: 14px; border-radius: 7px; background: #E3E8EF; overflow: visible; }
.rail-zone { position: absolute; top: 0; bottom: 0; right: 0; border-radius: 0 7px 7px 0;
             background: repeating-linear-gradient(135deg, rgba(235,104,52,.30) 0 6px, rgba(235,104,52,.16) 6px 12px); }
.rail-thr { position: absolute; top: -7px; width: 2px; height: 28px; background: #14213D; transform: translateX(-1px); }
.rail-dot { position: absolute; top: 50%; width: 22px; height: 22px; border-radius: 50%; border: 3px solid #fff;
            box-shadow: 0 0 0 1.5px #14213D; transform: translate(-50%, -50%);
            transition: left .35s ease; }
.rail-ghost { position: absolute; top: 50%; width: 18px; height: 18px; border-radius: 50%; background: #F7F9FB;
              border: 3px solid #14213D; transform: translate(-50%, -50%); transition: left .35s ease; }
.rail-move { position: absolute; top: 50%; height: 3px; background: #14213D; transform: translateY(-50%);
             transition: left .35s ease, width .35s ease; }
@media (prefers-reduced-motion: reduce) { .rail-dot, .rail-ghost, .rail-move { transition: none; } }
.rail-scale { display: flex; justify-content: space-between; color: #5B6575; font-size: .8rem;
              font-variant-numeric: tabular-nums; margin-top: .45rem; }
.verdict { margin: .9rem 0 0; padding: .7rem .9rem; border-left: 3px solid; background: #fff; border-radius: 0 6px 6px 0; }
.small { color: #5B6575; font-size: .86rem; }
</style>
""", unsafe_allow_html=True)   # static CSS only — no user input is ever rendered as HTML


# ------------------------------------------------------------------------------------------- data helpers
PRESETS = {
    "Two months in, fibre, month-to-month": dict(
        tenure=2, MonthlyCharges=70.70, TotalCharges=151.65, gender="Female", SeniorCitizen=0, Partner="No",
        Dependents="No", PhoneService="Yes", MultipleLines="No", InternetService="Fiber optic", OnlineSecurity="No",
        OnlineBackup="No", DeviceProtection="No", TechSupport="No", StreamingTV="No", StreamingMovies="No",
        Contract="Month-to-month", PaperlessBilling="Yes", PaymentMethod="Electronic check"),
    "Heavy streamer, fibre, 20 months": dict(
        tenure=20, MonthlyCharges=100.50, TotalCharges=2010.00, gender="Male", SeniorCitizen=0, Partner="Yes",
        Dependents="No", PhoneService="Yes", MultipleLines="Yes", InternetService="Fiber optic", OnlineSecurity="No",
        OnlineBackup="Yes", DeviceProtection="No", TechSupport="No", StreamingTV="Yes", StreamingMovies="Yes",
        Contract="Month-to-month", PaperlessBilling="Yes", PaymentMethod="Electronic check"),
    "Five years in, two-year contract, DSL": dict(
        tenure=60, MonthlyCharges=65.50, TotalCharges=3930.00, gender="Male", SeniorCitizen=0, Partner="Yes",
        Dependents="Yes", PhoneService="Yes", MultipleLines="Yes", InternetService="DSL", OnlineSecurity="Yes",
        OnlineBackup="Yes", DeviceProtection="Yes", TechSupport="Yes", StreamingTV="No", StreamingMovies="No",
        Contract="Two year", PaperlessBilling="No", PaymentMethod="Bank transfer (automatic)"),
    "New sign-up, phone only, not billed yet": dict(
        tenure=0, MonthlyCharges=20.00, TotalCharges=0.0, gender="Female", SeniorCitizen=0, Partner="No",
        Dependents="No", PhoneService="Yes", MultipleLines="No", InternetService="No",
        OnlineSecurity="No internet service", OnlineBackup="No internet service",
        DeviceProtection="No internet service", TechSupport="No internet service",
        StreamingTV="No internet service", StreamingMovies="No internet service",
        Contract="Month-to-month", PaperlessBilling="No", PaymentMethod="Mailed check"),
}
ADD_ONS = {"OnlineSecurity": "Online security", "OnlineBackup": "Online backup",
           "DeviceProtection": "Device protection", "TechSupport": "Tech support",
           "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies"}
FLAG_FEATURES = {"Partner": "partner", "Dependents": "dependents", "PhoneService": "phone service",
                 "PaperlessBilling": "paperless billing", "MultipleLines": "multiple lines",
                 **{k: v.lower() for k, v in ADD_ONS.items()}}


def factor_text(feature: str, c: dict) -> str:
    """Turn a model feature into a sentence fragment about this customer."""
    if feature == "tenure":
        n = int(round(c["tenure"]))
        return "brand new (0 months)" if n == 0 else f"{n} month{'s' if n != 1 else ''} with us"
    if feature == "MonthlyCharges":
        return f"pays {c['MonthlyCharges']:.2f} a month"
    if feature == "TotalCharges":
        return f"{c['TotalCharges']:.0f} billed in total"
    if feature == "contract_months":
        return {"Month-to-month": "month-to-month contract", "One year": "one-year contract",
                "Two year": "two-year contract"}[c["Contract"]]
    if feature == "SeniorCitizen":
        return "senior citizen" if c["SeniorCitizen"] else "not a senior citizen"
    if feature == "gender_male":
        return "male" if c["gender"] == "Male" else "female"
    if feature.startswith("InternetService_"):
        kind = feature.split("_", 1)[1]
        label = {"Fiber optic": "fibre internet", "No": "no internet service"}[kind]
        return label if c["InternetService"] == kind else f"not on {label}" if kind != "No" else "has internet"
    if feature.startswith("PaymentMethod_"):
        kind = feature.split("_", 1)[1]
        return f"pays by {kind.lower()}" if c["PaymentMethod"] == kind else f"doesn't pay by {kind.lower()}"
    if feature in FLAG_FEATURES:
        return f"has {FLAG_FEATURES[feature]}" if c[feature] == "Yes" else f"no {FLAG_FEATURES[feature]}"
    return feature


def api(method: str, path: str, **kw):
    """Call the API; return (json, error message)."""
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=30, **kw)
    except requests.RequestException:
        return None, ("The prediction service isn't reachable. Start it in §11 of the notebook "
                      f"(it should answer at {API_URL}/health), then reload this page.")
    if r.status_code == 422:
        detail = r.json().get("detail", [])
        msgs = [d.get("msg", str(d)) for d in detail] if isinstance(detail, list) else [str(detail)]
        return None, "The service rejected this input: " + "; ".join(m.replace("Value error, ", "") for m in msgs)
    if r.status_code != 200:
        return None, f"The service answered with HTTP {r.status_code}. Check results/api.log in the notebook."
    return r.json(), None


@st.cache_data(ttl=60, show_spinner=False)
def model_info():
    return api("GET", "/model")


@st.cache_data(ttl=300, show_spinner=False)
def mlflow_tables(run_id: str):
    """Leaderboard + threshold table logged in §10 (read-only)."""
    if MODEL_DIR:
        try:
            info = json.loads((Path(MODEL_DIR) / "model_info.json").read_text())
            return pd.DataFrame(info["leaderboard"]), pd.DataFrame(info["threshold_table"])
        except (OSError, KeyError, ValueError):
            return None, None
    if not TRACKING_URI:
        return None, None
    try:
        import mlflow
        mlflow.set_tracking_uri(TRACKING_URI)
        as_df = lambda d: pd.DataFrame(d["data"], columns=d["columns"])
        thr = as_df(mlflow.artifacts.load_dict(f"runs:/{run_id}/tables/threshold_tradeoff_test.json"))
        parent = mlflow.get_run(run_id).data.tags.get("mlflow.parentRunId")
        board = as_df(mlflow.artifacts.load_dict(f"runs:/{parent}/leaderboard.json")) if parent else None
        return board, thr
    except Exception:
        return None, None


def safe_cell(v):
    """Neutralise spreadsheet formulas in values echoed back from an uploaded file (CSV injection)."""
    return "'" + v if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r") else v


# ------------------------------------------------------------------------------------------- sidebar
info, info_err = model_info()
with st.sidebar:
    st.markdown("### Decision threshold")
    default_thr = int(round(100 * (info or {}).get("threshold", 0.5)))
    thr_pct = st.slider("Flag customers at or above this risk", 5, 95, default_thr, 5, format="%d%%", key="thr")
    st.markdown('<p class="small">Lower it to catch more leavers, at the cost of more retention calls to people '
                'who would have stayed. The report uses 50%.</p>', unsafe_allow_html=True)
    st.divider()
    if info:
        st.markdown(f"**Model version {info['version']}**")
        st.caption(f"Logistic regression, business encoding. Test ROC-AUC {info['test']['test_roc_auc']:.3f}.")
    else:
        st.error(info_err)
threshold = thr_pct / 100

st.title("Churn risk desk")
st.markdown('<p class="lede">Score a customer before a retention call, or a whole list at once. Risks come from '
            'the champion model registered in MLflow; reasons are associations in past data, not causes.</p>',
            unsafe_allow_html=True)

tab_one, tab_list, tab_model = st.tabs(["Score a customer", "Score a list", "About the model"])

# ------------------------------------------------------------------------------------------- tab 1
with tab_one:
    def apply_preset():
        for k, v in PRESETS[st.session_state.preset].items():
            st.session_state[k] = v

    if "tenure" not in st.session_state:                       # first visit: start from an example
        st.session_state.preset = next(iter(PRESETS))
        apply_preset()

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.selectbox("Start from an example", list(PRESETS), key="preset", on_change=apply_preset)
        st.markdown("#### Account")
        a1, a2, a3 = st.columns(3)
        a1.number_input("Months with us", 0, 120, step=1, key="tenure")
        a2.number_input("Monthly bill", 0.0, 1000.0, step=5.0, format="%.2f", key="MonthlyCharges")
        a3.number_input("Billed in total", 0.0, 200000.0, step=50.0, format="%.2f", key="TotalCharges")
        b1, b2, b3, b4 = st.columns(4)
        b1.selectbox("Gender", ["Female", "Male"], key="gender")
        b2.selectbox("Senior citizen", [0, 1], format_func=lambda v: "Yes" if v else "No", key="SeniorCitizen")
        b3.selectbox("Partner", ["No", "Yes"], key="Partner")
        b4.selectbox("Dependents", ["No", "Yes"], key="Dependents")

        st.markdown("#### Contract and billing")
        c1, c2, c3 = st.columns([1.3, 1.4, 1.0])
        c1.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="Contract")
        c2.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                                        "Credit card (automatic)"], key="PaymentMethod")
        c3.selectbox("Paperless billing", ["No", "Yes"], key="PaperlessBilling")

        st.markdown("#### Services")
        s1, s2, s3 = st.columns(3)
        s1.selectbox("Phone line", ["No", "Yes"], key="PhoneService")
        no_phone = st.session_state.PhoneService == "No"
        if no_phone:
            st.session_state.MultipleLines = "No phone service"
        elif st.session_state.get("MultipleLines") == "No phone service":
            st.session_state.MultipleLines = "No"
        s2.selectbox("Multiple lines", ["No phone service"] if no_phone else ["No", "Yes"],
                     key="MultipleLines", disabled=no_phone)
        s3.selectbox("Internet", ["DSL", "Fiber optic", "No"],
                     format_func=lambda v: {"Fiber optic": "Fibre", "No": "None"}.get(v, v), key="InternetService")
        no_net = st.session_state.InternetService == "No"
        cols = st.columns(3)
        for i, (k, label) in enumerate(ADD_ONS.items()):
            if no_net:
                st.session_state[k] = "No internet service"
            elif st.session_state.get(k) == "No internet service":
                st.session_state[k] = "No"
            cols[i % 3].selectbox(label, ["No internet service"] if no_net else ["No", "Yes"], key=k,
                                  disabled=no_net)

    customer = {k: st.session_state[k] for k in PREDICTORS}

    # ---- everything for this tab is scored in ONE batch call: the customer, the offer, every single change,
    #      and both tenure paths (now / with the offer)
    st.session_state.setdefault("adjust", True)
    st.session_state.setdefault("offer", [])
    adjust = st.session_state.adjust
    singles = W.single_changes(customer, adjust)
    valid_ids = [ch["id"] for ch, _ in singles]
    st.session_state.offer = [i for i in st.session_state.offer if i in valid_ids]   # drop stale choices
    offer_ids = st.session_state.offer
    offer_customer = W.apply(customer, offer_ids, adjust) if offer_ids else None
    path_now = W.tenure_path(customer)
    path_offer = W.tenure_path(offer_customer) if offer_customer else []
    rows = [customer] + ([offer_customer] if offer_customer else []) + [x for _, x in singles] + path_now + path_offer
    batch, err = api("POST", "/predict/batch", json={"customers": rows, "threshold": threshold})
    if not err:
        preds = batch["predictions"]
        res, preds = preds[0], preds[1:]
        offer_res = preds.pop(0) if offer_customer else None
        single_res, preds = preds[:len(singles)], preds[len(singles):]
        path_now_p, path_offer_p = preds[:len(path_now)], preds[len(path_now):]

    with right:
        if err:
            st.error(err)
        else:
            p = res["churn_probability"]
            colour = ORANGE if res["churn"] else TEAL
            ghost = ""
            if offer_res:
                q = offer_res["churn_probability"]
                lo, hi = sorted((p, q))
                ghost = (f'<div class="rail-move" style="left:{100 * lo:.1f}%; width:{100 * (hi - lo):.1f}%"></div>'
                         f'<div class="rail-ghost" style="left:{100 * q:.1f}%"></div>')
            st.markdown(
                f'<div class="readout"><span class="pct" style="color:{colour}">{p:.0%}</span>'
                f'<span class="what">chance this customer leaves</span></div>'
                f'<div class="rail"><div class="rail-track">'
                f'<div class="rail-zone" style="left:{100 * threshold:.1f}%"></div>'
                f'<div class="rail-thr" style="left:{100 * threshold:.1f}%"></div>{ghost}'
                f'<div class="rail-dot" style="left:{100 * p:.1f}%; background:{colour}"></div>'
                f'</div><div class="rail-scale"><span>0%</span><span>threshold {threshold:.0%}</span>'
                f'<span>100%</span></div></div>', unsafe_allow_html=True)          # numbers only, formatted here
            if offer_res:
                q = offer_res["churn_probability"]
                pts = f"{100 * (q - p):+.0f}".replace("-", "−")
                verdict = (f"With your offer: <b>{q:.0%}</b> ({pts} points), monthly bill "
                           f"{offer_customer['MonthlyCharges']:.2f} instead of {customer['MonthlyCharges']:.2f}. "
                           + ("Still at or above the threshold." if offer_res["churn"] else "Now below the threshold."))
                v_colour = ORANGE if offer_res["churn"] else TEAL
            else:
                verdict = ("At or above your threshold: flag for a retention call." if res["churn"] else
                           "Below your threshold: no action needed.")
                v_colour = colour
            st.markdown(f'<p class="verdict" style="border-color:{v_colour}">{verdict}</p>', unsafe_allow_html=True)

            f = pd.DataFrame(res["top_factors"] or [])
            if not f.empty:
                f["reason"] = [factor_text(x, customer) for x in f.feature]
                up = [r for r, c in zip(f.reason, f.contribution) if c > 0][:2]
                down = [r for r, c in zip(f.reason, f.contribution) if c < 0][:2]
                st.markdown("#### Why")
                parts = []
                if up:
                    parts.append("Risk goes up because of " + " and ".join(up) + ".")
                if down:
                    parts.append("It comes down because of " + " and ".join(down) + ".")
                st.write(" ".join(parts))
                f["direction"] = np.where(f.contribution > 0, "raises risk", "lowers risk")
                chart = (alt.Chart(f).mark_bar(cornerRadiusEnd=3, height=18).encode(
                    x=alt.X("contribution:Q", title="effect on churn odds (log-odds)",
                            axis=alt.Axis(grid=True, gridColor="#E3E8EF")),
                    y=alt.Y("reason:N", sort=alt.SortField("contribution", order="descending"), title=None,
                            axis=alt.Axis(labelLimit=260, labelFontSize=12)),
                    color=alt.Color("direction:N", scale=alt.Scale(domain=["raises risk", "lowers risk"],
                                                                   range=[ORANGE, TEAL]), legend=None),
                    tooltip=["reason", alt.Tooltip("contribution:Q", format="+.2f")])
                    .properties(height=200, padding={"left": 12, "right": 8}).configure(font="IBM Plex Sans")
                    .configure_view(strokeWidth=0)
                    .configure_axis(labelColor=MUTED, titleColor=MUTED, domainColor="#C9D1DC"))
                st.altair_chart(chart, width="stretch")
                st.caption("Each bar is the model coefficient times this customer's standardised value. "
                           "These are patterns in past customers, not proof of cause.")

    # ---- what-if simulator ---------------------------------------------------------------------------
    if not err:
        st.divider()
        st.markdown("### What would change this risk?")
        st.markdown('<p class="lede">Each bar compares this customer with one who is identical except for a '
                    'single change. Pick changes below to build a retention offer; the rail above shows where '
                    'it would land.</p>', unsafe_allow_html=True)
        st.toggle("Change the bill with the services", key="adjust", help=(
            "On: adding or dropping a service also changes the monthly bill by its list price (fibre 50, DSL 25, "
            "phone 20, streaming 10, other add-ons 5 — the data fits this price list almost exactly), so the "
            "comparison is with a realistic customer. Off: only the service changes and the bill stays put. No "
            "customer in the data looks like that, which is why the two answers can differ a lot."))

        wf = pd.DataFrame([{"id": ch["id"], "change": ch["label"],
                            "risk": r["churn_probability"], "points": 100 * (r["churn_probability"] - p),
                            "bill": x["MonthlyCharges"], "bill change": x["MonthlyCharges"] - customer["MonthlyCharges"],
                            "in offer": ch["id"] in offer_ids}
                           for (ch, x), r in zip(singles, single_res)]).sort_values("points")
        wf["effect"] = np.where(wf.points < 0, "lowers risk", "raises risk")
        wf["label"] = [f"{v:+.0f}".replace("-", "−") for v in wf.points]
        w1, w2 = st.columns([1.35, 1], gap="large")
        with w1:
            order = list(wf.change)                     # already sorted: biggest drop at the top
            y = alt.Y("change:N", sort=order, title=None,
                      axis=alt.Axis(labelLimit=320, labelFontSize=12, labelOverlap=False, ticks=False))
            tip = [alt.Tooltip("change:N"), alt.Tooltip("risk:Q", format=".0%", title="risk after"),
                   alt.Tooltip("points:Q", format="+.1f", title="change (points)"),
                   alt.Tooltip("bill:Q", format=".2f", title="monthly bill after")]
            bars = alt.Chart(wf).mark_bar(cornerRadiusEnd=3, height=15).encode(
                y=y, x=alt.X("points:Q", title="change in churn risk (percentage points)",
                             axis=alt.Axis(grid=True, gridColor="#E3E8EF")),
                color=alt.Color("effect:N", scale=alt.Scale(domain=["lowers risk", "raises risk"],
                                                            range=[TEAL, ORANGE]), legend=None),
                opacity=alt.condition(alt.datum["in offer"], alt.value(1.0), alt.value(0.5)), tooltip=tip)
            neg = alt.Chart(wf[wf.points < 0]).mark_text(align="right", dx=-5, fontSize=11, color=MUTED).encode(
                y=y, x="points:Q", text="label:N")
            pos = alt.Chart(wf[wf.points >= 0]).mark_text(align="left", dx=5, fontSize=11, color=MUTED).encode(
                y=y, x="points:Q", text="label:N")
            zero = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color=INK, strokeWidth=1).encode(x="z:Q")
            span = max(5.0, wf.points.abs().max())
            chart = alt.layer(bars, neg, pos, zero).resolve_scale(y="shared").encode(
                x=alt.X("points:Q", scale=alt.Scale(domain=[-span * 1.15, max(span * 0.35, wf.points.max() * 1.25)])))
            st.altair_chart(chart.properties(height=26 * len(wf) + 40, padding={"left": 28})
                            .configure(font="IBM Plex Sans").configure_view(strokeWidth=0)
                            .configure_axis(labelColor=MUTED, titleColor=MUTED, domainColor="#C9D1DC"),
                            width="stretch")
        with w2:
            labels = dict(zip(wf.id, wf.change))

            def suggest():
                """Best risk-reducing changes a retention team can actually offer, one per kind."""
                ok = ("contract:", "pay:", "add:OnlineSecurity", "add:TechSupport", "add:OnlineBackup",
                      "add:DeviceProtection")
                picks, groups = [], set()
                for _, row in wf[wf.points < -0.5].iterrows():
                    grp = row.id.split(":")[0] + (":" + row.id.split(":")[1] if row.id.startswith("add:") else "")
                    if row.id.startswith(ok) and grp not in groups:
                        picks.append(row.id); groups.add(grp)
                    if len(picks) == 3:
                        break
                st.session_state.offer = picks

            st.multiselect("Retention offer", valid_ids, format_func=lambda i: labels.get(i, i), key="offer",
                           placeholder="Pick one or more changes")
            b1, b2 = st.columns(2)
            b1.button("Suggest an offer", on_click=suggest, width="stretch",
                      help="Picks up to three changes a retention team could offer (contract, automatic payment, "
                           "security or support add-ons) that lower this customer's risk the most.")
            b2.button("Clear", on_click=lambda: st.session_state.update(offer=[]), width="stretch")
            offerable = wf[wf.id.str.startswith(("contract:", "pay:", "add:", "paperless"))]
            if len(offerable) and offerable.points.iloc[0] < 0:
                best = offerable.iloc[0]
                st.markdown(f"Biggest drop you could offer: **{best.change.lower()}** ({best.label} points).")
            worst = wf.iloc[-1]
            if worst.points > 0:
                st.markdown(f"Biggest rise: **{worst.change.lower()}** ({worst.label} points).")
            st.caption("Some changes are not things you would do to keep a customer. Customers without internet "
                       "rarely leave in this data, for example, but cancelling someone's internet will not keep "
                       "them. The model shows patterns; whether an offer works needs a real test.")

        st.markdown("#### If they stay")
        tp = pd.DataFrame({"months": [x["tenure"] for x in path_now], "risk": [r["churn_probability"] for r in path_now_p],
                           "line": "as they are"})
        if path_offer:
            tp = pd.concat([tp, pd.DataFrame({"months": [x["tenure"] for x in path_offer],
                                              "risk": [r["churn_probability"] for r in path_offer_p],
                                              "line": "with your offer"})])
        now = pd.DataFrame({"months": [customer["tenure"]], "risk": [p]})
        tline = (alt.Chart(tp).mark_line(strokeWidth=2.5).encode(
            x=alt.X("months:Q", title="months with us", scale=alt.Scale(domain=[0, 72], nice=False)),
            y=alt.Y("risk:Q", title="churn risk", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("line:N", scale=alt.Scale(domain=["as they are", "with your offer"], range=[BLUE, TEAL]),
                            legend=alt.Legend(orient="top-right", title=None)),
            strokeDash=alt.StrokeDash("line:N", scale=alt.Scale(domain=["as they are", "with your offer"],
                                                                range=[[1, 0], [6, 4]]), legend=None),
            tooltip=[alt.Tooltip("months:Q"), alt.Tooltip("risk:Q", format=".0%"), "line:N"])
            + alt.Chart(now).mark_point(size=140, filled=True, color=INK).encode(x="months:Q", y="risk:Q")
            + alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(color=ORANGE, strokeDash=[4, 3]).encode(y="t:Q"))
        st.altair_chart(tline.properties(height=240).configure(font="IBM Plex Sans").configure_view(strokeWidth=0)
                        .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor="#E3E8EF"), width="stretch")
        st.caption("Same customer at every tenure, with the total billed growing at the current monthly bill. "
                   "The dot is today; the dashed orange line is your threshold.")

# ------------------------------------------------------------------------------------------- tab 2
with tab_list:
    st.markdown("Upload a CSV with one customer per row and the 19 columns of the IBM Telco file "
                "(`customerID` and `Churn` are optional). Up to 10,000 rows.")
    template = pd.DataFrame([{k: (v if v is not None else "") for k, v in p.items()} for p in PRESETS.values()])
    template.insert(0, "customerID", [f"EXAMPLE-{i + 1}" for i in range(len(template))])
    st.download_button("Download a template CSV", template.to_csv(index=False), "churn_template.csv", "text/csv")
    up_file = st.file_uploader("Customer list (CSV)", type=["csv"])

    if up_file is not None:
        try:
            raw = pd.read_csv(up_file, dtype=str, keep_default_na=False, nrows=MAX_ROWS + 1)
        except Exception:
            raw = None
            st.error("This file couldn't be read as a CSV. Save it as comma-separated text and upload it again.")
        if raw is not None:
            raw.columns = [c.strip() for c in raw.columns]
            missing = [c for c in PREDICTORS if c not in raw.columns]
            if missing:
                st.error("These columns are missing: " + ", ".join(missing) + ". Use the template as a guide.")
            elif len(raw) > MAX_ROWS:
                st.error(f"The file has more than {MAX_ROWS:,} rows. Split it into smaller files.")
            else:
                ids = raw["customerID"].str.strip() if "customerID" in raw else pd.Series(
                    [f"row {i + 2}" for i in range(len(raw))])
                good, good_ids, good_idx, bad = [], [], [], []
                for i, row in raw.iterrows():
                    d = {k: row[k].strip() for k in PREDICTORS}
                    try:
                        for k in ("tenure", "MonthlyCharges"):
                            d[k] = float(d[k])
                        d["TotalCharges"] = float(d["TotalCharges"]) if d["TotalCharges"] else None
                        d["SeniorCitizen"] = int(d["SeniorCitizen"])
                        good.append(Customer(**d).model_dump())
                        good_ids.append(ids.iloc[i])
                        good_idx.append(i)
                    except (ValueError, ValidationError) as e:
                        msg = e.errors()[0]["msg"] if isinstance(e, ValidationError) else "a number is not numeric"
                        bad.append({"customer": safe_cell(ids.iloc[i]), "problem": msg.replace("Value error, ", "")})

                preds, err = [], None
                for s in range(0, len(good), API_BATCH):
                    out, err = api("POST", "/predict/batch",
                                   json={"customers": good[s:s + API_BATCH], "threshold": threshold})
                    if err:
                        break
                    preds += out["predictions"]
                if err:
                    st.error(err)
                elif good:
                    res = pd.DataFrame({
                        "customer": [safe_cell(x) for x in good_ids],
                        "risk": [q["churn_probability"] for q in preds],
                        "flag": ["call" if q["churn"] else "" for q in preds],
                        "main reason": [factor_text(q["top_factors"][0]["feature"], c) if q["top_factors"] else ""
                                        for q, c in zip(preds, good)]}).sort_values("risk", ascending=False)
                    n_flag = (res.flag == "call").sum()
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Customers scored", f"{len(res):,}")
                    m2.metric(f"Flagged at {threshold:.0%}", f"{n_flag:,}  ({n_flag / len(res):.0%})")
                    m3.metric("Rows skipped", f"{len(bad):,}")

                    labels = raw.loc[good_idx, "Churn"].str.strip() if "Churn" in raw else None
                    if labels is not None and set(labels) <= {"Yes", "No"} and labels.nunique() == 2:
                        y = (labels == "Yes").astype(int).to_numpy()
                        prob = np.array([q["churn_probability"] for q in preds])
                        flagged = prob >= threshold
                        st.markdown(
                            f"This file has real outcomes, so here is how the model did on it: it ranked "
                            f"leavers above stayers {roc_auc_score(y, prob):.0%} of the time (ROC-AUC), caught "
                            f"**{flagged[y == 1].mean():.0%}** of the people who left, and **{y[flagged].mean():.0%}**"
                            f" of the flagged customers really left.")

                    hist = (alt.Chart(res).mark_bar(color=BLUE, opacity=.85).encode(
                        x=alt.X("risk:Q", bin=alt.Bin(step=0.05), title="churn risk",
                                axis=alt.Axis(format="%")), y=alt.Y("count()", title="customers"))
                        + alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(color=INK, strokeWidth=2)
                        .encode(x="t:Q")).properties(height=180).configure_view(strokeWidth=0) \
                        .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor="#E3E8EF")
                    st.altair_chart(hist, width="stretch")
                    st.dataframe(res, hide_index=True, width="stretch", column_config={
                        "risk": st.column_config.ProgressColumn("risk", min_value=0, max_value=1, format="percent")})
                    st.download_button("Download scored list", res.to_csv(index=False), "churn_scored.csv",
                                       "text/csv", type="primary")
                if bad:
                    with st.expander(f"{len(bad)} rows were skipped — see why"):
                        st.dataframe(pd.DataFrame(bad), hide_index=True, width="stretch")

# ------------------------------------------------------------------------------------------- tab 3
with tab_model:
    if not info:
        st.error(info_err)
    else:
        st.markdown(f"The app serves **{info['config']}**, version {info['version']} of "
                    f"`{info['name']}` in the MLflow registry (run `{info['run_id'][:8]}`).")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("CV ROC-AUC", f"{info['cv_roc_auc']:.3f} ± {info['cv_roc_auc_sd']:.3f}")
        k2.metric("Test ROC-AUC", f"{info['test']['test_roc_auc']:.3f}")
        k3.metric("Test recall at 50%", f"{info['test']['test_recall']:.2f}")
        k4.metric("Test precision at 50%", f"{info['test']['test_precision']:.2f}")
        board, thr_table = mlflow_tables(info["run_id"])
        st.markdown("#### Eight candidates, one rule")
        st.write("Every configuration from the report was scored by 5-fold cross-validation. The best Random "
                 "Forest leads by about 0.001 ROC-AUC, well inside one standard error, so the rule fixed in "
                 "advance treats it as a tie and keeps the simpler model you can explain to a customer.")
        if board is not None:
            st.dataframe(board, hide_index=True, width="stretch")
        if thr_table is not None:
            st.markdown("#### What the threshold costs")
            long = thr_table.melt("threshold", ["recall", "precision"], "measure", "value")
            long["measure"] = long.measure.map({"recall": "leavers caught (recall)",
                                                "precision": "flags that are real leavers (precision)"})
            line = (alt.Chart(long).mark_line(point=True, strokeWidth=2.5).encode(
                x=alt.X("threshold:Q", axis=alt.Axis(format="%"), title="threshold"),
                y=alt.Y("value:Q", axis=alt.Axis(format="%"), title=None, scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("measure:N", scale=alt.Scale(
                    domain=["leavers caught (recall)", "flags that are real leavers (precision)"], range=[BLUE, INK]),
                    legend=alt.Legend(orient="bottom", title=None, labelLimit=400)))
                + alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(color=INK, strokeDash=[4, 3]).encode(x="t:Q"))
            st.altair_chart(line.properties(height=260).configure_view(strokeWidth=0)
                            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor="#E3E8EF"),
                            width="stretch")
            st.caption("Measured on the 1,409 held-out test customers. The dashed line is your current threshold.")
        st.markdown("#### Before acting on a score")
        st.write("The data is one snapshot: for people who left, months with us and total billed were measured "
                 "at departure. Customers still with us look different, so check the model on new data before "
                 "relying on it, and treat the reasons as patterns rather than causes.")
