"""What-if scenarios for one customer: change a service, the contract, the payment method or the tenure.

Pure Python (no Streamlit, no model) so it can be unit-tested. Every scenario is a complete, valid customer
that the API scores like any other.

The monthly bill in the IBM data is almost exactly a price list (least-squares fit on the 7,043 customers,
R² = 0.999, mean error 0.78): base 0 + fibre 50 / DSL 25 + phone 20 + multiple lines 5 + each security,
backup, protection or support add-on 5 + each streaming add-on 10. With `adjust_bill=True` a scenario changes
the bill by that price, and the total billed by tenure × that price, so the scenario looks like a real
customer with that set-up. With `adjust_bill=False` only the service changes, which the data never shows
(services and bill always move together) — useful to see why the model's coefficients must be read jointly.
"""
from __future__ import annotations

ADD_ONS = {"OnlineSecurity": "online security", "OnlineBackup": "online backup",
           "DeviceProtection": "device protection", "TechSupport": "tech support",
           "StreamingTV": "streaming TV", "StreamingMovies": "streaming movies"}
PRICES = {"Fiber optic": 50.0, "DSL": 25.0, "PhoneService": 20.0, "MultipleLines": 5.0,
          "OnlineSecurity": 5.0, "OnlineBackup": 5.0, "DeviceProtection": 5.0, "TechSupport": 5.0,
          "StreamingTV": 10.0, "StreamingMovies": 10.0}
CONTRACTS = ["Month-to-month", "One year", "Two year"]
AUTOPAY = ["Bank transfer (automatic)", "Credit card (automatic)"]


def list_price(c: dict) -> float:
    """Monthly price of the customer's set-up according to the fitted price list."""
    p = PRICES.get(c["InternetService"], 0.0)
    p += PRICES["PhoneService"] if c["PhoneService"] == "Yes" else 0.0
    p += PRICES["MultipleLines"] if c["MultipleLines"] == "Yes" else 0.0
    return p + sum(PRICES[k] for k in ADD_ONS if c[k] == "Yes")


def _rebill(before: dict, after: dict, adjust_bill: bool) -> dict:
    if adjust_bill:
        delta = list_price(after) - list_price(before)
        after["MonthlyCharges"] = round(max(0.0, before["MonthlyCharges"] + delta), 2)
        after["TotalCharges"] = round(max(0.0, before["TotalCharges"] + before["tenure"] * delta), 2)
    return after


def _set_internet(c: dict, kind: str) -> dict:
    c = dict(c, InternetService=kind)
    for k in ADD_ONS:
        if kind == "No":
            c[k] = "No internet service"
        elif c[k] == "No internet service":
            c[k] = "No"
    return c


def _set_phone(c: dict, on: bool) -> dict:
    c = dict(c, PhoneService="Yes" if on else "No")
    c["MultipleLines"] = ("No" if c["MultipleLines"] == "No phone service" else c["MultipleLines"]) if on \
        else "No phone service"
    return c


def _consistent(c: dict) -> dict:
    """Re-apply the dataset's rules: add-ons need internet, extra lines need a phone line."""
    c = _set_internet(c, c["InternetService"])
    if c["PhoneService"] == "No":
        c["MultipleLines"] = "No phone service"
    return c


def changes(c: dict) -> list[dict]:
    """Every single change that makes sense for this customer: id, group, label, apply(customer)."""
    out = []
    has_net = c["InternetService"] != "No"
    for k, name in ADD_ONS.items():
        if not has_net:
            break
        if c[k] == "Yes":
            out.append({"id": f"drop:{k}", "group": k, "label": f"Drop {name}", "apply": lambda x, k=k: dict(x, **{k: "No"})})
        else:
            out.append({"id": f"add:{k}", "group": k, "label": f"Add {name}", "apply": lambda x, k=k: dict(x, **{k: "Yes"})})
    if c["PhoneService"] == "Yes":
        if c["MultipleLines"] == "Yes":
            out.append({"id": "drop:MultipleLines", "group": "MultipleLines", "label": "Drop the extra phone lines",
                        "apply": lambda x: dict(x, MultipleLines="No")})
        else:
            out.append({"id": "add:MultipleLines", "group": "MultipleLines", "label": "Add a second phone line",
                        "apply": lambda x: dict(x, MultipleLines="Yes")})
        if has_net:
            out.append({"id": "drop:PhoneService", "group": "PhoneService", "label": "Cancel the phone line",
                        "apply": lambda x: _set_phone(x, False)})
    else:
        out.append({"id": "add:PhoneService", "group": "PhoneService", "label": "Add a phone line",
                    "apply": lambda x: _set_phone(x, True)})
    for kind, label in (("Fiber optic", "Move to fibre internet"), ("DSL", "Move to DSL internet"),
                        ("No", "Cancel internet")):
        if c["InternetService"] != kind and not (kind == "No" and c["PhoneService"] == "No"):
            out.append({"id": f"internet:{kind}", "group": "InternetService", "label": label,
                        "apply": lambda x, kind=kind: _set_internet(x, kind)})
    for ct in CONTRACTS:
        if ct != c["Contract"]:
            label = "Go month-to-month" if ct == "Month-to-month" else f"Sign a {ct.lower()} contract"
            out.append({"id": f"contract:{ct}", "group": "Contract", "label": label,
                        "apply": lambda x, ct=ct: dict(x, Contract=ct)})
    for pm in AUTOPAY:
        if c["PaymentMethod"] != pm:
            out.append({"id": f"pay:{pm}", "group": "PaymentMethod",
                        "label": "Switch to automatic " + ("bank transfer" if "Bank" in pm else "card payment"),
                        "apply": lambda x, pm=pm: dict(x, PaymentMethod=pm)})
    out.append({"id": "paperless", "group": "PaperlessBilling",
                "label": "Stop paperless billing" if c["PaperlessBilling"] == "Yes" else "Start paperless billing",
                "apply": lambda x: dict(x, PaperlessBilling="No" if x["PaperlessBilling"] == "Yes" else "Yes")})
    out.append({"id": "tenure:+12", "group": "tenure", "label": "Stay 12 more months",
                "apply": lambda x: dict(x, tenure=x["tenure"] + 12,
                                        TotalCharges=round(x["TotalCharges"] + 12 * x["MonthlyCharges"], 2))})
    return out


def apply(c: dict, change_ids: list[str], adjust_bill: bool = True) -> dict:
    """Apply several changes in order (one per group; a later one in the same group wins)."""
    by_id = {ch["id"]: ch for ch in changes(c)}
    seen, result = {}, dict(c)
    for cid in change_ids:
        if cid in by_id:
            seen[by_id[cid]["group"]] = cid
    for cid in seen.values():
        ch = by_id[cid]
        nxt = _consistent(ch["apply"](result))    # e.g. "add tech support" after "cancel internet" is a no-op
        result = nxt if ch["group"] == "tenure" else _rebill(result, nxt, adjust_bill)
    return result


def single_changes(c: dict, adjust_bill: bool = True) -> list[tuple[dict, dict]]:
    """[(change, scenario customer)] — one change at a time."""
    return [(ch, apply(c, [ch["id"]], adjust_bill)) for ch in changes(c)]


def tenure_path(c: dict, months=range(0, 73, 3)) -> list[dict]:
    """The same customer at different tenures, with the total billed growing with the bill."""
    per_month = c["MonthlyCharges"]
    return [dict(c, tenure=float(m), TotalCharges=round(m * per_month, 2)) for m in months]
