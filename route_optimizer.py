"""
route_optimizer.py
==================
Pure-Python utility module for route analysis and vehicle recommendations.
Imported by both train_model.py and app.py.
"""

from __future__ import annotations
import numpy as np


# ── Region coordinates (approximate Indian region centroids) ──────────────────
REGION_COORDS: dict[str, tuple[float, float]] = {
    "north":   (28.6, 77.2),   # Delhi NCR
    "south":   (12.9, 80.1),   # Chennai / Bengaluru
    "east":    (22.5, 88.3),   # Kolkata
    "west":    (19.0, 72.8),   # Mumbai
    "central": (21.1, 82.0),   # Madhya Pradesh
}

# ── Weather risk multipliers ───────────────────────────────────────────────────
WEATHER_RISK: dict[str, float] = {
    "clear":  0.0,
    "cold":   0.15,
    "hot":    0.10,
    "foggy":  0.25,
    "rainy":  0.35,
    "stormy": 0.55,
}

# ── Vehicle speed profiles (km/h, realistic Indian road estimate) ─────────────
VEHICLE_SPEED: dict[str, float] = {
    "bike":    40.0,
    "ev bike": 38.0,
    "scooter": 35.0,
    "van":     55.0,
    "ev van":  52.0,
    "truck":   45.0,
}

# ── Delivery mode SLA windows (hours) ─────────────────────────────────────────
MODE_SLA: dict[str, float] = {
    "express":  4.0,
    "same day": 8.0,
    "two day":  48.0,
    "standard": 72.0,
}

# ── Fuel / energy cost rate (₹ per km) ────────────────────────────────────────
VEHICLE_COST_RATE: dict[str, float] = {
    "bike":    3.5,
    "ev bike": 1.8,
    "scooter": 4.0,
    "van":     8.5,
    "ev van":  4.5,
    "truck":   14.0,
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi  = np.radians(lat2 - lat1)
    dlam  = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def estimate_delivery_time(
    distance_km: float,
    vehicle: str,
    weather: str,
) -> float:
    """Return estimated delivery time in hours."""
    speed      = VEHICLE_SPEED.get(vehicle, 45.0)
    risk       = WEATHER_RISK.get(weather, 0.0)
    base_hours = distance_km / speed
    return round(base_hours * (1 + risk), 2)


def estimate_cost(distance_km: float, vehicle: str, weight_kg: float) -> float:
    """Return estimated delivery cost in ₹."""
    rate       = VEHICLE_COST_RATE.get(vehicle, 6.0)
    base_cost  = distance_km * rate
    weight_fee = weight_kg * 5.0        # ₹5 per kg
    return round(base_cost + weight_fee, 2)


def is_sla_feasible(
    distance_km: float,
    vehicle: str,
    weather: str,
    delivery_mode: str,
) -> tuple[bool, float]:
    """
    Returns (feasible, estimated_hours).
    feasible=True if estimated time fits within the mode SLA.
    """
    estimated   = estimate_delivery_time(distance_km, vehicle, weather)
    sla         = MODE_SLA.get(delivery_mode, 72.0)
    return estimated <= sla, estimated


def score_route_option(
    distance_km: float,
    vehicle: str,
    weather: str,
    delivery_mode: str,
    package_type: str,
    weight_kg: float,
) -> dict:
    """
    Score a single (vehicle, route) combination.
    Returns a rich dict with all computed metrics.
    """
    est_time     = estimate_delivery_time(distance_km, vehicle, weather)
    est_cost     = estimate_cost(distance_km, vehicle, weight_kg)
    sla, _       = is_sla_feasible(distance_km, vehicle, weather, delivery_mode)
    risk         = WEATHER_RISK.get(weather, 0.0)
    sla_hours    = MODE_SLA.get(delivery_mode, 72.0)

    # Composite score (lower delay risk + SLA fit + lower cost preference)
    delay_risk_score  = 1 - risk                        # 0-1
    sla_margin        = max(0, (sla_hours - est_time) / sla_hours)  # 0-1
    cost_norm         = 1 - min(est_cost / 5000, 1.0)  # 0-1 (penalise expensive)

    heavy_penalty = 0.0
    if package_type in {"automobile parts", "furniture", "electronics"} and \
       vehicle in {"bike", "ev bike", "scooter"}:
        heavy_penalty = 0.3

    composite = (
        0.40 * delay_risk_score +
        0.35 * sla_margin       +
        0.15 * cost_norm        -
        heavy_penalty
    )
    composite = max(0.0, min(1.0, composite))

    return {
        "vehicle":        vehicle,
        "est_time_hr":    est_time,
        "est_cost_inr":   est_cost,
        "sla_feasible":   sla,
        "weather_risk":   round(risk * 100, 1),
        "composite_score": round(composite, 4),
    }


def get_all_route_options(
    distance_km: float,
    weather: str,
    delivery_mode: str,
    package_type: str,
    weight_kg: float,
) -> list[dict]:
    """Return all vehicle options ranked by composite score."""
    results = []
    for v in VEHICLE_SPEED:
        r = score_route_option(
            distance_km, v, weather, delivery_mode, package_type, weight_kg
        )
        results.append(r)
    return sorted(results, key=lambda x: x["composite_score"], reverse=True)


def get_risk_label(prob: float) -> tuple[str, str]:
    """Return (label, emoji) for a delay probability."""
    if prob < 0.30:
        return "Low Risk",    "🟢"
    elif prob < 0.55:
        return "Medium Risk", "🟡"
    elif prob < 0.75:
        return "High Risk",   "🟠"
    else:
        return "Critical",    "🔴"
