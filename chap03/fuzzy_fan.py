#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fuzzy thermostat controller demo.
Implements fuzzy sets, four rules, and centroid defuzzification.
"""

from typing import Dict, Callable
import math

# ---------- Generic fuzzy primitives ----------
def trapezoid(x, a, b, c, d):
    """Trapezoid membership: 0..1 with core [b,c]."""
    if x <= a or x >= d: return 0.0
    if b <= x <= c:      return 1.0
    if a < x < b:        return (x - a) / (b - a)
    # c < x < d
    return (d - x) / (d - c)

def triangle(x, a, b, c):
    """Triangle as trapezoid with b=c."""
    return trapezoid(x, a, b, b, c)

fuzzy_and = min
fuzzy_or  = max
fuzzy_not = lambda a: 1.0 - a

# ---------- Input fuzzification ----------
def fuzzify_temp(t: float) -> Dict[str, float]:
    # cold: (0,0,10,20), warm: triangle (10,20,30), hot: (20,30,40,40)
    return {
        "cold":  trapezoid(t, 0, 0, 10, 20),
        "warm":  triangle(t, 10, 20, 30),
        "hot":   trapezoid(t, 20, 30, 40, 40),
    }

def fuzzify_humidity(h: float) -> Dict[str, float]:
    # dry: (0,0,40,60), humid: (40,60,100,100)
    return {
        "dry":   trapezoid(h, 0, 0, 40, 60),
        "humid": trapezoid(h, 40, 60, 100, 100),
    }

# ---------- Output membership (fan speed in 0..100%) ----------
def mu_off(x):    return trapezoid(x, 0, 0, 10, 20)
def mu_low(x):    return triangle(x, 10, 30, 50)
def mu_medium(x): return trapezoid(x, 30, 50, 60, 80)
def mu_fast(x):   return trapezoid(x, 60, 80, 100, 100)

MU_OUT = {
    "off":    mu_off,
    "slow":   mu_low,
    "medium": mu_medium,
    "fast":   mu_fast,
}

# ---------- Rule base ----------
def rules(temp: float, hum: float) -> Dict[str, float]:
    mu_t = fuzzify_temp(temp)
    mu_h = fuzzify_humidity(hum)

    # R1: IF (cold AND NOT humid) THEN fan = off
    # R2: IF (cold AND humid) THEN fan = low
    # R3: IF warm THEN fan = medium
    # R4: IF (warm OR hot) AND humid THEN fan = fast
    return {
        "off":    fuzzy_and(mu_t["cold"], fuzzy_not(mu_h["humid"])),
        "slow":   fuzzy_and(mu_t["cold"], mu_h["humid"]),
        "medium": mu_t["warm"],
        "fast":   fuzzy_and(fuzzy_or(mu_t["warm"], mu_t["hot"]), mu_h["humid"])
    }

# ---------- Aggregation + centroid defuzzification ----------
def aggregate_and_defuzzify(activations: Dict[str, float],
                            xmin=0.0, xmax=100.0, step=0.5) -> float:
    """Centroid of aggregated (max of clipped consequents)."""
    num, den = 0.0, 0.0
    x = xmin
    while x <= xmax + 1e-9:
        mu_agg = 0.0
        for label, r in activations.items():
            mu = MU_OUT[label](x)
            mu_agg = max(mu_agg, min(r, mu))  # clip then max-aggregate
        num += x * mu_agg
        den += mu_agg
        x += step
    return num / den if den > 1e-12 else 0.0

# ---------- End-to-end controller ----------
def fan_controller(temp: float, humidity: float) -> Dict[str, float]:
    acts = rules(temp, humidity)
    crisp = aggregate_and_defuzzify(acts)
    return {"crisp_speed": crisp, **acts}

# --- Demo (27.5°C, 70% RH): warm=0.25, hot=0.75, humid=0.8 -> fast dominates
if __name__ == "__main__":
    out = fan_controller(27.5, 70.0)
    # e.g., {'crisp_speed': ~78, 'off': 0.0, 'low': 0.0.., 'medium': 0.25, 'fast': 0.75}
    print(out)
