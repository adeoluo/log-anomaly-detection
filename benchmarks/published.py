"""Published Drain results on Loghub-2k and Loghub-2.0, quoted for comparison (not rerun).

Source: Jiang et al., "A Large-scale Evaluation for Log Parsing Techniques: How Far are We?",
ISSTA 2024, results table RQs_experiments/RQ2/effectiveness_results.csv and efficiency_results.csv
in https://github.com/logpai/loghub-2.0 (Drain column). Times are seconds on the authors' hardware.
"""

from __future__ import annotations

# dataset -> metric -> (Loghub-2k value, Loghub-2.0 value)
DRAIN_PUBLISHED: dict[str, dict[str, tuple[float, float]]] = {
    "Average": {
        "GA": (0.85, 0.84),
        "FGA": (0.74, 0.55),
        "PA": (0.32, 0.47),
        "FTA": (0.28, 0.28),
    },
    "Hadoop": {
        "GA": (0.948, 0.921),
        "FGA": (0.82, 0.785),
        "PA": (0.269, 0.541),
        "FTA": (0.276, 0.384),
    },
    "HDFS": {
        "GA": (0.998, 0.999),
        "FGA": (0.839, 0.935),
        "PA": (0.5, 0.621),
        "FTA": (0.323, 0.609),
    },
    "OpenStack": {
        "GA": (0.733, 0.752),
        "FGA": (0.117, 0.007),
        "PA": (0.018, 0.029),
        "FTA": (0.018, 0.002),
    },
    "Spark": {
        "GA": (0.922, 0.888),
        "FGA": (0.87, 0.861),
        "PA": (0.362, 0.394),
        "FTA": (0.493, 0.412),
    },
    "Zookeeper": {
        "GA": (0.967, 0.994),
        "FGA": (0.854, 0.904),
        "PA": (0.497, 0.843),
        "FTA": (0.354, 0.614),
    },
    "BGL": {
        "GA": (0.963, 0.919),
        "FGA": (0.833, 0.624),
        "PA": (0.342, 0.407),
        "FTA": (0.211, 0.193),
    },
    "HPC": {
        "GA": (0.887, 0.793),
        "FGA": (0.753, 0.309),
        "PA": (0.635, 0.721),
        "FTA": (0.366, 0.152),
    },
    "Thunderbird": {
        "GA": (0.955, 0.831),
        "FGA": (0.773, 0.237),
        "PA": (0.046, 0.216),
        "FTA": (0.248, 0.071),
    },
    "Linux": {
        "GA": (0.69, 0.686),
        "FGA": (0.93, 0.778),
        "PA": (0.183, 0.111),
        "FTA": (0.435, 0.259),
    },
    "Mac": {
        "GA": (0.786, 0.761),
        "FGA": (0.797, 0.229),
        "PA": (0.217, 0.357),
        "FTA": (0.199, 0.069),
    },
    "Apache": {
        "GA": (1.0, 1.0),
        "FGA": (1.0, 1.0),
        "PA": (0.694, 0.727),
        "FTA": (0.5, 0.517),
    },
    "OpenSSH": {
        "GA": (0.789, 0.707),
        "FGA": (0.88, 0.872),
        "PA": (0.508, 0.586),
        "FTA": (0.44, 0.487),
    },
    "HealthApp": {
        "GA": (0.78, 0.862),
        "FGA": (0.351, 0.01),
        "PA": (0.231, 0.312),
        "FTA": (0.109, 0.004),
    },
    "Proxifier": {
        "GA": (0.526, 0.692),
        "FGA": (0.538, 0.206),
        "PA": (0.0, 0.688),
        "FTA": (0.0, 0.176),
    },
}

# dataset -> (Loghub-2k seconds, Loghub-2.0 seconds)
DRAIN_PUBLISHED_SECONDS: dict[str, tuple[float, float]] = {
    "Average": (0.27, 425.36),
    "Hadoop": (0.25, 19.34),
    "HDFS": (0.22, 1123.34),
    "OpenStack": (0.5, 64.3),
    "Spark": (0.21, 1593.77),
    "Zookeeper": (0.22, 6.99),
    "BGL": (0.25, 498.26),
    "HPC": (0.25, 37.84),
    "Thunderbird": (0.22, 2491.79),
    "Linux": (0.29, 2.59),
    "Mac": (0.41, 17.06),
    "Apache": (0.22, 5.08),
    "OpenSSH": (0.26, 74.06),
    "HealthApp": (0.2, 17.66),
    "Proxifier": (0.3, 2.89),
}
