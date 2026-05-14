"""Golden QA pairs for the invoice-anomaly-detection codebase.

These are hand-crafted ground-truth Q&A pairs spanning:
  - Simple factual lookups
  - Multi-hop architectural questions
  - Conceptual/design questions
Used to measure pipeline quality and compare naive vs CRAG vs routed retrieval.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass
class GoldenQA:
    id: str
    question: str
    ground_truth: str
    difficulty: Literal["easy", "medium", "hard"]
    category: Literal["factual", "structural", "architectural", "comparative"]
    expected_route: Literal["simple", "complex", "conceptual"]


GOLDEN_QA_PAIRS: list[GoldenQA] = [
    # --- Factual / Simple ---
    GoldenQA(
        id="fq-01",
        question="What machine learning model is used for anomaly detection?",
        ground_truth="The codebase uses Isolation Forest from scikit-learn for anomaly detection. It is an unsupervised algorithm well-suited for detecting outliers in high-dimensional data.",
        difficulty="easy",
        category="factual",
        expected_route="simple",
    ),
    GoldenQA(
        id="fq-02",
        question="What does the AnomalyDetector class do?",
        ground_truth="AnomalyDetector is the main class responsible for detecting anomalies in invoice data. It wraps an Isolation Forest model, providing fit() to train on data and predict() to label records as normal (1) or anomalous (-1).",
        difficulty="easy",
        category="factual",
        expected_route="simple",
    ),
    GoldenQA(
        id="fq-03",
        question="How is invoice data loaded into the system?",
        ground_truth="Invoice data is loaded from CSV files using pandas read_csv. A dedicated load_data function accepts a file path and returns a DataFrame ready for preprocessing.",
        difficulty="easy",
        category="factual",
        expected_route="simple",
    ),
    GoldenQA(
        id="fq-04",
        question="What libraries does the codebase import?",
        ground_truth="The codebase imports pandas for data manipulation, scikit-learn (sklearn.ensemble.IsolationForest) for anomaly detection, numpy for numerical operations, and os/pathlib for file system operations.",
        difficulty="easy",
        category="factual",
        expected_route="simple",
    ),
    GoldenQA(
        id="fq-05",
        question="What is the default contamination parameter for anomaly detection?",
        ground_truth="The default contamination parameter is 0.05, meaning the model assumes approximately 5% of the training data are anomalies/outliers.",
        difficulty="easy",
        category="factual",
        expected_route="simple",
    ),
    GoldenQA(
        id="fq-06",
        question="What does the predict method return?",
        ground_truth="The predict method returns an array where -1 indicates an anomaly (outlier) and 1 indicates a normal data point, following the standard scikit-learn Isolation Forest output convention.",
        difficulty="easy",
        category="factual",
        expected_route="simple",
    ),
    GoldenQA(
        id="fq-07",
        question="What preprocessing steps are applied to the invoice data?",
        ground_truth="The preprocessing pipeline normalizes or scales numerical features, handles missing values, and selects relevant feature columns from the invoice DataFrame before passing data to the anomaly detection model.",
        difficulty="medium",
        category="factual",
        expected_route="simple",
    ),

    # --- Structural / Multi-hop ---
    GoldenQA(
        id="sq-01",
        question="How does the data flow from a CSV file to an anomaly prediction?",
        ground_truth="Data flows as follows: (1) load_data() reads the CSV into a pandas DataFrame; (2) a preprocessing function cleans and normalizes the features; (3) AnomalyDetector.fit() trains the Isolation Forest; (4) AnomalyDetector.predict() labels each row as normal or anomalous.",
        difficulty="medium",
        category="structural",
        expected_route="complex",
    ),
    GoldenQA(
        id="sq-02",
        question="What is the relationship between the data loading module and the AnomalyDetector class?",
        ground_truth="The data loading module (load_data function) is a dependency of the AnomalyDetector pipeline. It provides the pandas DataFrame that AnomalyDetector.fit() consumes for training and AnomalyDetector.predict() uses for inference.",
        difficulty="medium",
        category="structural",
        expected_route="complex",
    ),
    GoldenQA(
        id="sq-03",
        question="Which functions call the AnomalyDetector and how do they use its output?",
        ground_truth="The main pipeline or orchestration function calls AnomalyDetector.fit() during training, then AnomalyDetector.predict() during inference. The output array of 1s and -1s is used to filter or flag anomalous invoice records for further review.",
        difficulty="medium",
        category="structural",
        expected_route="complex",
    ),
    GoldenQA(
        id="sq-04",
        question="How is the trained model persisted or reloaded between runs?",
        ground_truth="The codebase may use joblib or pickle to serialize the trained Isolation Forest model to disk, allowing it to be reloaded without retraining. If not implemented, the model is retrained from scratch on each run.",
        difficulty="hard",
        category="structural",
        expected_route="complex",
    ),

    # --- Architectural / Conceptual ---
    GoldenQA(
        id="aq-01",
        question="What is the overall architecture of the invoice anomaly detection system?",
        ground_truth="The system follows a classic ML pipeline architecture: data ingestion (CSV loading) → preprocessing (feature engineering, normalization) → model training (Isolation Forest) → inference (anomaly scoring) → output (labeled DataFrame or report). It is designed as a batch processing pipeline rather than a real-time streaming system.",
        difficulty="medium",
        category="architectural",
        expected_route="conceptual",
    ),
    GoldenQA(
        id="aq-02",
        question="What are the main design patterns used in the codebase?",
        ground_truth="The codebase uses the Strategy pattern (swappable anomaly detection algorithms via a class interface), the Pipeline pattern (sequential data transformation steps), and functional decomposition (separate functions for loading, preprocessing, and detection).",
        difficulty="hard",
        category="architectural",
        expected_route="conceptual",
    ),
    GoldenQA(
        id="aq-03",
        question="What are the main sources of technical debt in this codebase?",
        ground_truth="Potential technical debt includes: hardcoded file paths and configuration parameters, lack of input validation on the CSV data, no unit tests for the preprocessing pipeline, tight coupling between data loading and model training, and limited error handling for missing or malformed data.",
        difficulty="hard",
        category="architectural",
        expected_route="conceptual",
    ),
    GoldenQA(
        id="aq-04",
        question="How scalable is the current anomaly detection approach and what would need to change for production?",
        ground_truth="The current batch pandas-based approach is suitable for moderate data volumes. For production at scale, changes needed include: streaming data support (Kafka/Flink), distributed processing (Spark), model versioning and A/B testing infrastructure, real-time scoring API (FastAPI), monitoring for model drift, and a feature store.",
        difficulty="hard",
        category="architectural",
        expected_route="conceptual",
    ),

    # --- Comparative ---
    GoldenQA(
        id="cq-01",
        question="Compare the fit and predict methods of AnomalyDetector in terms of what data they expect and return.",
        ground_truth="fit() takes a 2D feature matrix X (numpy array or DataFrame), trains the Isolation Forest in-place, and returns self for method chaining. predict() also takes a 2D feature matrix X and returns a 1D array of labels: 1 for normal, -1 for anomaly. Both methods delegate directly to the underlying sklearn model.",
        difficulty="medium",
        category="comparative",
        expected_route="complex",
    ),
    GoldenQA(
        id="cq-02",
        question="What is the difference between how training data and inference data are handled in the pipeline?",
        ground_truth="Training data goes through the full preprocessing pipeline including fit_transform (fitting scalers to training statistics), then is passed to AnomalyDetector.fit(). Inference data only uses transform (applying already-fitted scaler statistics) to avoid data leakage, then is passed to AnomalyDetector.predict().",
        difficulty="hard",
        category="comparative",
        expected_route="complex",
    ),
    GoldenQA(
        id="cq-03",
        question="How does the anomaly detection approach compare to a supervised classification approach?",
        ground_truth="Isolation Forest is unsupervised — it requires no labeled anomaly examples and learns 'normal' patterns from unlabeled data. A supervised classifier would need labeled (normal/anomalous) training data, would likely achieve higher precision on known anomaly types, but would fail on novel anomaly patterns. The unsupervised approach is better when labeled data is scarce.",
        difficulty="hard",
        category="comparative",
        expected_route="conceptual",
    ),
    GoldenQA(
        id="cq-04",
        question="Which parts of the codebase are most tightly coupled and which are most modular?",
        ground_truth="The data loading and preprocessing steps tend to be tightly coupled (preprocessing assumes specific CSV column names from load_data). The AnomalyDetector class is the most modular component — it has a clean interface and could be swapped for another sklearn estimator. The main pipeline script is typically the most tightly coupled orchestrator.",
        difficulty="hard",
        category="comparative",
        expected_route="complex",
    ),
]


def get_by_difficulty(difficulty: str) -> list[GoldenQA]:
    return [q for q in GOLDEN_QA_PAIRS if q.difficulty == difficulty]


def get_by_category(category: str) -> list[GoldenQA]:
    return [q for q in GOLDEN_QA_PAIRS if q.category == category]
