import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from src.models.base_model import StockPredictor
from src.models.classification_utils import compute_classification_metrics
from src.models.evaluation import calculate_metrics, calculate_strategy_returns
from src.models.trading_config import predictions_to_signals

logger = logging.getLogger(__name__)


def _build_direction_split(
    split_df: pd.DataFrame,
    price_col: str,
    feature_cols: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build classification arrays from a single split only.
    Last row is dropped because next-day label is unavailable within split.
    """
    if len(split_df) < 2:
        return (
            np.empty((0, len(feature_cols))),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=float),
        )

    labeled = split_df.copy()
    labeled["_target"] = (labeled[price_col].shift(-1) > labeled[price_col]).astype(int)
    labeled = labeled.iloc[:-1].copy()
    X = labeled[list(feature_cols)].values
    y = labeled["_target"].values
    prices = labeled[price_col].values
    return X, y, prices


def _create_lstm_sequences(
    features_scaled: np.ndarray,
    close_scaled: np.ndarray,
    close_raw: np.ndarray,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create LSTM sequences and return:
    X: [samples, sequence_length, n_features]
    y: scaled close targets at t
    prev_close: raw close at t-1 (used for direction labels)
    """
    X, y, prev_close = [], [], []
    for i in range(sequence_length, len(features_scaled)):
        X.append(features_scaled[i - sequence_length : i, :])
        y.append(close_scaled[i])
        prev_close.append(close_raw[i - 1])

    if not X:
        return (
            np.empty((0, sequence_length, features_scaled.shape[1])),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(prev_close, dtype=np.float32),
    )


class WalkForwardValidator:
    """
    Framework for walk-forward validation of time-series models.
    Supports rolling windows for training, validation, and testing.
    """

    def __init__(
        self,
        train_years: int = 3,
        val_months: int = 3,
        test_months: int = 3,
        step_months: int = 3,
    ):
        self.train_years = train_years
        self.val_months = val_months
        self.test_months = test_months
        self.step_months = step_months

    def _get_splits(self, data: pd.DataFrame) -> List[Dict[str, pd.DataFrame]]:
        """Generate chronological splits based on the window sizes."""
        splits = []
        if "date" not in data.columns:
            raise ValueError("Data must contain a 'date' column.")

        data = data.copy()
        data["date"] = pd.to_datetime(data["date"])
        data = data.sort_values("date").reset_index(drop=True)

        min_date = data["date"].min()
        max_date = data["date"].max()
        current_train_start = min_date

        while True:
            train_end = current_train_start + relativedelta(years=self.train_years)
            val_end = train_end + relativedelta(months=self.val_months)
            test_end = val_end + relativedelta(months=self.test_months)

            if test_end > max_date:
                break

            train_mask = (data["date"] >= current_train_start) & (data["date"] < train_end)
            val_mask = (data["date"] >= train_end) & (data["date"] < val_end)
            test_mask = (data["date"] >= val_end) & (data["date"] < test_end)

            train_df = data[train_mask].copy()
            val_df = data[val_mask].copy()
            test_df = data[test_mask].copy()
            if train_df.empty or val_df.empty or test_df.empty:
                current_train_start = current_train_start + relativedelta(months=self.step_months)
                continue

            splits.append(
                {
                    "train": train_df,
                    "val": val_df,
                    "test": test_df,
                    "metadata": {
                        "train_start": current_train_start,
                        "train_end": train_end,
                        "val_start": train_end,
                        "val_end": val_end,
                        "test_start": val_end,
                        "test_end": test_end,
                    },
                }
            )

            current_train_start = current_train_start + relativedelta(months=self.step_months)

        return splits

    @staticmethod
    def _call_model_train(
        model: StockPredictor,
        X_train,
        y_train,
        X_val,
        y_val,
    ):
        """Call train with new API and fallback for older signature objects."""
        try:
            return model.train(X_train, y_train, X_val=X_val, y_val=y_val)
        except TypeError:
            # Backward compatibility for legacy tests or external models.
            return model.train(X_train, y_train, X_val, y_val)

    @staticmethod
    def _resolve_columns(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
        lookup = {col.lower(): col for col in df.columns}
        resolved = []
        missing = []
        for col in columns:
            match = lookup.get(col.lower())
            if match is None:
                missing.append(col)
            else:
                resolved.append(match)
        if missing:
            raise ValueError(f"Missing columns in walk-forward data: {missing}")
        return resolved

    def _validate_classification(
        self,
        model: StockPredictor,
        split: Dict[str, Any],
        feature_cols: Sequence[str],
        price_col: str,
    ) -> Optional[Dict[str, Any]]:
        train_df = split["train"]
        val_df = split["val"]
        test_df = split["test"]

        X_train, y_train, _ = _build_direction_split(train_df, price_col=price_col, feature_cols=feature_cols)
        X_val, y_val, val_prices = _build_direction_split(val_df, price_col=price_col, feature_cols=feature_cols)
        X_test, y_test, test_prices = _build_direction_split(test_df, price_col=price_col, feature_cols=feature_cols)

        if min(len(X_train), len(X_val), len(X_test)) == 0:
            logger.warning("Skipping split due to insufficient rows after split-safe targeting.")
            return None

        self._call_model_train(model, X_train, y_train, X_val, y_val)

        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)
        test_preds = model.predict(X_test)

        train_probs = model.predict_proba(X_train)[:, 1] if hasattr(model, "predict_proba") else None
        val_probs = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
        test_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        train_cls = compute_classification_metrics(y_train, train_preds, train_probs)
        val_cls = compute_classification_metrics(y_val, val_preds, val_probs)
        test_cls = compute_classification_metrics(y_test, test_preds, test_probs)

        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

        overfitting_ratio = (
            val_cls["balanced_accuracy"] / train_cls["balanced_accuracy"]
            if train_cls["balanced_accuracy"] > 0
            else 0
        )

        test_signals = predictions_to_signals(test_preds, index=test_df.index[:-1])
        strategy_returns = calculate_strategy_returns(test_signals, pd.Series(test_prices, index=test_df.index[:-1]))
        financial_metrics = calculate_metrics(strategy_returns)

        return {
            "train_accuracy": train_cls["accuracy"],
            "val_accuracy": val_cls["accuracy"],
            "test_accuracy": test_cls["accuracy"],
            "train_balanced_accuracy": train_cls["balanced_accuracy"],
            "val_balanced_accuracy": val_cls["balanced_accuracy"],
            "test_balanced_accuracy": test_cls["balanced_accuracy"],
            "train_precision": train_cls["precision"],
            "val_precision": val_cls["precision"],
            "test_precision": test_cls["precision"],
            "train_recall": train_cls["recall"],
            "val_recall": val_cls["recall"],
            "test_recall": test_cls["recall"],
            "train_f1": train_cls["f1"],
            "val_f1": val_cls["f1"],
            "test_f1": test_cls["f1"],
            "train_roc_auc": train_cls["roc_auc"],
            "val_roc_auc": val_cls["roc_auc"],
            "test_roc_auc": test_cls["roc_auc"],
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "test_rmse": test_rmse,
            "overfitting_ratio": overfitting_ratio,
            **financial_metrics,
        }

    def _validate_lstm_regression(
        self,
        model: StockPredictor,
        split: Dict[str, Any],
        feature_cols: Sequence[str],
        price_col: str,
        sequence_length: int,
    ) -> Optional[Dict[str, Any]]:
        train_df = split["train"]
        val_df = split["val"]
        test_df = split["test"]

        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler = MinMaxScaler(feature_range=(0, 1))

        train_features = train_df[list(feature_cols)].values.astype(float)
        val_features = val_df[list(feature_cols)].values.astype(float)
        test_features = test_df[list(feature_cols)].values.astype(float)

        train_close_raw = train_df[price_col].values.astype(float)
        val_close_raw = val_df[price_col].values.astype(float)
        test_close_raw = test_df[price_col].values.astype(float)

        feature_scaler.fit(train_features)
        target_scaler.fit(train_close_raw.reshape(-1, 1))

        train_scaled = feature_scaler.transform(train_features)
        val_scaled = feature_scaler.transform(val_features)
        test_scaled = feature_scaler.transform(test_features)

        train_close_scaled = target_scaler.transform(train_close_raw.reshape(-1, 1)).reshape(-1)
        val_close_scaled = target_scaler.transform(val_close_raw.reshape(-1, 1)).reshape(-1)
        test_close_scaled = target_scaler.transform(test_close_raw.reshape(-1, 1)).reshape(-1)

        X_train, y_train, train_prev = _create_lstm_sequences(
            train_scaled,
            train_close_scaled,
            train_close_raw,
            sequence_length=sequence_length,
        )
        val_context_scaled = np.vstack([train_scaled[-sequence_length:], val_scaled])
        val_context_close_scaled = np.concatenate([train_close_scaled[-sequence_length:], val_close_scaled])
        val_context_close_raw = np.concatenate([train_close_raw[-sequence_length:], val_close_raw])
        X_val, y_val, val_prev = _create_lstm_sequences(
            val_context_scaled,
            val_context_close_scaled,
            val_context_close_raw,
            sequence_length=sequence_length,
        )

        pre_test_scaled = np.vstack([train_scaled, val_scaled])
        pre_test_close_scaled = np.concatenate([train_close_scaled, val_close_scaled])
        pre_test_close_raw = np.concatenate([train_close_raw, val_close_raw])
        test_context_scaled = np.vstack([pre_test_scaled[-sequence_length:], test_scaled])
        test_context_close_scaled = np.concatenate([pre_test_close_scaled[-sequence_length:], test_close_scaled])
        test_context_close_raw = np.concatenate([pre_test_close_raw[-sequence_length:], test_close_raw])
        X_test, y_test, test_prev = _create_lstm_sequences(
            test_context_scaled,
            test_context_close_scaled,
            test_context_close_raw,
            sequence_length=sequence_length,
        )

        if min(len(X_train), len(X_val), len(X_test)) == 0:
            logger.warning("Skipping split due to insufficient LSTM sequence rows.")
            return None

        model.train(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
        )

        train_pred_scaled = model.predict(X_train).reshape(-1, 1)
        val_pred_scaled = model.predict(X_val).reshape(-1, 1)
        test_pred_scaled = model.predict(X_test).reshape(-1, 1)

        train_actual = target_scaler.inverse_transform(y_train.reshape(-1, 1)).reshape(-1)
        val_actual = target_scaler.inverse_transform(y_val.reshape(-1, 1)).reshape(-1)
        test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

        train_pred = target_scaler.inverse_transform(train_pred_scaled).reshape(-1)
        val_pred = target_scaler.inverse_transform(val_pred_scaled).reshape(-1)
        test_pred = target_scaler.inverse_transform(test_pred_scaled).reshape(-1)

        train_rmse = np.sqrt(mean_squared_error(train_actual, train_pred))
        val_rmse = np.sqrt(mean_squared_error(val_actual, val_pred))
        test_rmse = np.sqrt(mean_squared_error(test_actual, test_pred))

        train_actual_dir = (train_actual > train_prev).astype(int)
        val_actual_dir = (val_actual > val_prev).astype(int)
        test_actual_dir = (test_actual > test_prev).astype(int)

        train_pred_dir = (train_pred > train_prev).astype(int)
        val_pred_dir = (val_pred > val_prev).astype(int)
        test_pred_dir = (test_pred > test_prev).astype(int)

        train_cls = compute_classification_metrics(train_actual_dir, train_pred_dir)
        val_cls = compute_classification_metrics(val_actual_dir, val_pred_dir)
        test_cls = compute_classification_metrics(test_actual_dir, test_pred_dir)

        overfitting_ratio = (
            val_cls["balanced_accuracy"] / train_cls["balanced_accuracy"]
            if train_cls["balanced_accuracy"] > 0
            else 0
        )

        test_signals = predictions_to_signals(test_pred_dir)
        strategy_returns = calculate_strategy_returns(test_signals, pd.Series(test_actual))
        financial_metrics = calculate_metrics(strategy_returns)

        return {
            "train_accuracy": train_cls["accuracy"],
            "val_accuracy": val_cls["accuracy"],
            "test_accuracy": test_cls["accuracy"],
            "train_balanced_accuracy": train_cls["balanced_accuracy"],
            "val_balanced_accuracy": val_cls["balanced_accuracy"],
            "test_balanced_accuracy": test_cls["balanced_accuracy"],
            "train_precision": train_cls["precision"],
            "val_precision": val_cls["precision"],
            "test_precision": test_cls["precision"],
            "train_recall": train_cls["recall"],
            "val_recall": val_cls["recall"],
            "test_recall": test_cls["recall"],
            "train_f1": train_cls["f1"],
            "val_f1": val_cls["f1"],
            "test_f1": test_cls["f1"],
            "train_roc_auc": float("nan"),
            "val_roc_auc": float("nan"),
            "test_roc_auc": float("nan"),
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "test_rmse": test_rmse,
            "overfitting_ratio": overfitting_ratio,
            **financial_metrics,
        }

    def validate(
        self,
        model: StockPredictor,
        data: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        price_col: str = "close",
        task_type: str = "classification",
        sequence_length: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Run walk-forward validation.

        Args:
            model: Model implementing StockPredictor.
            data: DataFrame containing at least date and feature columns.
            feature_cols: List of feature column names.
            target_col: For classification, target/price source. For LSTM, regression target column.
            price_col: Price column used for strategy returns.
            task_type: "classification" or "lstm_regression".
            sequence_length: Sequence length for LSTM validation.
        """
        splits = self._get_splits(data)
        if not splits:
            logger.warning("No splits generated. Check data range and window sizes.")
            return []

        resolved_feature_cols = self._resolve_columns(data, feature_cols)
        resolved_price_col = self._resolve_columns(data, [price_col])[0]
        resolved_target_col = self._resolve_columns(data, [target_col])[0]

        results = []
        for i, split in enumerate(splits):
            logger.info(
                "Processing Step %s/%s: %s to %s",
                i + 1,
                len(splits),
                split["metadata"]["test_start"].date(),
                split["metadata"]["test_end"].date(),
            )

            if task_type == "lstm_regression":
                metrics = self._validate_lstm_regression(
                    model=model,
                    split=split,
                    feature_cols=resolved_feature_cols,
                    price_col=resolved_target_col,
                    sequence_length=sequence_length,
                )
            else:
                metrics = self._validate_classification(
                    model=model,
                    split=split,
                    feature_cols=resolved_feature_cols,
                    price_col=resolved_price_col,
                )

            if metrics is None:
                continue

            results.append(
                {
                    "step": i + 1,
                    "metadata": split["metadata"],
                    "metrics": metrics,
                }
            )

        return results
