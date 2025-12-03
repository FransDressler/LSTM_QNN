import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix
from sklearn.metrics import classification_report
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class ModelMetrics:
    def __init__(self):
        pass

    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_score_macro': f1_score(y_true, y_pred, average='macro')
        }

    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        y_true_mean = np.mean(y_true)
        naive_forecast = np.full_like(y_true, y_true_mean)
        naive_mae = mean_absolute_error(y_true, naive_forecast)

        mase = mae / naive_mae if naive_mae != 0 else np.inf

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true_mean) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mase': mase,
            'r2': r2
        }

    def calculate_rmsse(self, y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
        """
        Root Mean Squared Scaled Error as mentioned in the paper
        """
        if len(y_train) < 2:
            return np.inf

        naive_forecast = y_train[:-1]
        naive_actual = y_train[1:]
        denominator = np.mean((naive_actual - naive_forecast) ** 2)

        if denominator == 0:
            return np.inf

        numerator = np.mean((y_true - y_pred) ** 2)
        rmsse = np.sqrt(numerator / denominator)

        return rmsse

    def calculate_directional_accuracy(self, y_true_prices: np.ndarray, y_pred_trends: np.ndarray) -> float:
        """
        Calculate directional accuracy - how often the model predicts the correct direction
        """
        actual_directions = np.diff(y_true_prices) > 0
        predicted_directions = y_pred_trends[1:] == 1

        return accuracy_score(actual_directions, predicted_directions)

    def calculate_trend_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Specific metrics for trend prediction
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_up = tp / (tp + fp) if (tp + fp) > 0 else 0
        precision_down = tn / (tn + fn) if (tn + fn) > 0 else 0

        return {
            'sensitivity_up_trend': sensitivity,
            'specificity_down_trend': specificity,
            'precision_up_trend': precision_up,
            'precision_down_trend': precision_down,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }

    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                            y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate all relevant metrics for stock trend prediction
        """
        metrics = {}

        classification_metrics = self.calculate_classification_metrics(y_true, y_pred)
        metrics.update(classification_metrics)

        trend_metrics = self.calculate_trend_metrics(y_true, y_pred)
        metrics.update(trend_metrics)

        if y_pred_proba is not None:
            metrics['log_loss'] = self.calculate_log_loss(y_true, y_pred_proba)

        return metrics

    def calculate_log_loss(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Calculate logarithmic loss
        """
        from sklearn.metrics import log_loss
        return log_loss(y_true, y_pred_proba)

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            class_names: list = None, save_path: str = None):
        """
        Plot confusion matrix
        """
        if class_names is None:
            class_names = ['Down', 'Up']

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        if save_path:
            plt.savefig(save_path)
        plt.show()

        return cm

    def plot_metrics_comparison(self, metrics_dict: Dict[str, Dict], save_path: str = None):
        """
        Plot comparison of metrics across different models
        """
        models = list(metrics_dict.keys())
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score']

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        for i, metric in enumerate(metric_names):
            values = [metrics_dict[model].get(metric, 0) for model in models]
            axes[i].bar(models, values)
            axes[i].set_title(f'{metric.title()}')
            axes[i].set_ylabel('Score')
            axes[i].set_ylim(0, 1)

            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def print_detailed_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                            model_name: str = "xLSTM-TS"):
        """
        Print a detailed classification report
        """
        print(f"\n{'='*50}")
        print(f"DETAILED EVALUATION REPORT - {model_name}")
        print(f"{'='*50}")

        metrics = self.calculate_all_metrics(y_true, y_pred)

        print(f"\nClassification Metrics:")
        print(f"  Accuracy:     {metrics['accuracy']:.4f}")
        print(f"  Precision:    {metrics['precision']:.4f}")
        print(f"  Recall:       {metrics['recall']:.4f}")
        print(f"  F1-Score:     {metrics['f1_score']:.4f}")

        print(f"\nTrend-Specific Metrics:")
        print(f"  Up Trend Sensitivity:   {metrics['sensitivity_up_trend']:.4f}")
        print(f"  Down Trend Specificity: {metrics['specificity_down_trend']:.4f}")
        print(f"  Up Trend Precision:     {metrics['precision_up_trend']:.4f}")
        print(f"  Down Trend Precision:   {metrics['precision_down_trend']:.4f}")

        print(f"\nConfusion Matrix Breakdown:")
        print(f"  True Positives (Correct Up):   {metrics['true_positives']}")
        print(f"  True Negatives (Correct Down): {metrics['true_negatives']}")
        print(f"  False Positives (Wrong Up):    {metrics['false_positives']}")
        print(f"  False Negatives (Wrong Down):  {metrics['false_negatives']}")

        print(f"\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['Down', 'Up']))

        return metrics