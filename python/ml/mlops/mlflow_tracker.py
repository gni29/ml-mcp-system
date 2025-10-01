"""
MLflow Integration
Experiment tracking and model registry for ML workflows

Features:
- Experiment tracking (parameters, metrics, artifacts)
- Model registry (versioning, staging, production)
- Run comparison and analysis
- Model lineage tracking
- Artifact storage (local, S3, Azure, GCS)
- Automatic logging for scikit-learn, XGBoost, TensorFlow

Usage:
    from mlflow_tracker import MLflowTracker

    tracker = MLflowTracker(experiment_name='customer_churn')

    with tracker.start_run(run_name='xgboost_v1'):
        tracker.log_params({'max_depth': 5, 'n_estimators': 100})
        tracker.log_metrics({'accuracy': 0.92, 'f1': 0.89})
        tracker.log_model(model, 'xgboost_model')

    # Register model
    tracker.register_model('xgboost_model', 'ChurnPredictor', stage='Production')
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import joblib

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    import mlflow.tensorflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not installed. Install with: pip install mlflow")


class MLflowTracker:
    """
    MLflow integration for experiment tracking and model registry

    Provides simplified interface for MLflow operations
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = 'default',
        artifact_location: Optional[str] = None
    ):
        """
        Initialize MLflow tracker

        Args:
            tracking_uri: MLflow tracking server URI (default: local mlruns/)
            experiment_name: Name of experiment
            artifact_location: Location to store artifacts (S3, Azure, GCS, local)
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is required. Install with: pip install mlflow")

        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Create/get experiment
        self.experiment_name = experiment_name
        try:
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location
            )
        except:
            # Experiment already exists
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = self.experiment.experiment_id

        mlflow.set_experiment(experiment_name)

        # Current run
        self.current_run = None

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        Start a new MLflow run

        Args:
            run_name: Name for the run
            tags: Additional tags for the run

        Returns:
            Context manager for the run
        """
        self.current_run = mlflow.start_run(run_name=run_name, tags=tags)
        return self.current_run

    def end_run(self):
        """End the current run"""
        if self.current_run:
            mlflow.end_run()
            self.current_run = None

    def log_param(self, key: str, value: Any):
        """Log a single parameter"""
        mlflow.log_param(key, value)

    def log_params(self, params: Dict[str, Any]):
        """
        Log multiple parameters

        Args:
            params: Dictionary of parameter names and values
        """
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """
        Log a single metric

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number for tracking over time
        """
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log multiple metrics

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number
        """
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log a file as an artifact

        Args:
            local_path: Path to local file
            artifact_path: Path within artifact store
        """
        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """
        Log a directory of artifacts

        Args:
            local_dir: Local directory path
            artifact_path: Path within artifact store
        """
        mlflow.log_artifacts(local_dir, artifact_path=artifact_path)

    def log_model(
        self,
        model: Any,
        artifact_path: str = 'model',
        registered_model_name: Optional[str] = None,
        **kwargs
    ):
        """
        Log a model

        Args:
            model: Model object
            artifact_path: Path within artifact store
            registered_model_name: Name to register model under
            **kwargs: Additional arguments for model logging
        """
        # Auto-detect model type and log appropriately
        model_type = type(model).__name__

        if 'sklearn' in str(type(model).__module__):
            mlflow.sklearn.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name,
                **kwargs
            )
        elif 'xgboost' in str(type(model).__module__):
            mlflow.xgboost.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name,
                **kwargs
            )
        else:
            # Generic logging with joblib
            mlflow.sklearn.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name,
                **kwargs
            )

    def log_dict(self, dictionary: Dict, artifact_file: str):
        """
        Log a dictionary as JSON

        Args:
            dictionary: Dictionary to log
            artifact_file: Filename for artifact
        """
        mlflow.log_dict(dictionary, artifact_file)

    def log_figure(self, figure, artifact_file: str):
        """
        Log a matplotlib figure

        Args:
            figure: Matplotlib figure
            artifact_file: Filename for artifact
        """
        mlflow.log_figure(figure, artifact_file)

    def set_tag(self, key: str, value: str):
        """Set a tag for the current run"""
        mlflow.set_tag(key, value)

    def set_tags(self, tags: Dict[str, str]):
        """Set multiple tags"""
        mlflow.set_tags(tags)

    def register_model(
        self,
        model_uri: str,
        name: str,
        stage: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Register a model in MLflow Model Registry

        Args:
            model_uri: URI of model (e.g., 'runs:/<run_id>/model')
            name: Registered model name
            stage: Stage to promote to ('Staging', 'Production', 'Archived')
            description: Model description

        Returns:
            Model version information
        """
        # Register model
        result = mlflow.register_model(model_uri, name)

        # Set description if provided
        if description:
            client = mlflow.tracking.MlflowClient()
            client.update_model_version(
                name=name,
                version=result.version,
                description=description
            )

        # Transition to stage if provided
        if stage:
            self.transition_model_stage(name, result.version, stage)

        return {
            'name': result.name,
            'version': result.version,
            'creation_timestamp': result.creation_timestamp,
            'current_stage': result.current_stage
        }

    def transition_model_stage(self, name: str, version: int, stage: str):
        """
        Transition model to a different stage

        Args:
            name: Model name
            version: Model version
            stage: Target stage ('Staging', 'Production', 'Archived')
        """
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage
        )

    def load_model(self, model_uri: str) -> Any:
        """
        Load a model from MLflow

        Args:
            model_uri: Model URI (e.g., 'models:/ModelName/Production')

        Returns:
            Loaded model
        """
        return mlflow.pyfunc.load_model(model_uri)

    def get_run(self, run_id: str) -> Dict[str, Any]:
        """
        Get information about a run

        Args:
            run_id: Run ID

        Returns:
            Run information
        """
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)

        return {
            'run_id': run.info.run_id,
            'experiment_id': run.info.experiment_id,
            'status': run.info.status,
            'start_time': run.info.start_time,
            'end_time': run.info.end_time,
            'params': run.data.params,
            'metrics': run.data.metrics,
            'tags': run.data.tags
        }

    def search_runs(
        self,
        filter_string: str = "",
        order_by: List[str] = None,
        max_results: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Search for runs in the experiment

        Args:
            filter_string: Filter string (e.g., "metrics.accuracy > 0.9")
            order_by: List of order by clauses
            max_results: Maximum number of results

        Returns:
            List of matching runs
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            order_by=order_by,
            max_results=max_results
        )

        return runs.to_dict('records')

    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple runs

        Args:
            run_ids: List of run IDs to compare

        Returns:
            Comparison data
        """
        client = mlflow.tracking.MlflowClient()

        comparison = {
            'runs': [],
            'params': {},
            'metrics': {}
        }

        for run_id in run_ids:
            run = client.get_run(run_id)

            comparison['runs'].append({
                'run_id': run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'unnamed'),
                'status': run.info.status,
                'start_time': run.info.start_time
            })

            # Collect params
            for key, value in run.data.params.items():
                if key not in comparison['params']:
                    comparison['params'][key] = {}
                comparison['params'][key][run_id] = value

            # Collect metrics
            for key, value in run.data.metrics.items():
                if key not in comparison['metrics']:
                    comparison['metrics'][key] = {}
                comparison['metrics'][key][run_id] = value

        return comparison

    def get_best_run(self, metric: str, direction: str = 'max') -> Dict[str, Any]:
        """
        Get the best run based on a metric

        Args:
            metric: Metric name
            direction: 'max' or 'min'

        Returns:
            Best run information
        """
        order_by = [f"metrics.{metric} {'DESC' if direction == 'max' else 'ASC'}"]

        runs = self.search_runs(order_by=order_by, max_results=1)

        if not runs:
            return None

        return runs[0]

    def delete_run(self, run_id: str):
        """Delete a run"""
        client = mlflow.tracking.MlflowClient()
        client.delete_run(run_id)

    def get_model_versions(self, name: str) -> List[Dict[str, Any]]:
        """
        Get all versions of a registered model

        Args:
            name: Model name

        Returns:
            List of model versions
        """
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{name}'")

        return [
            {
                'name': v.name,
                'version': v.version,
                'stage': v.current_stage,
                'creation_timestamp': v.creation_timestamp,
                'last_updated_timestamp': v.last_updated_timestamp,
                'description': v.description
            }
            for v in versions
        ]


# CLI interface
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MLflow Tracker')
    parser.add_argument('command', choices=['ui', 'list-experiments', 'list-runs', 'compare'],
                        help='Command to execute')
    parser.add_argument('--experiment', type=str, help='Experiment name')
    parser.add_argument('--port', type=int, default=5000, help='UI port')
    parser.add_argument('--run-ids', type=str, nargs='+', help='Run IDs for comparison')

    args = parser.parse_args()

    if args.command == 'ui':
        # Start MLflow UI
        import subprocess
        subprocess.run(['mlflow', 'ui', '--port', str(args.port)])

    elif args.command == 'list-experiments':
        import mlflow
        experiments = mlflow.search_experiments()
        for exp in experiments:
            print(f"\nExperiment: {exp.name}")
            print(f"  ID: {exp.experiment_id}")
            print(f"  Artifact Location: {exp.artifact_location}")

    elif args.command == 'list-runs':
        if not args.experiment:
            print("Error: --experiment required")
            exit(1)

        tracker = MLflowTracker(experiment_name=args.experiment)
        runs = tracker.search_runs(max_results=10)

        for run in runs:
            print(f"\nRun ID: {run.get('run_id')}")
            print(f"  Status: {run.get('status')}")
            print(f"  Metrics: {run.get('metrics', {})}")

    elif args.command == 'compare':
        if not args.experiment or not args.run_ids:
            print("Error: --experiment and --run-ids required")
            exit(1)

        tracker = MLflowTracker(experiment_name=args.experiment)
        comparison = tracker.compare_runs(args.run_ids)

        print("\nRun Comparison:")
        print(json.dumps(comparison, indent=2))
