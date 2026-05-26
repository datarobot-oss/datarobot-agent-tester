---
name: datarobot-predictions
description: Tools and guidance for making predictions with DataRobot deployments, including real-time predictions, batch scoring, prediction dataset generation, and prediction explanations (SHAP/XEMP). Use when making predictions, running batch scoring, generating prediction datasets, or explaining individual predictions from a deployment.
---

# DataRobot Predictions Skill

This skill provides comprehensive guidance for working with DataRobot predictions, including real-time predictions, batch scoring, and generating prediction datasets.

## Quick Start

**Most common use case**: Generate predictions for a deployment

1. **Get deployment features**: `get_deployment_features(deployment_id)` to understand required columns
2. **Generate template**: `generate_prediction_data_template(deployment_id, n_rows)` to create CSV structure
3. **Make predictions**: Use `deployment.predict_batch(...)` (works for both single-row “real-time” and batch scoring)

**Example**: "Generate a prediction dataset template for deployment abc123 with 10 rows"

**To also explain predictions**: pass `--max-explanations N` to `make_prediction.py` (or the
`max_explanations=N` kwarg in code). See [Prediction Explanations](#prediction-explanations) below.

## When to use this skill

Use this skill when you need to:
- Make predictions from deployed DataRobot models
- Explain individual predictions from a deployment (SHAP or XEMP, per-row)
- Generate prediction dataset templates
- Validate prediction data before scoring
- Understand deployment feature requirements
- Perform batch predictions on large datasets
- Get sample training data to understand expected formats

> For post-hoc explanations against a **training project / leaderboard model** (not a deployment),
> use the `datarobot-model-explainability` skill instead. This skill covers deployment-time
> explanations returned alongside scoring.

## Key capabilities

### 1. Understanding Deployment Requirements

Before making predictions, you need to understand what features a deployment requires:

- **Feature names and types**: Know which columns are needed (numeric, categorical, text, date)
- **Feature importance**: Understand which features matter most
- **Target information**: Know what you're predicting
- **Time series configuration**: If applicable, understand datetime columns and series IDs

### 2. Generating Prediction Datasets

Create properly formatted prediction datasets:

- Generate CSV templates with all required columns
- Include sample values appropriate for each feature type
- Add metadata comments explaining the model structure
- Ensure correct column ordering

### 3. Validating Prediction Data

Validate datasets before making predictions:

## Prediction Explanations

Deployments can return per-row explanations (top feature contributions) alongside predictions.
Two algorithms are available depending on how the deployment was configured:

- **SHAP** (`shap`): SHapley Additive exPlanations. Available on tree-based models when SHAP was
  enabled at deployment time. Returns signed contributions in the model's score space.
- **XEMP** (`xemp`): DataRobot's eXplainable AI for the eXact Model Prediction. Default when SHAP
  is not enabled. Returns top-N strongest features with a qualitative strength (`+++`, `--`, etc.).

If you omit `explanation_algorithm`, the deployment's default is used.



## Best practices

1. **Always validate first**: Validate prediction data before submitting predictions to catch errors early
2. **Use templates**: Generate templates to ensure correct structure and avoid missing columns
3. **Check feature types**: Ensure numeric features are numbers, categorical features match training values
4. **Handle time series**: For time series models, ensure datetime columns and series IDs are properly formatted
5. **Monitor batch jobs**: For large batch predictions, check job status and handle errors appropriately

## Common patterns

### Pattern 1: Get deployment features and make single prediction (optionally with explanations)
```python
import datarobot as dr
import os
import pandas as pd
from datarobot_predict.deployment import predict as dr_predict

# Initialize client
dr.Client(
    token=os.getenv("DATAROBOT_API_TOKEN"),
    endpoint=os.getenv("DATAROBOT_ENDPOINT"),
)

deployment = dr.Deployment.get("abc123")

prediction_data = {
    "feature1": value1,
    "feature2": value2,
    # ... all required features (excluding target)
}

# Score one row.
result = dr_predict(
    deployment=deployment,
    data_frame=pd.DataFrame([prediction_data]),
)
print(result.dataframe.to_dict(orient="records"))
```



## Resources

- [DataRobot Python SDK Documentation](https://datarobot-public-api-client.readthedocs-hosted.com/)
- [DataRobot Predictions Documentation](https://docs.datarobot.com/en/docs/predictions/index.html)
- [DataRobot API Reference](https://docs.datarobot.com/en/docs/api/reference/index.html)
- [Batch Predictions Guide](https://docs.datarobot.com/en/docs/api/reference/sdk/batch-predictions.html)

