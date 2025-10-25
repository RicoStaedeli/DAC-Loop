from pathlib import Path

# Create markdown table content for runs 11–24
table_md = """| Run  | Date       | Author | Config                                           | Model Student | Model Teacher | KD-Type | Dataset                                   | Hardware   | Train Time | Metrics                                             | Plots                                | Bands                                                        | Notes/Issues | Next Steps |
|------|------------|--------|--------------------------------------------------|---------------|---------------|---------|-------------------------------------------|------------|------------|----------------------------------------------------|--------------------------------------|--------------------------------------------------------------|--------------|------------|
"""

for i in range(1, 25):
    table_md += f'| [{i}] | 2025-10-25 | Rico   | [Config.toml](Experiment_3/Run_{i}/config.toml)  | TinyCNN       | ResNet18      | Train/Val/Test split of blanchon/EuroSAT_MSI | CosineEmbeddingLoss | DGX-7 Rico |            | [metrics.json](Experiment_3/Run_{i}/metrics.jsonl) | [plots](Experiment_3/Run_{i}/plots)  | ["NDVI", "NDWI", "NDMI"]["B02", "B03", "B04", "B07", "B08"]  |              |            |\n'

# Print the markdown table to console
print(table_md)


