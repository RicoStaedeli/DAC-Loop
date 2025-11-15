# R-L Net
bla bla

## Generate Plots
To generate plots for R-L Net, you can use the following code snippet:
Make sure you are in the **root directory** of the project and have the necessary dependencies installed.

```bash
uv run python -m framework.utils.generate_plots
```

```bash
uv run python -m framework.utils.plots.markdown_tables
```

## Baselines
In this section we provide baseline results for various tasks using the R-L Net framework.

### Baselines for Teacher-Student Learning
The baseline for teacher-student learning are the models trained from scratch without any teacher supervision.

**Results for Teacher-Student Learning on the EuroSAT dataset:**

| run_name | experiment_name | run_id | model    | dataset              | augmentation | indices_config                                                               | epochs | learning_rate | batch_size | weight_decay | scheduler | seed | best_epoch | best_train_loss | best_val_loss | best_train_accuracy (%) | best_val_accuracy (%) | best_val_MSE |
|----------|-----------------|--------|----------|----------------------|--------------|------------------------------------------------------------------------------|--------|---------------|------------|--------------|-----------|------|------------|-----------------|---------------|-------------------------|-----------------------|--------------|
| Run_1    | Experiment_1    | 1      | ResNet18 | blanchon/EuroSAT_MSI | False        | calc=[] - bands=[B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12] | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 50         | 0.000511        | 0.078400      | 99.9938                 | 98.4444               | 0.002705     |
| Run_2    | Experiment_1    | 2      | ResNet18 | blanchon/EuroSAT_MSI | False        | calc=[] - bands=[B02, B03, B04]                                              | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 42         | 0.000542        | 0.152066      | 100.0000                | 96.4630               | 0.005916     |
| Run_3    | Experiment_1    | 3      | ResNet18 | blanchon/EuroSAT_MSI | False        | calc=[NDVI, NDWI, NDMI] - bands=[B02, B03, B04, B07, B08]                    | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 43         | 0.000403        | 0.105652      | 100.0000                | 97.7593               | 0.003761     |
| Run_4    | Experiment_1    | 4      | ResNet10 | blanchon/EuroSAT_MSI | False        | calc=[] - bands=[B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12] | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 47         | 0.000747        | 0.075454      | 99.9815                 | 98.3519               | 0.002943     |
| Run_5    | Experiment_1    | 5      | ResNet10 | blanchon/EuroSAT_MSI | False        | calc=[] - bands=[B02, B03, B04]                                              | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 40         | 0.000834        | 0.184022      | 100.0000                | 95.4074               | 0.007689     |
| Run_6    | Experiment_1    | 6      | ResNet10 | blanchon/EuroSAT_MSI | False        | calc=[NDVI, NDWI, NDMI] - bands=[B02, B03, B04, B07, B08]                    | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 50         | 0.000367        | 0.089332      | 100.0000                | 97.5185               | 0.003900     |
| Run_7    | Experiment_1    | 7      | TinyCNN  | blanchon/EuroSAT_MSI | False        | calc=[] - bands=[B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12] | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 38         | 0.326540        | 0.342012      | 89.3210                 | 89.2222               | 0.016526     |
| Run_8    | Experiment_1    | 8      | TinyCNN  | blanchon/EuroSAT_MSI | False        | calc=[] - bands=[B02, B03, B04]                                              | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 43         | 0.655089        | 0.669697      | 77.1975                 | 76.7963               | 0.032936     |
| Run_9    | Experiment_1    | 9      | TinyCNN  | blanchon/EuroSAT_MSI | False        | calc=[NDVI, NDWI, NDMI] - bands=[B02, B03, B04, B07, B08]                    | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 43         | 0.378075        | 0.382251      | 87.4691                 | 87.8704               | 0.018549     |
| Run_10   | Experiment_1    | 10     | ResNet50 | blanchon/EuroSAT_MSI | False        | calc=[] - bands=[B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12] | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 43         | 0.001302        | 0.073089      | 99.9815                 | 98.5000               | 0.002550     |
| Run_11   | Experiment_1    | 11     | ResNet50 | blanchon/EuroSAT_MSI | False        | calc=[] - bands=[B02, B03, B04]                                              | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 46         | 0.000363        | 0.121315      | 99.9938                 | 97.3148               | 0.004474     |
| Run_12   | Experiment_1    | 12     | ResNet50 | blanchon/EuroSAT_MSI | False        | calc=[NDVI, NDWI, NDMI] - bands=[B02, B03, B04, B07, B08]                    | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 44         | 0.000345        | 0.081593      | 100.0000                | 98.3148               | 0.002873     |

**Results for Teacher-Student Learning on the RESISC45 dataset:**

| run_name | experiment_name | run_id | model    | dataset       | augmentation | indices_config                  | epochs | learning_rate | batch_size | weight_decay | scheduler | seed | best_epoch | best_train_loss | best_val_loss | best_train_accuracy (%) | best_val_accuracy (%) | best_val_MSE |
|----------|-----------------|--------|----------|---------------|--------------|---------------------------------|--------|---------------|------------|--------------|-----------|------|------------|-----------------|---------------|-------------------------|-----------------------|--------------|
| Run_1    | Experiment_5    | 1      | ResNet18 | timm/resisc45 | True         | calc=[] - bands=[B01, B02, B03] | 50     | 0.0003        | 128        | 1e-05        | cosine    | 42   | 47         | 0.312045        | 0.285836      | 90.8042                 | 91.3333               | 0.002906     |
| Run_2    | Experiment_5    | 2      | ResNet10 | timm/resisc45 | True         | calc=[] - bands=[B01, B02, B03] | 50     | 0.0003        | 128        | 1e-05        | cosine    | 42   | 49         | 0.522177        | 0.403941      | 84.6825                 | 88.3175               | 0.004024     |
| Run_3    | Experiment_5    | 3      | TinyCNN  | timm/resisc45 | True         | calc=[] - bands=[B01, B02, B03] | 50     | 0.0003        | 128        | 1e-05        | cosine    | 42   | 48         | 2.859013        | 2.551090      | 22.7037                 | 29.9206               | 0.018245     |
| Run_4    | Experiment_5    | 4      | ResNet50 | timm/resisc45 | True         | calc=[] - bands=[B01, B02, B03] | 50     | 0.0003        | 128        | 1e-05        | cosine    | 42   | 50         | 0.193962        | 0.297681      | 93.6561                 | 90.8889               | 0.002981     |
| Run_5    | Experiment_5    | 5      | ResNet50 | timm/resisc45 | True         | calc=[] - bands=[B01, B02, B03] | 100    | 0.003         | 128        | 1e-05        | cosine    | 42   | 83         | 0.079817        | 0.715278      | 97.2275                 | 89.3016               | 0.003688     |
| Run_6    | Experiment_5    | 6      | ResNet50 | timm/resisc45 | True         | calc=[] - bands=[B01, B02, B03] | 50     | 0.003         | 256        | 1e-05        | cosine    | 42   | 49         | 0.517164        | 1.220733      | 83.5767                 | 81.6190               | 0.005890     |
| Run_7    | Experiment_5    | 7      | ResNet50 | timm/resisc45 | True         | calc=[] - bands=[B01, B02, B03] | 100    | 0.0003        | 128        | 1e-05        | cosine    | 42   | 93         | 0.056677        | 0.283054      | 98.2275                 | 92.6825               | 0.002625     |
| Run_8    | Experiment_5    | 8      | ResNet18 | timm/resisc45 | False        | calc=[] - bands=[B01, B02, B03] | 50     | 0.0003        | 128        | 1e-05        | cosine    | 42   | 49         | 0.006621        | 0.345476      | 99.9947                 | 89.9683               | 0.003377     |
