# R-L Net

bla bla

## Generate Plots

To generate plots for R-L Net, you can use the following code snippet:
Make sure you are in the **root directory** of the project and have the necessary dependencies installed.

Generate the plots with:

```bash
uv run python -m framework.utils.plots.generate_plots
```

Create the experiment summary markdown tables with:

```bash
uv run python -m framework.utils.plots.generate_exp_results_md
```

Create the Experiment Description Table with:

```bash
uv run python -m framework.utils.plots.generate_exp_description_md
```

## Baselines

In this section we provide baseline results for various tasks using the R-L Net framework.

### Baselines for Teacher-Student Learning

The baseline for teacher-student learning are the models trained from scratch without any teacher supervision.

**Results for Teacher-Student Learning on the EuroSAT dataset:**

| run_name | experiment_name | run_id | model    | dataset              | augmentation | indices_config                                                               | epochs | learning_rate | batch_size | weight_decay | scheduler | seed | best_epoch | best_train_loss | best_val_loss | best_train_accuracy (%) | best_val_accuracy (%) | best_val_MSE |
|----------|-----------------|--------|----------|----------------------|--------------|------------------------------------------------------------------------------|--------|---------------|------------|--------------|-----------|------|------------|-----------------|---------------|-------------------------|-----------------------|--------------|
| Run_1    | Experiment_1    | 1      | ResNet18 | blanchon/EuroSAT_MSI | False        | calc=[] - bands=[B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12] | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 47         | 0.000555        | 0.086449      | 99.9938                 | 98.2778               | 0.002988     |
| Run_2    | Experiment_1    | 2      | ResNet18 | blanchon/EuroSAT_MSI | False        | calc=[] - bands=[B02, B03, B04]                                              | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 46         | 0.000678        | 0.141871      | 99.9938                 | 96.8519               | 0.005382     |
| Run_3    | Experiment_1    | 3      | ResNet18 | blanchon/EuroSAT_MSI | False        | calc=[NDVI, NDWI, NDMI] - bands=[B02, B03, B04, B07, B08]                    | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 50         | 0.000349        | 0.095817      | 99.9938                 | 97.9444               | 0.003487     |
| Run_4    | Experiment_1    | 4      | ResNet10 | blanchon/EuroSAT_MSI | False        | calc=[] - bands=[B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12] | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 24         | 0.008954        | 0.068166      | 99.7407                 | 98.4074               | 0.002831     |
| Run_5    | Experiment_1    | 5      | ResNet10 | blanchon/EuroSAT_MSI | False        | calc=[] - bands=[B02, B03, B04]                                              | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 37         | 0.001027        | 0.182201      | 99.9815                 | 95.2963               | 0.007731     |
| Run_6    | Experiment_1    | 6      | ResNet10 | blanchon/EuroSAT_MSI | False        | calc=[NDVI, NDWI, NDMI] - bands=[B02, B03, B04, B07, B08]                    | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 47         | 0.000445        | 0.084619      | 100.0000                | 97.8333               | 0.003587     |
| Run_7    | Experiment_1    | 7      | TinyCNN  | blanchon/EuroSAT_MSI | False        | calc=[] - bands=[B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12] | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 45         | 0.320458        | 0.337920      | 89.3951                 | 89.2222               | 0.016315     |
| Run_8    | Experiment_1    | 8      | TinyCNN  | blanchon/EuroSAT_MSI | False        | calc=[] - bands=[B02, B03, B04]                                              | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 44         | 0.658462        | 0.674295      | 77.1667                 | 76.7222               | 0.033131     |
| Run_9    | Experiment_1    | 9      | TinyCNN  | blanchon/EuroSAT_MSI | False        | calc=[NDVI, NDWI, NDMI] - bands=[B02, B03, B04, B07, B08]                    | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 39         | 0.380586        | 0.381733      | 87.4259                 | 87.8704               | 0.018571     |
| Run_10   | Experiment_1    | 10     | ResNet50 | blanchon/EuroSAT_MSI | False        | calc=[] - bands=[B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12] | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 48         | 0.000205        | 0.171966      | 99.9938                 | 96.9630               | 0.005394     |
| Run_11   | Experiment_1    | 11     | ResNet50 | blanchon/EuroSAT_MSI | False        | calc=[] - bands=[B02, B03, B04]                                              | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 45         | 0.002114        | 0.512845      | 99.9753                 | 90.9074               | 0.015801     |
| Run_12   | Experiment_1    | 12     | ResNet50 | blanchon/EuroSAT_MSI | False        | calc=[NDVI, NDWI, NDMI] - bands=[B02, B03, B04, B07, B08]                    | 50     | 0.0003        | 256        | 1e-05        | cosine    | 42   | 43         | 0.000237        | 0.205801      | 100.0000                | 96.1296               | 0.006716     |

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

**Additional baselines are found through research:**

- They had around the same accuracy training ResNet18 from scratch on RESISC45 with similar
  hyperparameters. [Source](https://arxiv.org/pdf/2507.10381)

| Model                         | Parameters (M) | Dataset  | Accuracy Theirs(%) | Accuracy Ours Baselines(%) | Source                                                                     |
|-------------------------------|----------------|----------|--------------------|----------------------------|----------------------------------------------------------------------------|
| ResNet18                      |                | RESISC45 | 92.10              | 91.33                      | [Source](https://arxiv.org/pdf/2507.10381)                                 |
| ResNet18                      |                | EuroSAT  | 98.06              | 98.27                      | [Source](https://arxiv.org/pdf/2507.10381)                                 |
| ResNet50 - InDomain           |                | RESISC45 | 96.80              | 92.68                      | [Source](http://arxiv.org/abs/1911.06721)                                  |
| ResNet50 - InDomain           |                | EuroSAT  | 99.20              | 96.96                      | [Source](http://arxiv.org/abs/1911.06721)                                  |
| ResNet50                      |                | EuroSAT  | 98.57              | 96.96                      | [Source](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8736785) |
| EfficientNetB0                | 4.7            | RESISC45 | 92.3               |                            | [Source](https://arxiv.org/pdf/2302.13028)                                 |
| EfficientNetB0 + Distillation | 4.7            | RESISC45 | 94.8               |                            | [Source](https://arxiv.org/pdf/2302.13028)                                 |
| EfficientNetB0 + Attention    | 9.4            | RESISC45 | 93.8               |                            | [Source](https://arxiv.org/pdf/2211.02820)                                 |

## Pruning results

The following table summarizes the pruning results obtained using the R-L Net framework:

| Model    | Dataset       | Initial Accuracy (%) | Accuracy without KD-Aware Pruning | Pruned + Fine-tuned Accuracy (%) | Parameter Reduction to Basemodel (%) | Parameter Reduction to Teacher (%) | Trainable Parameters |
|----------|---------------|----------------------|-----------------------------------|----------------------------------|--------------------------------------|------------------------------------|----------------------|
| ResNet10 | timm/resisc45 | 90.98                | 90.41                             | **91.13**                        | 34.57%                               | 86.94%                             | 3,219,996            |
| ResNet10 | timm/resisc45 | 90.98                | 89.54                             | **90.10**                        | 63.09%                               | 92.36%                             | 1,816,587            |
| ResNet18 | timm/resisc45 | 92.98                | 90.51                             | **93.00**                        | 46.03%                               | 75.49%                             | 6,040,725            |
| ResNet18 | timm/resisc45 | 92.98                | 88.48                             | **91.06**                        | 74.99%                               | 88.65%                             | 2,798,648            |

KD-Aware Pruning consistently outperforms standard pruning methods, achieving higher accuracy with significant parameter
reductions.