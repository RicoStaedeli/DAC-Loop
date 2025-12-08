## Generate Plots

To generate plots for R-L Net, you can use the following code snippet:
Make sure you are in the **root directory** of the project and have the necessary dependencies installed.
All generation scripts are hardcoded and have to be adjusted if new experiments are done.


Generate the plots with:

```bash
uv run python -m framework.utils.evaluation.generate_plots
```

Create the experiment summary markdown tables with:

```bash
uv run python -m framework.utils.evaluation.generate_exp_results_md
```

Create the Experiment Description Table with:

```bash
uv run python -m framework.utils.evaluation.generate_exp_description_md
```

Make Inference comparison

```bash
uv run python -m framework.utils.evaluation.inference
```
