# Evaluation

To generate plots you can use the following code snippet:
Make sure you are in the **root directory** of the project and have the necessary dependencies installed.
All generation scripts are hardcoded and have to be adjusted if new experiments are done.

## Generate plots
Generate the plots with:

| Argument            | description                                       |
|---------------------|---------------------------------------------------|
| --type              | One of: teacher_student, compression, overview    |
| --experiment        | Name like Experiment_7                            |
| --run-id            | Run number. If omitted or 0 → processes all runs  |
| --num-runs          | Number of runs to process                         | 
| --use-default-batch | Executes the original built-in configuration list |

Generate plot for a complete experiment
```bash
uv run python -m framework.utils.evaluation.generate_plots. \
  --experiment Experiment_1 \
  --num-runs 12 \
  --type teacher_student
```

Generate plot for a specific TS Experiment
```bash
uv run python -m framework.utils.evaluation.generate_plots \
  --experiment Experiment_4 \
  --run-id 3 \
  --type teacher_student
```
Generate predefined plots
```bash
uv run python -m framework.utils.evaluation.generate_plots --use-default-batch
```


## Experiment Results
Create the experiment summary Markdown tables with:

```bash
uv run python -m framework.utils.evaluation.generate_exp_results_md
```

## Experiment Description
Create the Experiment Description Table with:

```bash
uv run python -m framework.utils.evaluation.generate_exp_description_md
```
$
## Inference Evaluation

Make Inference comparison

```bash
uv run python -m framework.utils.evaluation.inference
```
