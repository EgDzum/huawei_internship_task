# huawei_internship_task
## Technical track: LLM Training Acceleration

For this track the Qwen model has been fine-tuned with AdamW, Moun and AdamW + Moun optimizers. All the experiments are stored in comparison_notebook/experiments-logs.ipynb file and can be seen in cells output. 

To dive into fine-tuning implementation, please, see the src/directory where you can find
- config.py - code for storing necessary hyperparameters 
- pipeline.py - code for fine-tuning pipeline implementation
- contants.py - file with all global constants

The models' weights are stored in model_weights/ directory. Feel free to choose model with necessary optimizer!

## Execution
To execute the pipeline, execute main.py file:

```python
python3 main.py
```

To reproduce the results just execute the cells with code from comparison_notebook/experiments-logs.ipynb.

## Evaluation
To evaluate fine-tuned models execute `eval.sh` script:

```bash
bash eval.sh
```

## Results 

The results of all experiments are described in **qwen_report.pdf**.
