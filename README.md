# huawei_internship_task
## Technical track: LLM Training Acceleration

For this track the Qwen model has been fine-tuned with AdamW, Moun and AdamW + Moun optimizers. All the experiments are stored in comparison_notebook/experiments.ipynb file and can be seen in cells output. To reproduce the results just execute the cells with code again. 

To dive into fine-tuning implementation, please, see the src/ directory where you can find
- config.py - code for storing necessary hyperparameters 
- pipeline.py - code for fine-tuning pipeline implementation
- contants.py - file with all global constants

To execute the pipeline, execute main.py file 

`
python3 main.py
`