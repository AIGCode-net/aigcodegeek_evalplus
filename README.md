# Evaluation for HumanEval(+) and MBPP(+)

## Update
2024-04-30: we have released our first finetuned 6.7B chat model [aigcode/AIGCodeGeek-DS-6.7B](https://huggingface.co/aigcode/AIGCodeGeek-DS-6.7B) and provided code for EvalPlus evaluation in this repo.

----

This repository contains scripts and code to run [EvalPlus](https://github.com/evalplus/evalplus) benchmarks on AIGCodeGeek series models. 

EvalPlus includes HumanEval(+) and MBPP(+) datasets for evaluating code completion preformance.

## 1. Generations and Results
`generations/` stores the generation files and `results/` gives the corresponding eval results:
```bash
generations
├── humaneval
│   └── aigcode--AIGCodeGeek-DS-6.7B_temp_0.0
├── mbpp
    └── aigcode--AIGCodeGeek-DS-6.7B_temp_0.0
```
```bash
results
├── humaneval
│   └── aigcodev1_temp_0.0.txt
└── mbpp
    └── aigcodev1_temp_0.0.txt
```

Both Base and Plus results are reported:
<table style="text-align:center;">
    <tr style="font-weight:bold">
        <td style="text-align: left">Model</td>
        <td>HumanEval</td>
        <td>HumanEval+</td>
        <td>MBPP</td>
        <td>MBPP+</td>
    </tr>
    <tr>
        <td style="text-align: left"><b>AIGCodeGeek-DS-6.7B</b></td>
        <td>82.3</td><td>76.2</td><td>77.7</td><td>64.4</td>
    </tr>
</table>

## 2. Environment

```shell
pip install evalplus --upgrade
pip install -r requirements.txt
```

## 3. Evalutaion
The evaluation is done with a single A100 GPU. Here are scripts:
```bash
bash run_humaneval.sh
bash run_mbpp.sh
```
