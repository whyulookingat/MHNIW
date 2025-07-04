# MHNIW

## Simulation Instructions

This guide explains how to run the simulation of **MHNIW** algorithm with configurable parameters.


## Structure

root/
│

├── train.py                        **# MHNIW algorithm**

├── mats.py                         **# HD-MATS algorithm**[1]

├── train_empirical.py              **# MH-Empirical algorithm**

##  Usage

Run the script using the command line:

*e.g.*

Firstly, install the prerequisites
```bash
pip install -r requirements.txt
```
Then, run the snippet
```bash
python train.py --data data/train.txt --agents 50 --rounds 4000
```
This means it will use 50 **agents** to train 4,000 **rounds** by **MHNIW** algorithm on train.txt
the dataset is available at [*Microsoft Research*](https://www.microsoft.com/en-us/research/project/mslr/)[2]



## References
[1] Verstraeten, T., Bargiacchi, E., Libin, P. J. K., Helsen, J., Roijers, D. M., & Nowé, A. (2023). Multi-agent Thompson sampling for bandit applications with sparse neighbourhood structures.arXiv preprint arXiv:2303.04567

[2] Qin, T., & Liu, T.-Y. (2013). *Introducing LETOR 4.0 Datasets*. arXiv preprint arXiv:1306.2597. http://arxiv.org/abs/1306.2597
