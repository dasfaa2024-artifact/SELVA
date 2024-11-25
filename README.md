# SELVA: A Reliable and Fast Selectivity Estimation Method for Query Plan Optimazition in Video Analytics
## Installation
1. Clone this repository
```shell
git clone https://github.com/lebenslange-lernen/SELVA.git
```
2. install requirements:
```shell
sudo apt update
sudo apt install python3.12
cd SELVA
python -m pip install -r requirements.txt
```
## Run experiment
1. Prepare the data ready for selectivity estimation. i.e., the predicate-evaluated transfromation algorithm output.
2. save it to a pickle file.
3. run `experiments/paras.py`:
```shell
python params.py [-h] [--run {veps,vdelt,vsel,probtrunc,probsn,cv,cmp}] --inpath INPATH [--repeat REPEAT]

options:
  -h, --help            show this help message and exit
  --run {veps,vdelt,vsel,probtrunc,probsn,cv,cmp}, -r {veps,vdelt,vsel,probtrunc,probsn,cv,cmp}
  --inpath INPATH, -i INPATH
                        The path of file containing results of evaluated predicates.
  --repeat REPEAT, -p REPEAT
                        The number of times to run the experiments.
  ```
