# FaMOEL - Fairness-aware Multiobjective Evolutionary Learning

This is the code for the paper "Fairness-aware Multiobjective Evolutionary Learning" in IEEE Transactions on Evolutionary Computation. 
<!-- The PDF file is available at the IEEE website. -->

**Personal Use Only. No Commercial Use**

Please consider citing this work if you use this repository. The bibtex is as followes:

````
@ARTICLE{9902997,
  author={Zhang, Qingquan and Liu, Jialin and Yao, Xin},
  journal={IEEE Transactions on Evolutionary Computation}, 
  title={Fairness-aware Multiobjective Evolutionary Learning}, 
  year={2024},
  volume={},
  number={},
  pages={},
  doi={}}
````

### Main environments that have been tested
```
Python==3.9.0
numpy==1.26.3
scikit-learn==1.5.1
torch==2.3.1+cu121
geatpy==2.7.0
```



## How to use

#### Load/Add dataset

1. Data pre-processing is based on the [approach](https://github.com/algofairness/fairness-comparison).
   1. Put the raw data in `FairEMOL/EvolutionaryCodes/data/raw`
   2. Write the code to process raw data in `FairEMOL/EvolutionaryCodes/data/objects`, e.g., `German.py`
   3. Run the code `FairEMOL/EvolutionaryCodes/preprocess.py` 
   4. Obtain the processed data in `FairEMOL/EvolutionaryCodes/data/preprocessed`, e.g., `german_numerical-for-NN.csv`
   5. Restore processed data in the folder `FairEMOL/EvolutionaryCodes/data`, e.g., `German`
2. Use `FairEMOL/EvolutionaryCodes/load_data.py` to splite the data into training data, validation data, ensemble data (if you need) and test data. You can also set `save_csv=1` to manually store them.
3. Add the dataset's name in `FairEMOL/EvolutionaryCodes/data/objects/list.py`

#### Run algorithm
1. Set algorithmic parameters in `FaMOEL.py`
2. Run `FaMOEL.py`

