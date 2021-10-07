# 1 MRATE
  
This repo is the official implementation for [*Multi-Relation Aware Temporal Interaction Network Embedding*]()

## 1.1 The framework of MRATE
<div align=center>
<img src="https://user-images.githubusercontent.com" width="400" height="400" alt="framework"/><br/>
<div align=left>

# 2 Prerequisites
- numpy==1.14.6
- gpustat==0.5.0
- tqdm==4.32.1
- torch==0.4.1
- scikit_learn==0.19.1

# 3 Data Preparation
 ## 3.1 Datasets
Links to datasets used in the paper:
 - [Reddit](http://snap.stanford.edu/jodie/reddit.csv)
 - [LastFM](http://snap.stanford.edu/jodie/lastfm.csv)
 - [Wikipedia](http://snap.stanford.edu/jodie/wikipedia.csv)

 ## 3.2 Generating Data
Recent versions of PyTorch, numpy, sklearn, tqdm, and gpustat. You can install all the required packages using the following command:
```
    $ pip install -r requirements.txt
```

To initialize the directories needed to store data and outputs, use the following command. This will create `data/`, `saved_models/`, and `results/` directories.
```
    $ chmod +x initialize.sh
    $ ./initialize.sh
```

To download the datasets used in the paper, use the following command. This will download three datasets under the `data/` directory: `reddit.csv`, `lastfm.csv`, and `wikipedia.csv`.
```
    $ chmod +x download_data.sh
    $ ./download_data.sh
```


# 4 Running

## 4.1 Training
To train the MRATE model using the `data/<network>.csv` dataset, use the following command. This will save a model for every epoch in the `saved_models/<network>/` directory.
```
   $ python mrate.py --network <network> --model mrate --epochs 50
```
## 4.2 Testing
To evaluate the performance of the model for the interaction prediction task, use the following command. The command iteratively evaluates the performance for all epochs of the model and outputs the final test performance. 
```
    $ chmod +x evaluate_all_epochs.sh
    $ ./evaluate_all_epochs.sh <network> interaction
```


# 5 Acknowledgements
This repo is based on [JODIE](https://github.com/srijankr/jodie). Great thanks to the original authors for their work!


# 6 Citation

Please cite this work if you find it useful.

If you have any question, feel free to contact: `shshyu@zju.edu.cn`