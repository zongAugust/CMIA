# Official Implementation of CMIA

**Interaction-level Membership Inference Attack for Recommender Systems via Cluster-based User Modeling**

Danyang Zong, Kaike Zhang, Qi Cao, Du Su and Fei Sun

Accepted at *The Web Conference (WWW) 2026* (Short Paper).

# Dataset Preparation

The datasets can be downloaded from the `data` folder of the official LightGCN repository: [LightGCN-PyTorch](https://github.com/gusye1234/LightGCN-PyTorch).

After downloading, preprocess the datasets using the Python scripts:

1. Run `splitdata.py` to split the data into shadow and target datasets.  
2. Re-index the user IDs in the target dataset starting from 0 by running `reidx.py`.  
3. Run `prepareattackdata.py` on the shadow and target datasets to generate the attack dataset and the test dataset, respectively.

# Training the Recommendation Models

- **LightGCN:** Follow the instructions in the official LightGCN repository: [LightGCN-PyTorch](https://github.com/gusye1234/LightGCN-PyTorch).  
- **NGCF:** Refer to the official NGCF repository for training: [Neural Graph Collaborative Filtering](https://github.com/xiangwang1223/neural_graph_collaborative_filtering).  
- **LFM:** For the LFM model, use the Python scripts under the `DL-MIA-RS` folder in this repository: [DL-MIA-KDD-2022](https://github.com/WZH-NLP/DL-MIA-KDD-2022/tree/main/DL-MIA-RS).

# Training and Evaluating the Attack Model

To train the attack model and evaluate its performance, run:

```bash
python cmia.py

