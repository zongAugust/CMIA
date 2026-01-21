import pandas as pd
from matplotlib import cm
import torch
import numpy as np
from torch import nn, softmax
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.spatial import distance
import random
import csv
import matplotlib.pyplot as plt
import argparse
import os
import pickle
from torch.utils.data import random_split
from torch.nn.init import xavier_normal_
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch.optim as optim
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from collections import defaultdict
from sklearn.metrics import roc_curve
import logging

def parse_args():
    parser = argparse.ArgumentParser(description="Run Attack Model")
    parser.add_argument("--topk", type=int, default=25, help="Number of top-k recommendations to consider")
    parser.add_argument("--cnum", type=int, default=3, help="Number of clusters to consider")
    parser.add_argument("--model", type=str, default="LightGCN", help="Recommendation model to use")
    parser.add_argument("--dataset", type=str, default="yelp2018", help="Dataset name")
    parser.add_argument("--epoch_num", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA")
    parser.add_argument("--cuda_device", type=int, default=7, help="CUDA device ID")
    parser.add_argument("--hidden_dims", type=int, nargs='*', default=[128, 128], help="MLP hidden dims")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate")

    return parser.parse_args()

args = parse_args()

log_path = f'./mia/logs/cmia/{args.dataset}/{args.model}_{args.dataset}_{args.topk}_{args.cnum}_epoch{args.epoch_num}.log'
os.makedirs(os.path.dirname(log_path), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode='w'),
        logging.StreamHandler()
    ]
)

logging.info("Logging initialized. Output file: %s", os.path.abspath(log_path))

def get_device(use_cuda, cuda_device):
    if use_cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{cuda_device}')
        print(f'Using CUDA device: {cuda_device}')
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 128], dropout_rate=0.01):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim

        # Create hidden layers
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        # Final output layer
        self.output_layer = nn.Linear(prev_dim, 2) 
        self.dropout = nn.Dropout(dropout_rate)
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize layers with Xavier normal distribution
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        for _, layer in enumerate(self.layers):
            x = F.relu(layer(x))
            x = self.dropout(x)  # Apply dropout after each hidden layer
        x = self.output_layer(x)
        return x
def load_interactions_with_labels(file_path):
    interactions = {}
    with open(file_path, 'r') as f:
        for i in range(5):
            print(f"Line {i + 1}: {f.readline().strip()}")
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')

            # Process each line
            user_id = int(parts[0])  # userID
            item_id = int(parts[1])  # artistID
            member_label = int(parts[2])  # member_label

            if user_id not in interactions:
                interactions[user_id] = {'member': [], 'nonmember': []}

            if member_label == 1:
                interactions[user_id]['member'].append(item_id)
            else:
                interactions[user_id]['nonmember'].append(item_id)
    return interactions

# Load recommendation lists for users
def load_recommendations(file_path):
    recommendations = {}
    with open(file_path, 'r') as file:
        for line in file:
            user_id, item_id = map(int, line.strip().split('\t')[:2])
            if user_id not in recommendations:
                recommendations[user_id] = []
            recommendations[user_id].append(item_id)
    return recommendations


def load_embeddings(model_path,model='LightGCN'):
    if model == 'LightGCN':
        checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
        #user_embeddings = checkpoint['embedding_user.weight'].cpu().numpy()
        item_embeddings = checkpoint['embedding_item.weight'].cpu().numpy()
        logging.info(f"Loaded LightGCN embeddings:{item_embeddings.shape[0]} items")
        return item_embeddings
    elif model == 'NGCF':
        checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
        item_embeddings = checkpoint['embedding_dict.item_emb'].cpu().numpy()
        logging.info(f"Loaded NGCF embeddings: {item_embeddings.shape[0]} items")
        return item_embeddings
    elif model == 'lfm':
        with open(model_path, "rb") as f:
            embeddings = pickle.load(f)
        item_embeddings = embeddings["item_embedding"].T 
        logging.info(f"Loaded lfm item embeddings: {item_embeddings.shape[0]} items")
        return item_embeddings

def save_attack_data_to_pkl(data, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    logging.info(f"Attack data saved to {save_path}")

def load_attack_data(load_path):
    try:
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        logging.info(f"Attack data loaded from {load_path}")
        return data
    except FileNotFoundError:
        logging.info(f"File not found at {load_path}")
        return None

def cosine_kmeans(embeddings, k=2, max_iter=500):
    
    embeddings = normalize(embeddings, axis=1)

    np.random.seed(42)
    indices = np.random.choice(len(embeddings), k, replace=False)
    centroids = embeddings[indices]

    for _ in range(max_iter):
        similarity = np.dot(embeddings, centroids.T)  # shape: (n_samples, k)
        labels = np.argmax(similarity, axis=1)

        new_centroids = []
        for i in range(k):
            cluster_points = embeddings[labels == i]
            if len(cluster_points) == 0:
                new_centroid = embeddings[np.random.choice(len(embeddings))]
            else:
                new_centroid = normalize(np.mean(cluster_points, axis=0, keepdims=True))[0]
            new_centroids.append(new_centroid)

        new_centroids = np.array(new_centroids)
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids

    return labels, centroids

def cluster_user_interest(user_train_dict, item_embeddings, k=2):
    user_cluster_info = {}
    user_cluster_labels = {}
    for user, items in user_train_dict.items():
        embeddings = [item_embeddings[i] for i in items if i < len(item_embeddings)]
        embeddings = np.stack(embeddings).astype(np.float32)
        labels, centroids = cosine_kmeans(embeddings, k=k)
        label_counts = Counter(labels)
        main_cluster = label_counts.most_common(1)[0][0]

        user_cluster_info[user] = {
            'labels': labels,
            'main_cluster': main_cluster,
            'centroids': centroids
        }
        user_cluster_labels[user] = labels.tolist()
    return user_cluster_info, user_cluster_labels

def aggregate_cluster(user_cluster_info, recommend_list):
    user_recommend_vectors = {}

    for user, _ in recommend_list.items():
        if user not in user_cluster_info:
            continue
        centroids = user_cluster_info[user]['centroids']
        final_vec = np.mean(centroids, axis=0)
        user_recommend_vectors[user] = final_vec

    return user_recommend_vectors


def prepare_attacktrain_data(
    interactions,
    recommendations,
    embeddings,
    user_recommend_vectors,
    user_cluster_info, 
    k=25,
    n=3,
    save_path=None,
    load_path=None
):
    if load_path:
        return load_attack_data(load_path)

    attack_data = []

    for user_id, user_interactions in interactions.items():
        if (user_id not in recommendations 
            or user_id not in user_recommend_vectors 
            or user_id not in user_cluster_info):
            logging.info(f"Warning: user_id {user_id} missing in recommendations or vectors.")
            continue

        final_user_vec = user_recommend_vectors[user_id].astype(np.float32)
        user_norm = np.linalg.norm(final_user_vec)
        if user_norm > 0:
            final_user_vec = final_user_vec / user_norm

        cluster_centers = user_cluster_info[user_id]['centroids'].astype(np.float32)
        cluster_centers = normalize(cluster_centers)
        def get_similarity_vector(item_vec):
            sim_vector = np.dot(cluster_centers, item_vec)  # shape: (k,)
            sim_vector = sim_vector 
            return sim_vector
        for _, items, label in [('member', user_interactions['member'], 1),
                                         ('nonmember', user_interactions['nonmember'], 0)]:
            for item_id in items:
                if 0 <= item_id < len(embeddings):
                    item_vec = embeddings[item_id].astype(np.float32)
                    item_norm = np.linalg.norm(item_vec)
                    if item_norm > 0:
                        item_vec = item_vec / item_norm

                    sim_vector = get_similarity_vector(item_vec)  # shape: (k,)
                    final_feature = np.concatenate([final_user_vec, item_vec,sim_vector])
                    attack_data.append((user_id, item_id, final_feature, label))
    if save_path:
        save_attack_data_to_pkl(attack_data, save_path)

    return attack_data
def compute_tpr_at_fpr(labels, probs, target_fpr=0.01):
    fpr, tpr, _ = roc_curve(labels, probs)
    return np.interp(target_fpr, fpr, tpr)

def run_attack_model(attack_data, input_dim, batch_size=64, model=None, learning_rate=0.01,
                     device=None, epoch_num=100, validation_split=0.2, hidden_dims=[32, 8],
                     dropout_rate=0.1, model_name="lgn", dataset_name="gowalla"):
    if model is None:
        random.shuffle(attack_data)
        split_idx = int(len(attack_data) * (1 - validation_split))
        train_data = attack_data[:split_idx]
        val_data = attack_data[split_idx:]

        mlp = model if model else MLP(input_dim, hidden_dims, dropout_rate)
        mlp.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        train_auc_scores, val_auc_scores = [], []

        for epoch in range(epoch_num):
            # === 训练阶段 ===
            mlp.train()
            train_loss, train_correct = 0, 0
            all_train_labels, all_train_probs = [], []

            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i + batch_size]
                features = torch.stack([torch.tensor(data[2], dtype=torch.float32) for data in batch]).to(device)
                labels = torch.tensor([data[3] for data in batch], dtype=torch.long).to(device)

                optimizer.zero_grad()
                output = mlp(features)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                _, predicted = output.max(1)
                train_correct += (predicted == labels).sum().item()
                train_loss += loss.item()*labels.size(0)
                all_train_labels.extend(labels.cpu().numpy())
                all_train_probs.extend(torch.softmax(output, dim=-1).detach().cpu().numpy())

            train_auc = roc_auc_score(all_train_labels, np.array(all_train_probs)[:, 1])
            train_losses.append(train_loss / (len(train_data)))
            train_accuracies.append(train_correct / len(train_data))
            train_auc_scores.append(train_auc)

            # === val ===
            mlp.eval()
            val_loss, val_correct = 0, 0
            all_val_labels, all_val_probs = [], []

            with torch.no_grad():
                for i in range(0, len(val_data), batch_size):
                    batch = val_data[i:i + batch_size]
                    features = torch.stack([torch.tensor(data[2], dtype=torch.float32) for data in batch]).to(device)
                    labels = torch.tensor([data[3] for data in batch], dtype=torch.long).to(device)
                    output = mlp(features)
                    loss = criterion(output, labels)

                    _, predicted = output.max(1)
                    val_correct += (predicted == labels).sum().item()
                    val_loss += loss.item()* labels.size(0)
                    all_val_labels.extend(labels.cpu().numpy())
                    all_val_probs.extend(torch.softmax(output, dim=-1).detach().cpu().numpy())

            val_auc = roc_auc_score(all_val_labels, np.array(all_val_probs)[:, 1])
            val_losses.append(val_loss / (len(val_data)))
            val_accuracies.append(val_correct / len(val_data))
            val_auc_scores.append(val_auc)

            logging.info(
                f"Epoch {epoch + 1}/{epoch_num}, "
                f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, Train AUC: {train_auc_scores[-1]:.4f}, "
                f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}, Val AUC: {val_auc_scores[-1]:.4f}"
            )
        model_save_path = f"./mia/model/cmia/{dataset_name}/{model_name}_{dataset_name}_{args.topk}_{args.cnum}_epoch{epoch}.pth"
        torch.save(mlp.state_dict(), model_save_path)
        logging.info(f"Model saved to {model_save_path}")
        return mlp

    else:
        mlp = model
        mlp.to(device)
        mlp.eval()

        correct = 0
        all_labels, all_probs = [], []

        with torch.no_grad():
            for i in range(0, len(attack_data), batch_size):
                batch = attack_data[i:i + batch_size]
                features = torch.stack([torch.tensor(data[2], dtype=torch.float32) for data in batch]).to(device)
                labels = torch.tensor([data[3] for data in batch], dtype=torch.long).to(device)
                output= mlp(features)

                probabilities = torch.softmax(output, dim=-1)
                _, predicted = probabilities.max(1)

                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())

        accuracy = correct / len(attack_data)
        all_probs_array = np.array(all_probs)[:, 1]
        test_auc = roc_auc_score(all_labels, all_probs_array)
        tpr_at_01_fpr = compute_tpr_at_fpr(all_labels, all_probs_array, target_fpr=0.01)
        tpr_at_005_fpr = compute_tpr_at_fpr(all_labels, all_probs_array, target_fpr=0.005)
        logging.info(f"Test Accuracy: {accuracy:.4f}")
        logging.info(f"Test AUC: {test_auc:.4f}")
        logging.info(f"TPR@01%FPR: {tpr_at_01_fpr:.4f}")
        logging.info(f"TPR@005%FPR: {tpr_at_005_fpr:.4f}")

def main():
    logging.info("Start running main...")
    device = get_device(args.use_cuda, args.cuda_device)  
    logging.info("loading data...")
    train_data = load_interactions_with_labels(f"./data/{args.dataset}/attackdata/attacktrain.txt")
    testdata =  load_interactions_with_labels(f"./data/{args.dataset}/attackdata/attacktest.txt")
    if args.model == 'LightGCN':
        train_recommendations = load_recommendations(f"./recommender/{args.model}/code/recommendations/gcn1_{args.dataset}_shadow_recommendations.txt")
        test_recommendations = load_recommendations(f"./recommender/{args.model}/code/recommendations/gcn_{args.dataset}_target_recommendations.txt")
        embeddings = load_embeddings(f"./recommender/{args.model}/code/checkpoints/lgn1-{args.dataset}-0-3-64.pth.tar",args.model)
    elif args.model =='NGCF':
        train_recommendations = load_recommendations(f"./recommender/{args.model}/recommendations/ngcf_{args.dataset}_shadow_recommendations.txt")
        test_recommendations = load_recommendations(f"./recommender/{args.model}/recommendations/ngcf_{args.dataset}_target_recommendations.txt")
        embeddings = load_embeddings(f"./recommender/{args.model}/model/{args.dataset}/{args.dataset}_shadow.pkl",args.model)
    elif args.model =='lfm':
        train_recommendations = load_recommendations(f"./recommender/NGCF/{args.dataset}_shadow_recommendations.txt")
        test_recommendations = load_recommendations(f"./recommender/NGCF/{args.dataset}_target_recommendations.txt")
        embeddings = load_embeddings(f"./recommender/NGCF/{args.dataset}_shadow.pkl",args.model)

    for user in train_recommendations:
        train_recommendations[user] = train_recommendations[user][:args.topk]

    user_cluster_info, _ = cluster_user_interest(train_recommendations, embeddings, k=args.cnum)

    logging.info("Aggregating user vectors from recommendation lists...")
    user_recommend_vectors = aggregate_cluster(
        user_cluster_info=user_cluster_info,
        recommend_list=train_recommendations
    )
    
    for user in test_recommendations:
        test_recommendations[user] = test_recommendations[user][:args.topk]

    testuser_cluster_info, _ = cluster_user_interest(test_recommendations, embeddings, k=args.cnum)

    testuser_recommend_vectors = aggregate_cluster(
        user_cluster_info=testuser_cluster_info,
        recommend_list=test_recommendations
    )

    logging.info("Preparing attack train/test data...")
    train_attack_data_path = f'./cmia/{args.dataset}/{args.model}_{args.dataset}_{args.topk}_{args.cnum}.pkl'
    test_attack_data_path = f'./cmia/{args.dataset}/{args.model}test_{args.dataset}_{args.topk}_{args.cnum}.pkl'
    train_attack_data = prepare_attacktrain_data(
        interactions=train_data,
        recommendations=train_recommendations,
        embeddings=embeddings,
        user_recommend_vectors=user_recommend_vectors,
        user_cluster_info=user_cluster_info, 
        k=args.topk,
        n=args.cnum,
        save_path=None,
        load_path=train_attack_data_path
    )

    test_attack_data = prepare_attacktrain_data(
        interactions=testdata,
        recommendations=test_recommendations,
        embeddings=embeddings,
        user_recommend_vectors=testuser_recommend_vectors,
        user_cluster_info=testuser_cluster_info, 
        k=args.topk,
        n= args.cnum,
        save_path=None,
        load_path=test_attack_data_path
    )
    embedding_dim = embeddings.shape[1]
    input_dim = 2*embedding_dim+args.cnum
    logging.info("\n--- Training ---")
    trained_model = run_attack_model(train_attack_data,input_dim, batch_size=args.batch_size, model=None,
                                     learning_rate=args.learning_rate, device=device, epoch_num=args.epoch_num, 
                                     hidden_dims=args.hidden_dims, dropout_rate=args.dropout_rate,model_name=args.model,dataset_name=args.dataset)


    logging.info("\n--- Testing ---")
    run_attack_model(test_attack_data, input_dim, batch_size=args.batch_size, model=trained_model,learning_rate=args.learning_rate, device=device)
if __name__ == "__main__":
    main()