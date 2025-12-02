import pandas as pd 
import numpy as np
from copy import deepcopy
from scipy.spatial.distance import pdist, squareform
import requests
import json

def matrix_prod(data):  # add item_view later
    rows = []
    for entry in data:
        user = entry["user_id"]              # string (UUID)
        liked_raw = entry.get("liked_items") or []    
        viewed_raw = entry.get("viewed_items") or []

        liked = set(liked_raw) 
        viewed = set(viewed_raw) 

        all_items = liked.union(viewed)
        for item in all_items:
            if item in liked:
                val = 2
            else:
                val = 1
            rows.append({"user_id": user, "liked_items": item, "value": val})
    
    df = pd.DataFrame(rows)

    matrix = df.pivot_table(
        index="liked_items",
        columns="user_id",
        values="value",
        fill_value=0,
    )

    matrix = matrix.fillna(0).astype(int)

    distance_mtx = squareform(pdist(matrix, 'cosine'))
    similarity_mtx = 1 - distance_mtx

    return similarity_mtx, matrix 

def calculate_user_rating(user_id, similarity_mtx, utility):
    # user_id is now a string column label (UUID)
    user_rating = utility[user_id]
    pred_rating = deepcopy(user_rating)

    default_rating = user_rating[user_rating > 0].mean()
    numerate = np.dot(similarity_mtx, user_rating)
    corr_sim = similarity_mtx[:, user_rating > 0]

    for i in range(len(pred_rating)):
        val = pred_rating.iat[i]  
        if val < 1:
            w_r = numerate[i]
            sum_w = corr_sim[i, :].sum()
            if w_r == 0 or sum_w == 0:
                temp = default_rating
            else:
                temp = w_r / sum_w
            pred_rating.iat[i] = temp

    return pred_rating

def recommendation_unseen(utility, user_id, top_n, pred_rating):
    user_rating = utility[user_id]
    unseen = pred_rating[user_rating == 0]
    top_items = unseen.sort_values(ascending=False).head(top_n)
    return top_items.to_dict()

if __name__ == "__main__":
    r = requests.get("http://api.dooleyonline.net/user/interactions", timeout=10)
    r.raise_for_status()
    data = r.json()

    similar_matrix, matrix = matrix_prod(data)

    res = {}
    
    for user in data:
        # print(similar_matrix)
        target_user_id = user["user_id"]
        prediction = calculate_user_rating(target_user_id, similar_matrix, matrix)
        rec_dict = recommendation_unseen(matrix, target_user_id, 10, prediction)
        res[target_user_id] = list(rec_dict.keys())
        print(json.dumps(res, indent=2))
        
    
