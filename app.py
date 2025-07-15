# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Recommendation API is running!"

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    all_store_orders = data.get("all_store_orders", {})
    target_store = data.get("target_store")
    method = data.get("method", "user")
    top_n = int(data.get("top_n", 5))

    if not all_store_orders or target_store not in all_store_orders:
        return jsonify({"error": "Missing store data or target_store"}), 400

    all_products = sorted({p for o in all_store_orders.values() for p in o})
    matrix_data = []
    for store, orders in all_store_orders.items():
        row = [orders.get(prod, 0) for prod in all_products]
        matrix_data.append(row)

    user_product_matrix = pd.DataFrame(matrix_data, index=all_store_orders.keys(), columns=all_products)

    if method == "user":
        sim_df = pd.DataFrame(
            cosine_similarity(user_product_matrix),
            index=user_product_matrix.index,
            columns=user_product_matrix.index,
        )
        top_similar = sim_df[target_store].sort_values(ascending=False)[1:4]
        similar_orders = user_product_matrix.loc[top_similar.index]
        weighted_scores = similar_orders.T.dot(top_similar)
        already_ordered = user_product_matrix.loc[target_store]
        recommendations = weighted_scores[already_ordered == 0].sort_values(ascending=False).head(top_n)

    else:  # item-based
        item_sim_df = pd.DataFrame(
            cosine_similarity(user_product_matrix.T),
            index=user_product_matrix.columns,
            columns=user_product_matrix.columns,
        )
        store_orders = user_product_matrix.loc[target_store]
        recommendations = pd.Series(dtype='float64')
        for product, score in store_orders.items():
            if score > 0:
                recommendations = recommendations.add(item_sim_df[product] * score, fill_value=0)
        recommendations = recommendations[store_orders == 0].sort_values(ascending=False).head(top_n)

    return jsonify({"store": target_store, "recommendations": recommendations.index.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
