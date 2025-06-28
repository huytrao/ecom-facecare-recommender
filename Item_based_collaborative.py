import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm # Thư viện để hiển thị thanh tiến trình

###### Step 1: Tải và Chuẩn Bị Dữ Liệu (Không đổi)

def load_and_prepare_data(filepath, min_ratings_threshold=5):
    """Tải dữ liệu, làm sạch và lọc người dùng có ít đánh giá."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file '{filepath}'. Vui lòng kiểm tra lại đường dẫn.")
        return None

    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    if 'product_name_x' in df.columns:
        df.rename(columns={'product_name_x': 'product_name'}, inplace=True)

    print(f"Số lượng đánh giá ban đầu: {len(df)}")
    print(f"Số lượng người dùng ban đầu: {df['user_id'].nunique()}")

    user_counts = df['user_id'].value_counts()
    print(f"\nNgưỡng số đánh giá tối thiểu để giữ lại người dùng: {int(min_ratings_threshold)}")

    active_users = user_counts[user_counts >= min_ratings_threshold].index
    df_filtered = df[df['user_id'].isin(active_users)]

    print(f"Số lượng đánh giá sau khi lọc người dùng: {len(df_filtered)}")
    print(f"Số lượng người dùng sau khi lọc: {df_filtered['user_id'].nunique()}")

    return df_filtered

###### Step 2: Chia Dữ Liệu thành Tập Train và Test (Không đổi)

def create_train_test_split(df, test_size=0.2):
    """Chia dữ liệu theo từng người dùng."""
    train_list = []
    test_list = []

    for user_id, group in tqdm(df.groupby('user_id'), desc="Đang chia Train/Test..."):
        test_part = group.sample(frac=test_size, random_state=42)
        train_part = group.drop(test_part.index)
        train_list.append(train_part)
        test_list.append(test_part)

    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)

    print("\n--- Kích Thước Dữ Liệu ---")
    print(f"Tập Train: {len(train_df)} đánh giá")
    print(f"Tập Test: {len(test_df)} đánh giá")

    return train_df, test_df

###### Step 3: Xây Dựng và "Huấn Luyện" Mô Hình ITEM-BASED

class ItemBasedCollaborativeFilteringModel:
    def __init__(self):
        self.item_similarity = None
        self.train_matrix = None
        self.user_mean_ratings = None

    def fit(self, train_df):
        """Xây dựng ma trận tương đồng ITEM-ITEM từ tập train."""
        print("\nĐang huấn luyện mô hình (xây dựng ma trận ITEM-ITEM)...")
        self.train_matrix = train_df.pivot_table(index='user_id', columns='product_name', values='rating')
        self.user_mean_ratings = self.train_matrix.mean(axis=1)

        matrix_norm = self.train_matrix.subtract(self.user_mean_ratings, axis='rows')
        self.item_similarity = matrix_norm.corr(method='pearson').fillna(0)

    def predict(self, user_id, product_name, n_similar_items=15):
        """Dự đoán rating của một user cho một sản phẩm dựa trên các item tương tự."""
        # Fallback to user's average rating if user or item is new
        if user_id not in self.user_mean_ratings.index or product_name not in self.item_similarity.columns:
            return self.user_mean_ratings.get(user_id, np.nan)

        similarities = self.item_similarity[product_name]
        user_ratings = self.train_matrix.loc[user_id].dropna()

        # Find items rated by the user that are also in the similarity list
        rated_similar_items = similarities.index.intersection(user_ratings.index)

        if rated_similar_items.empty:
             # If no similar items were rated by the user, return user's average rating
             return self.user_mean_ratings.get(user_id, np.nan)


        valid_similarities = similarities.loc[rated_similar_items]
        valid_ratings = user_ratings.loc[rated_similar_items]

        # Get top N similar items that the user has rated
        # Ensure we only consider items with positive similarity for the weighted sum
        top_items_mask = valid_similarities > 0
        top_items = valid_similarities[top_items_mask].nlargest(n_similar_items)


        if top_items.empty:
            # If no positively correlated similar items were rated, return user's average rating
            return self.user_mean_ratings.get(user_id, np.nan)

        # Calculate weighted sum of ratings
        numerator = (top_items * valid_ratings.loc[top_items.index]).sum()
        denominator = top_items.sum()

        # Handle case where denominator is zero (e.g., all top similarities are negative or zero,
        # but filtered above by top_items_mask) - though the nlargest(positive similarities) should handle this.
        # Added a small epsilon to denominator just in case, though should not be needed with top_items_mask
        epsilon = 1e-9
        if denominator == 0: # Should ideally not happen with top_items based on positive similarities
             return self.user_mean_ratings.get(user_id, np.nan)


        predicted_score = numerator / denominator
        # Item-based often predicts deviations from mean, so add user's mean back
        # However, the prediction formula is usually SUM(sim * rating) / SUM(sim).
        # If using normalized ratings (rating - mean), the formula is different.
        # Let's re-check standard item-based prediction formula for ratings directly.
        # It's typically (sum(sim(i,j) * r(u,j))) / sum(|sim(i,j)|) for item i, user u, over rated items j.
        # Let's use the formula for raw ratings: sum(sim * rating) / sum(|sim|)
        # The current calculation was numerator/denominator using 'top_items' which were positive correlations.
        # The standard formula is sum(sim * rating) / sum(ABS(sim)).
        # Let's re-implement the prediction based on the standard formula for raw ratings.

        # Re-calculating numerator and denominator based on ALL rated similar items (not just top N initially)
        # Then take top N *after* getting all similarities and ratings.
        all_similarities_rated = similarities.loc[user_ratings.index]
        all_ratings_rated = user_ratings.loc[user_ratings.index]

        # Calculate weighted ratings and sum of absolute similarities for *all* rated similar items first
        weighted_ratings = all_similarities_rated * all_ratings_rated
        sum_abs_similarities = all_similarities_rated.abs().sum()

        if sum_abs_similarities == 0:
             return self.user_mean_ratings.get(user_id, np.nan)

        # Now, filter for top N based on similarity magnitude? Or predict using all rated?
        # Standard Item-based prediction is often a weighted average over ALL items the user rated,
        # weighted by similarity to the target item. Let's use that.

        # Re-calculating based on ALL rated items by the user, weighted by their similarity to the target product_name
        user_rated_items = self.train_matrix.loc[user_id].dropna()
        rated_item_names = user_rated_items.index

        # Get similarities of the target product_name with ALL items the user has rated
        similarities_with_rated_items = self.item_similarity.loc[rated_item_names, product_name]

        # Remove the target item itself if it's in the list
        if product_name in similarities_with_rated_items.index:
             similarities_with_rated_items = similarities_with_rated_items.drop(product_name)
             user_rated_items = user_rated_items.drop(product_name)


        # Ensure alignment of similarities and ratings after dropping
        common_items = similarities_with_rated_items.index.intersection(user_rated_items.index)
        similarities_with_rated_items = similarities_with_rated_items.loc[common_items]
        user_rated_items = user_rated_items.loc[common_items]


        # Calculate weighted sum of ratings: sum(sim(i,j) * r(u,j)) for target item i, user u, over rated items j
        numerator = (similarities_with_rated_items * user_rated_items).sum()
        # Calculate sum of absolute similarities: sum(|sim(i,j)|)
        denominator = similarities_with_rated_items.abs().sum()

        if denominator == 0:
             # If no similar items rated by user, return user's average rating
             return self.user_mean_ratings.get(user_id, np.nan)

        predicted_score = numerator / denominator

        # Clip predictions to rating scale
        predicted_score = max(1, min(5, predicted_score))


        return predicted_score


    def recommend_top_k(self, user_id, k=10):
        """Gợi ý top-k sản phẩm cho một user."""
        try:
            rated_products = self.train_matrix.loc[user_id].dropna().index
        except KeyError:
            return [] # User không có trong tập train

        all_products = self.train_matrix.columns
        products_to_recommend = all_products.drop(rated_products, errors='ignore')

        predictions = {}
        # Only predict for items that the model knows about
        known_products_to_recommend = products_to_recommend.intersection(self.item_similarity.columns)

        for product in known_products_to_recommend:
            pred = self.predict(user_id, product)
            if not np.isnan(pred):
                predictions[product] = pred

        sorted_predictions = sorted(predictions.items(), key=lambda item: item[1], reverse=True)
        return [product for product, score in sorted_predictions[:k]]


###### Step 4: Đánh Giá Mô Hình (Không đổi)

def evaluate_model(model, train_df, test_df, k=10):
    """Tính toán tất cả các chỉ số: MAE, RMSE, Precision, Recall, F1, Coverage."""
    print("\nBắt đầu đánh giá mô hình trên tập Test...")

    predictions, actuals = [], []
    # Iterate through test_df and predict for each user-item pair
    for index, row in tqdm(test_df.iterrows(), desc="Tính MAE/RMSE...", total=len(test_df)):
        user_id = row['user_id']
        product_name = row['product_name']
        actual_rating = row['rating']

        pred = model.predict(user_id, product_name)
        if not np.isnan(pred):
            predictions.append(pred)
            actuals.append(actual_rating)

    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))

    # # --- Đánh giá các chỉ số xếp hạng ---
    # precisions, recalls, all_recommended_items = [], [], set()
    # # Use users who are in the test set AND have rated items in the train set
    # test_users = test_df['user_id'].unique()
    # train_users = model.train_matrix.index
    # relevant_test_users = [u for u in test_users if u in train_users]


    # for user_id in tqdm(relevant_test_users, desc=f"Tính Precision/Recall/Coverage@{k}..."):
    #     # Relevant items are those rated >= 4.0 in the test set
    #     relevant_items = set(test_df[(test_df['user_id'] == user_id) & (test_df['rating'] >= 4.0)]['product_name'])
    #     if not relevant_items:
    #         continue

    #     recommended_items = set(model.recommend_top_k(user_id, k=k))
    #     all_recommended_items.update(recommended_items)
    #     hits = len(recommended_items.intersection(relevant_items))

    #     if k > 0:
    #         precisions.append(hits / k)
    #     else:
    #         precisions.append(0)

    #     if len(relevant_items) > 0:
    #         recalls.append(hits / len(relevant_items))
    #     else:
    #         recalls.append(0)


    # avg_precision = np.mean(precisions) if precisions else 0
    # avg_recall = np.mean(recalls) if recalls else 0
    # avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

    # # Coverage is calculated based on all unique items in the train matrix columns
    # total_items_in_train = len(model.train_matrix.columns)
    # coverage = len(all_recommended_items) / total_items_in_train if total_items_in_train > 0 else 0

    print("\n==============================================")
    print("      KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (ITEM-BASED)    ")
    print("==============================================")
    print(f"MAE (Mean Absolute Error):      {mae:.4f}")
    print(f"RMSE (Root Mean Squared Error):   {rmse:.4f}")
    # print("----------------------------------------------")
    # print(f"Precision@10:                     {avg_precision:.4f}")
    # print(f"Recall@10:                        {avg_recall:.4f}")
    # print(f"F1-Score@10:                      {avg_f1:.4f}")
    # print("----------------------------------------------")
    # print(f"Coverage:                         {coverage:.4f}")
    # print("==============================================")

###### MAIN SCRIPT
if __name__ == '__main__':
    start_time = time.time()
    filepath = 'data_reviews_purchase.csv'

    # Reduced threshold for demonstration, you can adjust this back to 4 if preferred
    df_filtered = load_and_prepare_data(filepath, min_ratings_threshold=4)

    if df_filtered is not None:
        train_set, test_set = create_train_test_split(df_filtered, test_size=0.2)

        # Use the corrected ITEM-BASED model
        model = ItemBasedCollaborativeFilteringModel()
        model.fit(train_set)

        evaluate_model(model, train_set, test_set, k=10)

    end_time = time.time()
    print(f"\nTổng thời gian thực thi: {end_time - start_time:.2f} giây")