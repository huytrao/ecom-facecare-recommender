import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm # Thư viện để hiển thị thanh tiến trình

###### Step 1: Tải và Chuẩn Bị Dữ Liệu

def load_and_prepare_data(filepath, min_ratings_threshold=5):
    """Tải dữ liệu, làm sạch và lọc người dùng có ít đánh giá."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file '{filepath}'. Vui lòng kiểm tra lại đường dẫn.")
        return None

    # Làm sạch cơ bản
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    if 'product_name_x' in df.columns:
        df.rename(columns={'product_name_x': 'product_name'}, inplace=True)

    print(f"Số lượng đánh giá ban đầu: {len(df)}")
    print(f"Số lượng người dùng ban đầu: {df['user_id'].nunique()}")

    # Lọc người dùng có ít hơn 'min_ratings_threshold' đánh giá
    user_counts = df['user_id'].value_counts()
    print(f"\nNgưỡng số đánh giá tối thiểu để giữ lại người dùng: {int(min_ratings_threshold)}")

    active_users = user_counts[user_counts >= min_ratings_threshold].index
    df_filtered = df[df['user_id'].isin(active_users)]

    print(f"Số lượng đánh giá sau khi lọc người dùng: {len(df_filtered)}")
    print(f"Số lượng người dùng sau khi lọc: {df_filtered['user_id'].nunique()}")

    return df_filtered

###### Step 2: Chia Dữ Liệu thành Tập Train và Test

def create_train_test_split(df, test_size=0.2):
    """
    Chia dữ liệu theo từng người dùng để đảm bảo mỗi người dùng
    đều có dữ liệu trong cả tập train và test.
    """
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

###### Step 3: Xây Dựng và "Huấn Luyện" Mô Hình

class CollaborativeFilteringModel:
    def __init__(self):
        self.user_similarity = None
        self.matrix_norm = None
        self.train_matrix = None
        self.user_mean_ratings = None

    def fit(self, train_df):
        """Xây dựng ma trận tương đồng từ tập train."""
        print("\nĐang huấn luyện mô hình (xây dựng ma trận)...")
        self.train_matrix = train_df.pivot_table(index='user_id', columns='product_name', values='rating')
        self.user_mean_ratings = self.train_matrix.mean(axis=1)
        self.matrix_norm = self.train_matrix.subtract(self.user_mean_ratings, axis='rows')
        self.user_similarity = self.matrix_norm.T.corr().fillna(0)

    def predict(self, user_id, product_name, n_similar_users=15):
        """Dự đoán rating của một user cho một sản phẩm."""
        if user_id not in self.user_similarity or product_name not in self.matrix_norm.columns:
            return self.user_mean_ratings.get(user_id, np.nan)

        product_ratings = self.matrix_norm[product_name]
        similarities = self.user_similarity[user_id]
        rated_users = product_ratings.dropna().index

        valid_similarities = similarities.loc[similarities.index.intersection(rated_users)]
        valid_ratings = product_ratings.loc[product_ratings.index.intersection(rated_users)]

        if valid_similarities.empty:
            return self.user_mean_ratings.get(user_id, np.nan)

        top_users = valid_similarities.nlargest(n_similar_users)

        numerator = (top_users * valid_ratings.loc[top_users.index]).sum()
        denominator = top_users.sum()

        if denominator == 0:
            return self.user_mean_ratings.get(user_id, np.nan)

        predicted_score = self.user_mean_ratings[user_id] + (numerator / denominator)
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
        for product in products_to_recommend:
            pred = self.predict(user_id, product)
            if not np.isnan(pred):
                predictions[product] = pred

        sorted_predictions = sorted(predictions.items(), key=lambda item: item[1], reverse=True)
        return [product for product, score in sorted_predictions[:k]]


###### Step 4: Đánh Giá Mô Hình (Bổ sung F1 và Coverage)

def evaluate_model(model, train_df, test_df, k=10):
    """Tính toán tất cả các chỉ số: MAE, RMSE, Precision, Recall, F1, Coverage."""
    print("\nBắt đầu đánh giá mô hình trên tập Test...")

    # --- Đánh giá MAE và RMSE ---
    predictions = []
    actuals = []
    for row in tqdm(test_df.itertuples(), desc="Tính MAE/RMSE...", total=len(test_df)):
        pred = model.predict(row.user_id, row.product_name)
        if not np.isnan(pred):
            predictions.append(pred)
            actuals.append(row.rating)

    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))

    # # --- Đánh giá các chỉ số xếp hạng ---
    # precisions = []
    # recalls = []
    # all_recommended_items = set() # THÊM MỚI: Set để tính coverage
    # test_users = test_df['user_id'].unique()

    # for user_id in tqdm(test_users, desc=f"Tính Precision/Recall/Coverage@{k}..."):
    #     # Sản phẩm thực sự liên quan: các sản phẩm trong test set mà user đánh giá cao (>=4)
    #     relevant_items = set(test_df[(test_df['user_id'] == user_id) & (test_df['rating'] >= 4.0)]['product_name'])

    #     if not relevant_items:
    #         continue

    #     # Sản phẩm được mô hình gợi ý
    #     recommended_items = set(model.recommend_top_k(user_id, k=k))

    #     # THÊM MỚI: Cập nhật set các item được gợi ý để tính coverage
    #     all_recommended_items.update(recommended_items)

    #     # Số lượng sản phẩm gợi ý đúng
    #     hits = len(recommended_items.intersection(relevant_items))

    #     precisions.append(hits / k)
    #     recalls.append(hits / len(relevant_items))

    # # --- Tính toán các chỉ số trung bình ---
    # avg_precision = np.mean(precisions) if precisions else 0
    # avg_recall = np.mean(recalls) if recalls else 0

    # # THÊM MỚI: Tính F1-Score
    # if (avg_precision + avg_recall) > 0:
    #     avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    # else:
    #     avg_f1 = 0

    # # THÊM MỚI: Tính Coverage
    # total_items_in_train = len(model.train_matrix.columns)
    # coverage = len(all_recommended_items) / total_items_in_train if total_items_in_train > 0 else 0

    # In kết quả
    print("\n==============================================")
    print("      KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH HOÀN CHỈNH      ")
    print("==============================================")
    print(f"MAE (Mean Absolute Error):      {mae:.4f}")
    print(f"RMSE (Root Mean Squared Error):   {rmse:.4f}")
    # print("----------------------------------------------")
    # print(f"Precision@10:                     {avg_precision:.4f}")
    # print(f"Recall@10:                        {avg_recall:.4f}")
    # print(f"F1-Score@10:                      {avg_f1:.4f}") # THÊM MỚI
    # print("----------------------------------------------")
    # print(f"Coverage:                         {coverage:.4f}") # THÊM MỚI
    # print("==============================================")

###### MAIN SCRIPT
if __name__ == '__main__':
    start_time = time.time()

    # Đường dẫn đến file dữ liệu đánh giá của bạn
    filepath = 'data_reviews_purchase.csv'

    # Bước 1
    df_filtered = load_and_prepare_data(filepath, min_ratings_threshold=5)

    if df_filtered is not None:
        # Bước 2
        # Tăng test_size lên một chút để có tập test đủ lớn cho việc đánh giá
        train_set, test_set = create_train_test_split(df_filtered, test_size=0.01)

        # Bước 3
        model = CollaborativeFilteringModel()
        model.fit(train_set)

        # Bước 4
        evaluate_model(model, train_set, test_set, k=10)

    end_time = time.time()
    print(f"\nTổng thời gian thực thi: {end_time - start_time:.2f} giây")