import pandas as pd
import numpy as np

class MatrixFactorizationRecommender():

  def __init__(self, data: pd.DataFrame, k: int, alpha: float, lam: float, tol: float = 1e-4 ,max_iters: int = 100):
    """
    @params
      - data: dataframe với dòng~user, cột~item, value ~ rating
      - K: số chiều của nhân tố ẩn, cho cả U và V
      - alpha : cỡ bước
      - lam : tham số hiệu chỉnh
      - max_iters: số vòng lặp
    """
    self.data = data
    self.R = self.data.to_numpy().astype(np.float32)
    self.m, self.n = self.data.shape
    self.k = k
    self.tol = tol
    self.max_iters = max_iters
    # Khởi tạo các ma trận nhân tố ẩn U(m x k) và V(n x k)
    self.U = np.random.randn(self.m, k).astype(np.float32)
    self.V = np.random.randn(self.n, k).astype(np.float32)

    # Khởi tạo các siêu tham số (hyperparameter)
    self.alpha = alpha
    self.lam = lam

  def init_biases(self):
    """
    Khởi tạo các bias cho người dùng và item
    """
    self.b_user = np.zeros(self.m).astype(np.float32) # bias đối với users
    self.b_item = np.zeros(self.n).astype(np.float32) # bias đối với item

  def sgd(self):
    """
    Hàm Stochastic Gradient Descent để cập nhật bias và ma trận nhân tố ẩn U, V
    """
    for i, j, r in self.S:
      # Ước lượng rating
      prediction = self.get_rating(i, j)
      # Tính sai số
      e = r - prediction

      # Cập nhật biases
      self.b_user[i] += self.alpha * (e - self.lam * self.b_user[i])
      self.b_item[j] += self.alpha * (e - self.lam * self.b_item[j])

      # Tạo ma trận U, V trung gian
      U_i = self.U[i, :]
      V_i = self.V[j, :]

      # Cập nhật ma trận U và V
      self.U[i, :] += self.alpha * (e * V_i - self.lam * U_i)
      self.V[j, :] += self.alpha * (e * U_i - self.lam * V_i)

  def get_rating(self, i: int, j: int) -> float:
    """
    Dự đoán rating của user i đối với item j
    """
    pred = self.b_user[i] + self.b_item[j] + self.U[i, :] @ self.V[j, :].T
    return pred

  def full_matrix(self) -> pd.DataFrame:
    """
      Cập nhật ma trận đầy đủ sử dụng các bias, ma trận nhân tố ẩn U và V
    """
    predicted_R = self.b_user[:, None] + self.b_item[None, :] + self.U @ self.V.T
    return pd.DataFrame(np.clip(predicted_R, 1, 5),
                        index=self.data.index, columns=self.data.columns)

  def rmse(self) -> float:
    """
      Hàm tính căn bậc hai trung bình bình phương sai số (RMSE)
    """
    # Ma trận rating dự đoán
    predicted = self.full_matrix().to_numpy()
    # Lấy các vị trí mà r_ij != nan
    mask = ~np.isnan(self.R)
    return np.sqrt(np.mean((self.R[mask] - predicted[mask])**2))

  def loss(self) -> float:
    return 0.5 * (self.rmse() ** 2 + self.lam * (
        np.sum(self.U**2) + np.sum(self.V**2) +
        np.sum(self.b_user**2) + np.sum(self.b_item**2)
    ))

  def train(self):
    """
    Huấn luyện mô hình bằng phương pháp Stochastic Gradient Descent (SGD)
    """
    self.init_biases()
    # Tạo tập S = [(i,j,r_ij): r_ij != nan]
    self.S = [(i, j, self.R[i, j])
              for i in range(self.m)
              for j in range(self.n)
              if not np.isnan(self.R[i, j])]

    training_process = [] # Danh sách lưu quá trình huấn luyện
    patience = 5  # Số lần lặp không cải thiện trước khi dừng
    best_rmse = float('inf') # Biến lưu giá trị RMSE tốt nhất
    no_improve = 0 # Biến đếm số lần không cải thiện

    for iter in range(self.max_iters):
      np.random.shuffle(self.S)
      # Cập nhật U và V
      self.sgd()
      # Tính sai số rmse
      error = self.rmse()
      # Hàm loss
      loss = self.loss()
      # Lưu lần lặp và sai số tương ứng
      training_process.append((iter, loss, error))
      # In thông tin quá trình huấn luyện
      if (iter + 1) % 2 == 0:
        print(f"Iter {iter+1}: Loss = {loss:.8f}, RMSE = {error:.8f}")
      # Kiểm tra điều kiện dừng sớm
      if error < best_rmse - self.tol:
          best_rmse = error
          no_improve = 0
      else:
          no_improve += 1
          if no_improve >= patience:
              print(f"Early stopping at iteration {iter+1}")
              break

    return training_process

  def pred_for_users(self, target_user: int) -> pd.DataFrame:
    """
    Dự đoán rating cho các item chưa được đánh giá bởi người dùng mục tiêu
    @params
      - target_user: id của người dùng mục tiêu
    @returns
      - DataFrame chứa các item chưa được đánh giá và rating dự đoán
    """
    rated = self.data.loc[target_user].dropna().index
    unrated = self.data.columns.difference(rated)
    return (self.full_matrix().loc[[target_user], unrated]
            .T.rename(columns={target_user: 'pred_rating'})
            .sort_values('pred_rating', ascending=False)
            .round(2))