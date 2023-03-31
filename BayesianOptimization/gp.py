import numpy as np
# import matplotlib.pyplot as plt

# 径向基函数 k(s, t) = σ^2 exp(-||s-t||^2 / 2t^2)
def gaussian_kernel(x1, x2, C, l=1.0, sigma_f=1.0):
    m, n = x1.shape[0], x2.shape[0]
    dist_matrix = np.zeros((m, n), dtype=float)
    sim_matrix = np.zeros((m, n), dtype=float)
    for i in range(m):
        for j in range(n):
            dist_matrix[i][j] = np.sum((x1[i] - x2[j]) ** 2)
            sim_matrix[i][j] = np.sum((C[i] - C[j]) ** 2)

    return sigma_f ** 2 * np.exp(-0.5 / 1 ** 2 * dist_matrix)
    # return sigma_f ** 2 * (np.exp(-0.5 / 1 ** 2 * dist_matrix) - 0.5 * sim_matrix)  


train_X = np.array([1, 3, 7, 9]).reshape(-1, 1) # 转为4*1的矩阵形式
train_C = np.array([0.145, 0.317, 0.054, 0.145]).reshape(-1, 1)
print(gaussian_kernel(train_X, train_X, train_C))  # 4*4的协方差矩阵，当i=j时就是自身的方差

# 生成观测值
# def getY(X):
#     X = np.asarray(X)
#     # y = 0.4sin(x) + N(0, 0.05)
#     Y = np.sin(X) * 0.4 + np.random.normal(0, 0.05, size=X.shape)
#     return Y.tolist()

# 根据观察点X，修正生成高斯过程新的均值和协方差
# def update(X, X_star):
#     X = np.asarray(X)
#     X_star = np.asarray(X_star)
#     K_YY = gaussian_kernel(X, X)
#     K_ff = gaussian_kernel(X_star, X_star)
#     K_Yf = gaussian_kernel(X, X_star)
#     K_fY = K_Yf.T
#     K_YY_inv = np.linalg.inv(K_YY + 1e-8 * np.eye(len(X)))

#     Y = getY(X)
#     mu_star = K_fY.dot(K_YY_inv).dot(Y)
#     cov_star = K_ff - K_fY.dot(K_YY_inv).dot(K_Yf)
#     return mu_star, cov_star

# f, ax = plt.subplots(2, 1, sharex=True, sharey=True)

# 绘制高斯过程的先验
# X_pre = np.arange(0, 10, 0.1)
# mu_pre = np.array([0]*len(X_pre))
# Y_pre = mu_pre
# cov_pre = gaussian_kernel(X_pre, X_pre)
# uncertainty = 1.96 * np.sqrt(np.diag(cov_pre))#取95%置信区间
# ax[0].fill_between(X_pre, Y_pre + uncertainty, Y_pre - uncertainty, alpha=0.1)
# ax[0].plot(X_pre, Y_pre, label="expection")
# ax[0].legend()

# 绘制基于观测值的高斯过程后验
# X = np.array([1, 3, 7, 9]).reshape(-1, 1)
# Y = getY(X)
# X_star = np.arange(0, 10, 0.1).reshape(-1, 1)
# mu_star, cov_star = update(X, X_star)
# Y_star = mu_star.ravel()
# uncertainty = 1.96 * np.sqrt(np.diag(cov_star))#取95%置信区间
# ax[1].fill_between(X_star.ravel(), Y_star + uncertainty, Y_star - uncertainty, alpha=0.1)
# ax[1].plot(X_star, Y_star, label="expection")
# ax[1].scatter(X, Y, label="observation point", c="red", maker="x")
# ax[1].legend()
# plt.show()

