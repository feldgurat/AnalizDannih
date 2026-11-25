import numpy as np
import pandas as pd
from scipy import stats

alpha = 0.05
t_gamma = 1.96

data = {
    "x1": [13.26, 10.16, 13.72, 12.82, 10.63, 9.12, 25.83, 23.39, 14.68, 10.05],
    "x2": [1.23, 1.04, 1.80, 0.43, 0.88, 0.57, 1.72, 1.70, 0.84, 0.60],
    "x3": [1.45, 1.30, 1.37, 1.65, 1.91, 1.68, 1.94, 1.89, 1.94, 2.06],
}
df = pd.DataFrame(data)
n = len(df)

# а) Средние, СКО, матрица R
means = df.mean().to_numpy()
S = df.std(ddof=0).to_numpy()
R = df.corr().to_numpy()
print("а) Вектор средних:", np.round(means, 4))
print("   Вектор СКО:   ", np.round(S, 4))
print("   Матрица R:\n", np.round(R, 4))

r12 = R[0, 1]

# б) Значимость r12 и интервальная оценка
t12 = r12 * np.sqrt(n - 2) / np.sqrt(1 - r12**2)
t12_crit = stats.t.ppf(1 - alpha / 2, n - 2)
r12_crit = t12_crit / np.sqrt(t12_crit**2 + n - 2)
Z12 = np.arctanh(r12)
dZ12 = t_gamma / np.sqrt(n - 3)
print(f"\nб) r12 = {r12:.4f}, t набл = {t12:.4f}, t кр = {t12_crit:.4f}, r кр = {r12_crit:.4f}")
print(f"   Вывод: {'значим' if abs(t12) > t12_crit else 'не значим'}")
print(f"   Z' = {Z12:.4f}, Z in [{Z12 - dZ12:.4f}; {Z12 + dZ12:.4f}]")
print(f"   rho12 in [{np.tanh(Z12 - dZ12):.3f}; {np.tanh(Z12 + dZ12):.3f}]")

# в) Частный коэффициент r12/3 через алгебраические дополнения
A11 = R[1, 1] * R[2, 2] - R[1, 2] * R[2, 1]
A22 = R[0, 0] * R[2, 2] - R[0, 2] * R[2, 0]
A12 = -(R[0, 1] * R[2, 2] - R[0, 2] * R[2, 1])
r12_3 = -A12 / np.sqrt(A11 * A22)
print(f"\nв) A11 = {A11:.4f}, A22 = {A22:.4f}, A12 = {A12:.4f}")
print(f"   r12/3 = {r12_3:.4f}")

# г) Значимость r12/3 и интервальная оценка
l = 1
nu = n - l - 2
t_p = r12_3 * np.sqrt(nu) / np.sqrt(1 - r12_3**2)
t_p_crit = stats.t.ppf(1 - alpha / 2, nu)
r_p_crit = t_p_crit / np.sqrt(t_p_crit**2 + nu)
Z_p = np.arctanh(r12_3)
dZ_p = t_gamma / np.sqrt(n - l - 3)
print(f"\nг) v = {nu}, t набл = {t_p:.4f}, t кр = {t_p_crit:.4f}, r кр = {r_p_crit:.4f}")
print(f"   Вывод: {'значим' if abs(t_p) > t_p_crit else 'не значим'}")
print(f"   Z' = {Z_p:.4f}, Z in [{Z_p - dZ_p:.4f}; {Z_p + dZ_p:.4f}]")
print(f"   rho12/3 in [{np.tanh(Z_p - dZ_p):.3f}; {np.tanh(Z_p + dZ_p):.3f}]")

# д) Множественный коэффициент корреляции и F-критерий
det_R = np.linalg.det(R)
r_mult = np.sqrt(1 - det_R / A11)
k = 3
F_obs = (r_mult**2 / (k - 1)) / ((1 - r_mult**2) / (n - k))
F_crit = stats.f.ppf(1 - alpha, k - 1, n - k)
print(f"\nд) |R| = {det_R:.4f}, r1(2,3) = {r_mult:.4f}, r^2 = {r_mult**2:.4f}")
print(f"   F набл = {F_obs:.4f}, F кр = {F_crit:.4f}")
print(f"   Вывод: {'значим' if F_obs > F_crit else 'не значим'}")
