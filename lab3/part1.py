import numpy as np
from scipy import stats

X = np.array([22, 45, 61, 66, 72, 83, 73, 82, 78, 82, 81, 82, 77, 80])
Y = np.array([8, 18, 24, 26, 28, 29, 30, 30, 30, 32, 33, 34, 35, 34])
n = len(X)
alpha = 0.05
t_gamma = 1.96

# Средние и СКО (деление на n)
x_mean, y_mean = X.mean(), Y.mean()
Sx, Sy = X.std(ddof=0), Y.std(ddof=0)
r = ((X * Y).mean() - x_mean * y_mean) / (Sx * Sy)

# Критерий Стьюдента
t_obs = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)
t_crit = stats.t.ppf(1 - alpha / 2, n - 2)
r_crit = t_crit / np.sqrt(t_crit**2 + n - 2)

# Фишер
Z = np.arctanh(r)
dZ = t_gamma / np.sqrt(n - 3)
rho_min, rho_max = np.tanh(Z - dZ), np.tanh(Z + dZ)

print(f"n = {n}, v = {n - 2}")
print(f"Средние: X = {x_mean:.4f}, Y = {y_mean:.4f}")
print(f"СКО: Sx = {Sx:.4f}, Sy = {Sy:.4f}")
print(f"r = {r:.4f}")
print(f"t набл = {t_obs:.4f}, t кр = {t_crit:.4f}, r кр = {r_crit:.4f}")
print(f"Вывод: {'значим' if abs(t_obs) > t_crit else 'не значим'}")
print(f"Z' = {Z:.4f}, Z in [{Z - dZ:.4f}; {Z + dZ:.4f}]")
print(f"rho in [{rho_min:.3f}; {rho_max:.3f}]")
