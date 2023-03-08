import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('V_170.csv',header=None).values
w_node= pd.read_csv('un_dtw_170.csv',header=None).values

df_mean = np.mean(df, axis=0)
df_std = np.std(df, axis=0)

print(df_mean.shape)
print(df_std.shape)

index_mean = np.argsort(df_mean)

index_find = index_mean[50:70]
print(index_find)

df_save = df[:, index_find]
print(df_save[:4, -1])

np.savetxt('V_20.csv', df_save, delimiter=',', fmt='%.2f')

# plt.xlabel('mean')
# plt.ylabel('std')
#
# plt.scatter(df_mean, df_std)
# plt.show()

