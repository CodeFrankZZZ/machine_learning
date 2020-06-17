import pandas
import matplotlib.pyplot as plt
import numpy as np
iris = pandas.read_csv("/Users/frank/Desktop/shuju4444.csv")
# shuffle rows
shuffled_rows = np.random.permutation(iris.index)
iris = iris.loc[shuffled_rows,:]
print(iris.head())
'''
             grade  grants  jingjinan  ...  fromcity  fromcountry  quxiang
5636  3.065766     0.0          1  ...         1            0        1
5373  2.736650     0.0          0  ...         1            0        1
27    3.529005     0.0          0  ...         0            1        1
2026  2.282046     0.0          0  ...         0            1        3
2388  2.926846     0.0          0  ...         1            0        1
'''
# There are 2 species
print(iris.quxiang.unique())
'''
[[1 3 2 0]]
'''
iris.hist()
plt.show()