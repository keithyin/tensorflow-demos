import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

print(tf.__version__)

df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar',
                         'foo', 'bar', 'foo', 'foo'],
                   'B': ['one', 'one', 'two', 'three',
                         'two', 'two', 'one', 'three'],
                   'C': np.random.randn(8),
                   'D': np.random.randn(8)})

df.groupby('A')['C'].plot(kind="kde", legend=True)
# for i in range(0, 10, 0.5):
#     print(i)


