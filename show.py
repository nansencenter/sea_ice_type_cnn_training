import matplotlib.pyplot as plt
import numpy as np
reconstructed_arr = np.load("/abs/pat/to/reconstructed/npz/file")
plt.imshow(reconstructed_arr['arr_0'])
plt.show()
