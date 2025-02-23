import numpy 

data_t = numpy.load("/home/alic-li/文档/DFT_量化交易/data_t.npy", allow_pickle=True)
print("data_t:", data_t.shape)
data_X = numpy.load("/home/alic-li/文档/DFT_量化交易/data_X.npy", allow_pickle=True)
print("data_X:", data_X.shape)
data_Y = numpy.load("/home/alic-li/文档/DFT_量化交易/data_Y.npy", allow_pickle=True)
print("data_Y:", data_Y.shape)
X_cloums = numpy.load("/home/alic-li/文档/DFT_量化交易/X_colums.npy", allow_pickle=True)
print("X_cloums:", X_cloums.shape)
y_cloums = numpy.load("/home/alic-li/文档/DFT_量化交易/Y_colums.npy", allow_pickle=True)
print("y_cloums:", y_cloums.shape)