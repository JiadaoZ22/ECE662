from scipy.stats import kde
from utilities import *

window_bw = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Initialize array to store the error rate for different dimensions, N
emp_error = []

#########################################################################
# 2 dimensional for 5 different classes

# Covariance matrices
cov_mats = {}
for i in range(1, 6):
    cov_mats[i] = i * np.eye(2)

# mean vectors
mu_vecs = {}
for i, j in zip(range(1, 6), [[0, 0], [3, 0], [4, 5], [3, 6], [7, 3]]):
    mu_vecs[i] = np.array(j).reshape(2, 1)

# Generating the random samples
all_samples = {}
for i in range(1, 6):
    # generating 80x5 dimensional arrays with random Gaussian-distributed samples
    class_samples = np.random.multivariate_normal(mu_vecs[i].ravel(), cov_mats[i], 80)
    # adding class label to 3rd column
    class_samples = np.append(class_samples, np.zeros((80, 1))+i, axis=1)
    all_samples[i] = class_samples

# Dividing the samples into training and test datasets
train_set = np.append(all_samples[1][0:40], all_samples[2][0:40], axis=0)
train_set = np.append(train_set, all_samples[3][0:40], axis=0)
train_set = np.append(train_set, all_samples[4][0:40], axis=0)
train_set = np.append(train_set, all_samples[5][0:40], axis=0)

test_set = np.append(all_samples[1][40:80], all_samples[2][40:80], axis=0)
test_set = np.append(test_set, all_samples[3][40:80], axis=0)
test_set = np.append(test_set, all_samples[4][40:80], axis=0)
test_set = np.append(test_set, all_samples[5][40:80], axis=0)

# 40 for training and 40 for testing for each class, thus training data and testing data should have
# size of (40 * 5 = 200, 3) shape, last column is the class column
assert(train_set.shape == (200, 3))
assert(test_set.shape == (200, 3))

for bw in window_bw:
    class1_kde = kde.gaussian_kde(train_set[train_set[:,2] == 1].T[0:2], bw_method=bw)
    class2_kde = kde.gaussian_kde(train_set[train_set[:,2] == 2].T[0:2], bw_method=bw)

    classification_dict, error = empirical_error(test_set, [1,2], bayes_classifier,
                                 [[class1_kde, class2_kde]])

    labels_predicted = ['w{} (predicted)'.format(i) for i in [1,2]]
    labels_predicted.insert(0,'test dataset')

    train_conf_mat = prettytable.PrettyTable(labels_predicted)
    for i in [1, 2]:
        a, b = [classification_dict[i][j] for j in [1, 2]]
        # workaround to unpack (since Python does not support just '*a')
        train_conf_mat.add_row(['w{} (actual)'.format(i), a, b])
    print(train_conf_mat)
    print('Empirical Error: {:.2f} ({:.2f}%)'.format(error, error * 100))
    emp_error.append(error)

#########################################################################
# Plots of the graphs
# First plot, 2 dimensional for 2 classes
fig = plt.figure()

plt.plot(window_bw, emp_error)
plt.title('Error rate vs Parzen window sizes for 2 features from 5 classes')
plt.xlabel('Parzen window sizes')
plt.ylabel('Error rate')
plt.grid()
plt.show()
