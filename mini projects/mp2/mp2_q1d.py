from scipy.stats import kde
from utilities import *

training_samples = [10, 50, 100, 500, 1000, 5000, 10000]

# Initialize array to store the error rate for different dimensions, N
emp_error_2n = []
emp_error_3n = []
emp_error_4n = []

#########################################################################
# 2 dimensional for 2 different classes
# Covariance matrices
cov_mats = {}
for i in range(1, 3):
    cov_mats[i] = i * np.eye(2)

# mean vectors
mu_vecs = {}
for i, j in zip(range(1, 3), [[0, 0], [3, 0]]):
    mu_vecs[i] = np.array(j).reshape(2, 1)

# Generating the random samples
for train_size in training_samples:
    all_samples = {}
    for i in range(1, 3):
        # generating 80x3 dimensional arrays with random Gaussian-distributed samples
        class_samples = np.random.multivariate_normal(mu_vecs[i].ravel(), cov_mats[i], train_size)
        # adding class label to 3rd column
        class_samples = np.append(class_samples, np.zeros((train_size, 1))+i, axis=1)
        all_samples[i] = class_samples

    # Dividing the samples into training and test datasets, half for train and testing
    train_set = np.append(all_samples[1][0:int((train_size/2))], all_samples[2][0:(int(train_size/2))], axis=0)
    test_set = np.append(all_samples[1][int((train_size/2)):int(train_size)], all_samples[2][int((train_size/2)):
                                                                                             int(train_size)], axis=0)

    assert(train_set.shape == (train_size, 3))
    assert(test_set.shape == (train_size, 3))

    class1_kde = kde.gaussian_kde(train_set[train_set[:,2] == 1].T[0:2], bw_method='scott')
    class2_kde = kde.gaussian_kde(train_set[train_set[:,2] == 2].T[0:2], bw_method='scott')

    classification_dict, error = empirical_error(test_set, [1, 2], bayes_classifier,
                                 [[class1_kde, class2_kde]])

    labels_predicted = ['w{} (predicted)'.format(i) for i in [1,2]]
    labels_predicted.insert(0, 'test dataset')

    train_conf_mat = prettytable.PrettyTable(labels_predicted)
    for i in [1, 2]:
        a, b = [classification_dict[i][j] for j in [1, 2]]
        # workaround to unpack (since Python does not support just '*a')
        train_conf_mat.add_row(['w{} (actual)'.format(i), a, b])
    print(train_conf_mat)
    print('Empirical Error: {:.2f} ({:.2f}%)'.format(error, error * 100))
    emp_error_2n.append(error)

#########################################################################
# 3 dimensional for 2 different classes
# Covariance matrices
cov_mats = {}
for i in range(1,3):
    cov_mats[i] = i * np.eye(3)

# mean vectors
mu_vecs = {}
for i, j in zip(range(1, 3), [[0, 0, 0], [3, 0, 3]]):
    mu_vecs[i] = np.array(j).reshape(3, 1)

for train_size in training_samples:
    all_samples = {}
    for i in range(1, 3):
        # generating 80x3 dimensional arrays with random Gaussian-distributed samples
        class_samples = np.random.multivariate_normal(mu_vecs[i].ravel(), cov_mats[i], train_size)
        # adding class label to 3rd column
        class_samples = np.append(class_samples, np.zeros((train_size, 1))+i, axis=1)
        all_samples[i] = class_samples

    # Dividing the samples into training and test datasets, half for train and testing
    train_set = np.append(all_samples[1][0:int((train_size/2))], all_samples[2][0:(int(train_size/2))], axis=0)
    test_set = np.append(all_samples[1][int((train_size/2)):int(train_size)], all_samples[2][int((train_size/2)):
                                                                                             int(train_size)], axis=0)

    assert(train_set.shape == (train_size, 4))
    assert(test_set.shape == (train_size, 4))

    class1_kde = kde.gaussian_kde(train_set[train_set[:, 3] == 1].T[0:3], bw_method='scott')
    class2_kde = kde.gaussian_kde(train_set[train_set[:, 3] == 2].T[0:3], bw_method='scott')

    classification_dict, error = empirical_error(test_set, [1, 2], bayes_classifier,
                                 [[class1_kde, class2_kde]])

    labels_predicted = ['w{} (predicted)'.format(i) for i in [1, 2]]
    labels_predicted.insert(0, 'test dataset')

    train_conf_mat = prettytable.PrettyTable(labels_predicted)
    for i in [1, 2]:
        a, b = [classification_dict[i][j] for j in [1, 2]]
        # workaround to unpack (since Python does not support just '*a')
        train_conf_mat.add_row(['w{} (actual)'.format(i), a, b])
    print(train_conf_mat)
    print('Empirical Error: {:.2f} ({:.2f}%)'.format(error, error * 100))
    emp_error_3n.append(error)

#########################################################################
# 4 dimensional for 2 classes
# Covariance matrices
cov_mats = {}
for i in range(1, 3):
    cov_mats[i] = i * np.eye(4)

# mean vectors
mu_vecs = {}
for i, j in zip(range(1, 3), [[1, 1, 0, 7], [3, 8, 3, 1], [4, 5, 9, 6]]):
    mu_vecs[i] = np.array(j).reshape(4, 1)

for train_size in training_samples:
    all_samples = {}
    for i in range(1, 3):
        # generating 80x3 dimensional arrays with random Gaussian-distributed samples
        class_samples = np.random.multivariate_normal(mu_vecs[i].ravel(), cov_mats[i], train_size)
        # adding class label to 3rd column
        class_samples = np.append(class_samples, np.zeros((train_size, 1))+i, axis=1)
        all_samples[i] = class_samples

    # Dividing the samples into training and test datasets, half for train and testing
    train_set = np.append(all_samples[1][0:int((train_size/2))], all_samples[2][0:(int(train_size/2))], axis=0)
    test_set = np.append(all_samples[1][int((train_size/2)):int(train_size)], all_samples[2][int((train_size/2)):
                                                                                             int(train_size)], axis=0)

    assert(train_set.shape == (train_size, 5))
    assert(test_set.shape == (train_size, 5))

    class1_kde = kde.gaussian_kde(train_set[train_set[:, 4] == 1].T[0:4], bw_method='scott')
    class2_kde = kde.gaussian_kde(train_set[train_set[:, 4] == 2].T[0:4], bw_method='scott')

    classification_dict, error = empirical_error(test_set, [1, 2], bayes_classifier,
                                 [[class1_kde, class2_kde]])

    labels_predicted = ['w{} (predicted)'.format(i) for i in [1, 2]]
    labels_predicted.insert(0, 'test dataset')

    train_conf_mat = prettytable.PrettyTable(labels_predicted)
    for i in [1, 2]:
        a, b = [classification_dict[i][j] for j in [1, 2]]
        # workaround to unpack (since Python does not support just '*a')
        train_conf_mat.add_row(['w{} (actual)'.format(i), a, b])
    print(train_conf_mat)
    print('Empirical Error: {:.2f} ({:.2f}%)'.format(error, error * 100))
    emp_error_4n.append(error)


#########################################################################
# Plots of the graphs
# First plot, 2 dimensional for 2 classes
fig = plt.figure()

plt.plot(training_samples, emp_error_2n)
plt.title('Error rate vs different training sizes for 2 dimensional feature vectors')
plt.xlabel('Training sizes')
plt.ylabel('Error rate')
plt.grid()
plt.show()

# Second plot, 3 dimensional for 2 classes
fig = plt.figure()

plt.plot(training_samples, emp_error_3n)
plt.title('Error rate vs different training sizes for 3 dimensional feature vectors')
plt.xlabel('Training sizes')
plt.ylabel('Error rate')
plt.grid()
plt.show()

# Third plot, 4 dimensional for 2 classes
fig = plt.figure()

plt.plot(training_samples, emp_error_4n)
plt.title('Error rate vs different training sizes for 4 dimensional feature vectors')
plt.xlabel('Training sizes')
plt.ylabel('Error rate')
plt.grid()
plt.show()
