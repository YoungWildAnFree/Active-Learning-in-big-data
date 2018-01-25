from time import time
from elm import ELMClassifier
from random_hidden_layer import SimpleRandomHiddenLayer
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

import ELM_Spark_Active


def res_dist(x, y, e, n_runs=100, random_state=None):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)

    test_res = []
    train_res = []
    start_time = time()

    for i in xrange(n_runs):
        e.fit(x_train, y_train)
        train_res.append(e.score(x_train, y_train))
        test_res.append(e.score(x_test, y_test))
        if (i%5 == 0): print "%d"%i,

    print "\nTime: %.3f secs" % (time() - start_time)

    print "Test Min: %.3f Mean: %.3f Max: %.3f SD: %.3f" % (min(test_res), np.mean(test_res), max(test_res), np.std(test_res))
    print "Train Min: %.3f Mean: %.3f Max: %.3f SD: %.3f" % (min(train_res), np.mean(train_res), max(train_res), np.std(train_res))
    print
    return (train_res, test_res)


stdsc = StandardScaler()

iris = load_iris()
irx, iry = stdsc.fit_transform(iris.data), iris.target
irx_train, irx_test, iry_train, iry_test = train_test_split(irx, iry, test_size=0.2)

srh = SimpleRandomHiddenLayer(activation_args='sigmoid',n_hidden=500)
elmc = ELMClassifier(hidden_layer=srh)
# elmc = ELMClassifier(SimpleRandomHiddenLayer(activation_func='sigmoid'))
# print "SimpleRandomHiddenLayer(activation_func='sigmoid')"
# tr,ts = res_dist(irx, iry, elmc, n_runs=100, random_state=0)
# plt.hist(tr),plt.hist(tr)
# plt.show()


elmc.fit(irx_train,iry_train)
r = elmc.predict(irx_test)
print r
res = elmc.score(irx_test,iry_test)
print res