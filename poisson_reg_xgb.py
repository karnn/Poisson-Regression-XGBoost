#%%
import numpy as np
import xgboost as xgb
from typing import Tuple, Dict, List
from time import time
import argparse
import matplotlib
from matplotlib import pyplot as plt



kRows = 5000
train_portion = 4000
kBoostRound = 100
kSeed = 23

np.random.seed(seed=kSeed)

def generate_data() -> Tuple[xgb.DMatrix, xgb.DMatrix]:
    # Generates data distributed as follows:
    # y_i ~ poisson(lambda_i)
    # lambda_i = exp(alpha[0]*x[i,0] + alpha[1]*x[i,1])

    x = np.random.randn(kRows,2)
    alpha = np.random.randn(1,2)*.4
    y = np.random.poisson(lam=np.exp(np.sum(alpha*x,axis=1)), size=(kRows,))
    
    train_x: np.ndarray = x[: train_portion]
    train_y: np.ndarray = y[: train_portion]
    dtrain = xgb.DMatrix(train_x, label=train_y)

    test_x = x[train_portion:]
    test_y = y[train_portion:]
    dtest = xgb.DMatrix(test_x, label=test_y)
    true_params = {'alpha':alpha}
    return dtrain, dtest, true_params

# %%
def poisson_lossfn(dtrain: xgb.DMatrix, dtest: xgb.DMatrix) -> Dict:
    
    def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        y = dtrain.get_label()
        return np.exp(predt) - y
    
    def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        return np.exp(predt)

    def poisson_loss(predt: np.ndarray,
                    dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        '''Squared Log Error objective. A simplified version for RMSLE used as
        objective function.

        :math:`\frac{1}{2}[log(pred + 1) - log(label + 1)]^2`

        '''
        grad = gradient(predt, dtrain)
        hess = hessian(predt, dtrain)
        return grad, hess
    
    def loss(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
        ''' Root mean squared log error metric.

        :math:`\sqrt{\frac{1}{N}[log(pred + 1) - log(label + 1)]^2}`
        '''
        y = dtrain.get_label()
        elements = np.exp(predt) - y
        return 'PoissonReg', float(np.mean(elements**2))

    results: Dict[str, Dict[str, List[float]]] = {}
    start = time()
    bst = xgb.train({'eta': 0.1,
        'max_depth':3,'tree_method': 'hist', 'seed': 23,
               'disable_default_eval_metric': 1,
               },
               num_boost_round  = 100,
               early_stopping_rounds=5,
              dtrain=dtrain,
              obj=poisson_loss,
              feval=loss,
              evals_result=results,
              evals=[(dtrain, 'dtrain'), (dtest, 'dtest')],
              )
    print('Finished Error in:', time() - start)
    return bst, results

#%%
def plot_decision_lines(bst, base_data, col=True):
    a0, a1 = base_data['alpha'][0]
    if col==1:
        col1 = np.linspace(-2,2,1000).reshape(-1,1)
        mat = xgb.DMatrix(np.c_[col1,np.ones((1000,1))])
        y_pred_log = np.exp(bst.predict(mat))
        plt.plot(col1, np.exp(a0*col1 + a1), 'g-')
        plt.plot(col1,y_pred, 'r.' )
    else:
        col1 = np.linspace(-2,2,1000).reshape(-1,1)
        mat = xgb.DMatrix(np.c_[np.ones((1000,1)), col1])
        y_pred_log = bst.predict(mat)
        plt.plot(col1, np.exp(a1*col1 + a0), 'g-')
        plt.plot(col1,np.exp(y_pred_log), 'r.' )
#%%
def main():
    dtrain, dtest, base_data = generate_data()
    bst, results = poisson_lossfn(dtrain,dtest)
    #plot_history(results)
    bst2, results2 = xgb_rmse(dtrain, dtest)
    return bst, results, base_data,bst2, results2

# %%
if __name__ == "__main__":
    bst, results, base_data = main()
    plot_decision_lines(bst, base_data, col=1)
