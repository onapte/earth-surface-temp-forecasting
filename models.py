import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error as mse
import math

class MLModel:
    def __init__(self, model, X_train, X_test, y_train, y_test):
        
        self.model = model
        self.model_name = type(self.model).__name__
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred_train = None
        self.y_pred_test = None
        
        self.test_stats = {
            'R2 score': 0.0,
            'MAE': 0.0,
            'MSE': 0.0,
            'RMSE': 0.0,
        }
        
    def fit(self):
        self.model.fit(self.X_train, self.y_train)
        
        self.y_pred_train = self.model.predict(self.X_train)
        self.y_pred_test = self.model.predict(self.X_test)
        
        self.plot(best_fit=True)
        
    def plot(self, best_fit=False):
        plt.scatter(self.y_test, self.y_pred_test, color='purple', alpha=0.3, label='Ground Truth Vs Predictions')
        
        if best_fit:
            regressor = LinearRegression()
            regressor.fit(self.y_test.to_numpy().reshape(-1, 1), self.y_pred_test)
            y_line = regressor.predict(self.y_test.to_numpy().reshape(-1, 1))
            
            plt.plot(self.y_test, y_line, color='red', linewidth=2, label='Best-fit Line')
            
        plt.xlabel('Ground Truth Values (y_test)')
        plt.ylabel('Predicted Values (y_pred)')
        plt.title(f'{self.model_name}: True Vs Predicted')
        plt.legend()
        
        plt.show()
        
    def show_test_statistics(self):
        r2_val = self.model.score(self.X_test, self.y_test)
        mae_val = mean_absolute_error(self.y_test, self.y_pred_test)
        mse_val = mse(self.y_test, self.y_pred_test)
        rmse_val = rmse(self.y_test, self.y_pred_test)
        
        self.test_stats['R2 score'] = r2_val
        self.test_stats['MAE'] = mae_val
        self.test_stats['MSE'] = mse_val
        self.test_stats['RMSE'] = rmse_val
        
        print("Test Statistics")
        for stat in self.test_stats:
            print(f'{stat}: {self.test_stats[stat]:.3f}')
