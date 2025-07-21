import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TotalMonetaryAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X['TOTAL_SPENDING'] = X['PURCHASES'] + X['CASH_ADVANCE']
        return X

class FixPurchases(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    
    def transform(self, X):
        X = X.copy()
        # Fix mismatched PURCHASES where sum doesn't match (with float tolerance)
        mask = ~np.isclose(X['PURCHASES'], X['ONEOFF_PURCHASES'] + X['INSTALLMENTS_PURCHASES'])
        X.loc[mask, 'PURCHASES'] = X.loc[mask, 'ONEOFF_PURCHASES'] + X.loc[mask, 'INSTALLMENTS_PURCHASES']
        return X

# --- Custom Data Cleaning ---
class CustomImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = X.copy()
        # === Special cases ===
        self.credit_limit_median = X['CREDIT_LIMIT'].median()
        self.min_pay_median = X.loc[X['PAYMENTS'] != 0, 'MINIMUM_PAYMENTS'].median()
        # === Mode columns (bounded/categorical-like behavior) ===
        self.mode_cols = [
            'BALANCE_FREQUENCY', 'PURCHASES_FREQUENCY',
            'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
            'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX','PURCHASES_TRX','TENURE','PRC_FULL_PAYMENT']
        self.modes = {col: X[col].mode()[0] for col in self.mode_cols}
        # === Median columns (skewed numeric) ===
        self.median_cols = [
            'BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES',
            'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
            'PAYMENTS']
        self.medians = {col: X[col].median() for col in self.median_cols}
        return self
    def transform(self, X):
        X = X.copy()
        # === Impute CREDIT_LIMIT ===
        X['CREDIT_LIMIT'] = X['CREDIT_LIMIT'].fillna(self.credit_limit_median)
        # === Impute MINIMUM_PAYMENTS with condition ===
        X['MINIMUM_PAYMENTS'] = X.apply(
            lambda row: 0 if pd.isna(row['MINIMUM_PAYMENTS']) and row['PAYMENTS'] == 0
            else self.min_pay_median if pd.isna(row['MINIMUM_PAYMENTS'])
            else row['MINIMUM_PAYMENTS'],
            axis=1)
        # === Mode imputation for frequency columns ===
        for col in self.mode_cols:
            X[col] = X[col].fillna(self.modes[col])
        # === Median imputation for continuous/skewed cols ===
        for col in self.median_cols:
            X[col] = X[col].fillna(self.medians[col])
        return X