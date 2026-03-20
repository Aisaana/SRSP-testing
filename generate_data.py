import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 1000

data = {
    'lines_added': np.random.randint(1, 500, n_samples),       
    'lines_deleted': np.random.randint(0, 300, n_samples),     
    'files_changed': np.random.randint(1, 50, n_samples),      
    'hour_of_day': np.random.randint(0, 24, n_samples),        
    'is_weekend': np.random.choice([0, 1], n_samples),         
    'developer_experience': np.random.randint(1, 10, n_samples)
}

df = pd.DataFrame(data)
bug_probability = (
    (df['lines_added'] / 500) + 
    (df['files_changed'] / 50) - 
    (df['developer_experience'] / 10)
)

bug_probability += np.random.normal(0, 0.1, n_samples)

df['is_bug'] = (bug_probability > 0.5).astype(int)
df.to_csv('commits_dataset.csv', index=False)
print("Синтетическая база данных создана: commits_dataset.csv")
print(df.head())