import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Weekly_U.S.Diesel_Retail_Prices.csv')
# print(data.head())

data['Week of'] = pd.to_datetime(data['Week of'])
data = data.sort_values(by='Week of', ascending=True, ignore_index=True)
# print(data.head())

data['year'] = data['Week of'].dt.strftime('%Y')
data['month'] = data['Week of'].dt.strftime('%m')
print(data.head())


sns.lineplot(data=data, x='year', y='Series ID: PET.EMD_EPD2D_PTE_NUS_DPG.W Dollars per Gallon')
plt.show()

sns.lineplot(data=data, x='month', y='Series ID: PET.EMD_EPD2D_PTE_NUS_DPG.W Dollars per Gallon')
plt.show()


# groupby 활용

