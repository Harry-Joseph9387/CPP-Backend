```python

```


```python
# pip install catboost category_encoders --upgrade --force-reinstall numpy==1.23.5
```


```python

from google.colab import files
files.upload()
```


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from category_encoders.binary import BinaryEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings("ignore")
pd.pandas.set_option("display.max_columns", None)
# Create Dataframe
df = pd.read_csv("cardekho_dataset.csv")

# Print shape of dataset
print(df.shape)
```

    (15411, 14)


* Handling Missing values
* Handling Duplicates
* Check data type
* Understand the dataset


```python
df.isnull().sum()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Unnamed: 0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>car_name</th>
      <td>0</td>
    </tr>
    <tr>
      <th>brand</th>
      <td>0</td>
    </tr>
    <tr>
      <th>model</th>
      <td>0</td>
    </tr>
    <tr>
      <th>vehicle_age</th>
      <td>0</td>
    </tr>
    <tr>
      <th>km_driven</th>
      <td>0</td>
    </tr>
    <tr>
      <th>seller_type</th>
      <td>0</td>
    </tr>
    <tr>
      <th>fuel_type</th>
      <td>0</td>
    </tr>
    <tr>
      <th>transmission_type</th>
      <td>0</td>
    </tr>
    <tr>
      <th>mileage</th>
      <td>0</td>
    </tr>
    <tr>
      <th>engine</th>
      <td>0</td>
    </tr>
    <tr>
      <th>max_power</th>
      <td>0</td>
    </tr>
    <tr>
      <th>seats</th>
      <td>0</td>
    </tr>
    <tr>
      <th>selling_price</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>




```python
df.duplicated().sum()
```




    0



#### Brand and model column can be dropped as the information is already available in car_name


```python
df.drop(['brand','model', 'Unnamed: 0'], axis=1, inplace=True)
```

### Type of Features


```python
num_features = [feature for feature in df.columns if df[feature].dtype != 'O']
cat_features = [feature for feature in df.columns if df[feature].dtype == 'O']
discrete_features=[feature for feature in num_features if len(df[feature].unique())<=25]
continuous_features=[feature for feature in num_features if feature not in discrete_features]

print(" Continuous Features Count {} \n Discrete Features Count {} \n Numerical features Count {} \n Categorical features Count {} \n ".format(len(continuous_features),len(discrete_features),len(num_features),len(cat_features)))
```

     Continuous Features Count 5 
     Discrete Features Count 2 
     Numerical features Count 7 
     Categorical features Count 4 
     


####Detecting Outlier and Capping it


```python
def detect_outliers(col):
    # Finding the IQR
    percentile25 = df[col].quantile(0.25)
    percentile75 = df[col].quantile(0.75)
    print('\n ####', col , '####')
    print("percentile25",percentile25)
    print("percentile75",percentile75)
    iqr = percentile75 - percentile25
    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr
    print("Upper limit",upper_limit)
    print("Lower limit",lower_limit)
    df.loc[(df[col]>upper_limit), col]= upper_limit
    df.loc[(df[col]<lower_limit), col]= lower_limit
    return df
for col in continuous_features:
  detect_outliers(col)
```

    
     #### km_driven ####
    percentile25 30000.0
    percentile75 70000.0
    Upper limit 130000.0
    Lower limit -30000.0
    
     #### mileage ####
    percentile25 17.0
    percentile75 22.7
    Upper limit 31.25
    Lower limit 8.450000000000001
    
     #### engine ####
    percentile25 1197.0
    percentile75 1582.0
    Upper limit 2159.5
    Lower limit 619.5
    
     #### max_power ####
    percentile25 74.0
    percentile75 117.3
    Upper limit 182.25
    Lower limit 9.050000000000011
    
     #### selling_price ####
    percentile25 385000.0
    percentile75 825000.0
    Upper limit 1485000.0
    Lower limit -275000.0


####Checking Skewness after Outlier Capping


```python
df[continuous_features].skew(axis=0, skipna=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>km_driven</th>
      <td>0.617437</td>
    </tr>
    <tr>
      <th>mileage</th>
      <td>0.067940</td>
    </tr>
    <tr>
      <th>engine</th>
      <td>0.684096</td>
    </tr>
    <tr>
      <th>max_power</th>
      <td>1.067229</td>
    </tr>
    <tr>
      <th>selling_price</th>
      <td>0.968836</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>




```python
from sklearn.model_selection import train_test_split
X = df.drop(['selling_price'], axis=1)
y = df['selling_price']
```

Encoding Categorical values into integer values


```python
# Create Column Transformer with 3 types of transformers
num_features = X.select_dtypes(exclude="object").columns
onehot_columns = ['seller_type','fuel_type','transmission_type']
binary_columns = ['car_name']

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders.binary import BinaryEncoder
from sklearn.compose import ColumnTransformer

numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder()
binary_transformer = BinaryEncoder()

preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", oh_transformer, onehot_columns),
         ("StandardScaler", numeric_transformer, num_features),
        ("BinaryEncoder", binary_transformer, binary_columns)

    ]
)
```


```python
X= preprocessor.fit_transform(X)
```


```python
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape, X_test.shape
```




    ((12328, 23), (3083, 23))




```python
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
```


```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import numpy as np

def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square  # ✅ Fixed return statement

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "XGBRegressor": XGBRegressor(),
    "CatBoosting Regressor": CatBoostRegressor(silent=True),  # ✅ Fixed verbose issue
    "AdaBoost Regressor": AdaBoostRegressor()
}

model_list = []
r2_list = []

for name, model in models.items():
    model.fit(X_train, y_train)  # Train model

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluate performance
    model_train_mae, model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
    model_test_mae, model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

    # Store results
    model_list.append(name)
    r2_list.append(model_test_r2)

    # Print results
    print(f"{name}")
    print("Model performance for Training set:")
    print(f"- Root Mean Squared Error: {model_train_rmse:.4f}")
    print(f"- Mean Absolute Error: {model_train_mae:.4f}")
    print(f"- R2 Score: {model_train_r2:.4f}")
    print('----------------------------------')
    print("Model performance for Test set:")
    print(f"- Root Mean Squared Error: {model_test_rmse:.4f}")
    print(f"- Mean Absolute Error: {model_test_mae:.4f}")
    print(f"- R2 Score: {model_test_r2:.4f}")
    print("=" * 35, "\n")

```

    Linear Regression
    Model performance for Training set:
    - Root Mean Squared Error: 148767.4017
    - Mean Absolute Error: 112822.8151
    - R2 Score: 0.8380
    ----------------------------------
    Model performance for Test set:
    - Root Mean Squared Error: 155478.8345
    - Mean Absolute Error: 118417.1508
    - R2 Score: 0.8343
    =================================== 
    
    Lasso
    Model performance for Training set:
    - Root Mean Squared Error: 148767.4172
    - Mean Absolute Error: 112822.8031
    - R2 Score: 0.8380
    ----------------------------------
    Model performance for Test set:
    - Root Mean Squared Error: 155479.5196
    - Mean Absolute Error: 118418.6806
    - R2 Score: 0.8343
    =================================== 
    
    Ridge
    Model performance for Training set:
    - Root Mean Squared Error: 148770.8205
    - Mean Absolute Error: 112835.0850
    - R2 Score: 0.8380
    ----------------------------------
    Model performance for Test set:
    - Root Mean Squared Error: 155485.9482
    - Mean Absolute Error: 118436.6096
    - R2 Score: 0.8343
    =================================== 
    
    K-Neighbors Regressor
    Model performance for Training set:
    - Root Mean Squared Error: 81895.5371
    - Mean Absolute Error: 54735.2369
    - R2 Score: 0.9509
    ----------------------------------
    Model performance for Test set:
    - Root Mean Squared Error: 102838.6220
    - Mean Absolute Error: 68701.9543
    - R2 Score: 0.9275
    =================================== 
    
    Decision Tree
    Model performance for Training set:
    - Root Mean Squared Error: 19892.7304
    - Mean Absolute Error: 5386.6533
    - R2 Score: 0.9971
    ----------------------------------
    Model performance for Test set:
    - Root Mean Squared Error: 121519.9083
    - Mean Absolute Error: 78700.0135
    - R2 Score: 0.8988
    =================================== 
    
    Random Forest Regressor
    Model performance for Training set:
    - Root Mean Squared Error: 39145.4487
    - Mean Absolute Error: 26188.8776
    - R2 Score: 0.9888
    ----------------------------------
    Model performance for Test set:
    - Root Mean Squared Error: 99026.7517
    - Mean Absolute Error: 65822.5171
    - R2 Score: 0.9328
    =================================== 
    
    XGBRegressor
    Model performance for Training set:
    - Root Mean Squared Error: 63212.0596
    - Mean Absolute Error: 46145.6406
    - R2 Score: 0.9708
    ----------------------------------
    Model performance for Test set:
    - Root Mean Squared Error: 93184.7308
    - Mean Absolute Error: 63641.7109
    - R2 Score: 0.9405
    =================================== 
    
    CatBoosting Regressor
    Model performance for Training set:
    - Root Mean Squared Error: 74608.5033
    - Mean Absolute Error: 54468.1250
    - R2 Score: 0.9593
    ----------------------------------
    Model performance for Test set:
    - Root Mean Squared Error: 90935.6072
    - Mean Absolute Error: 63175.7719
    - R2 Score: 0.9433
    =================================== 
    
    AdaBoost Regressor
    Model performance for Training set:
    - Root Mean Squared Error: 162367.6884
    - Mean Absolute Error: 132690.2967
    - R2 Score: 0.8070
    ----------------------------------
    Model performance for Test set:
    - Root Mean Squared Error: 170919.0064
    - Mean Absolute Error: 138907.9950
    - R2 Score: 0.7998
    =================================== 
    



```python
pd.DataFrame(list(zip(model_list, r2_list*100)), columns=['Model Name', 'R2_Score']).sort_values(by=["R2_Score"],ascending=False)
```





  <div id="df-19805419-bce8-435d-be15-dee4013dbdd2" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model Name</th>
      <th>R2_Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>CatBoosting Regressor</td>
      <td>0.943329</td>
    </tr>
    <tr>
      <th>6</th>
      <td>XGBRegressor</td>
      <td>0.940491</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Random Forest Regressor</td>
      <td>0.932796</td>
    </tr>
    <tr>
      <th>3</th>
      <td>K-Neighbors Regressor</td>
      <td>0.927522</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Decision Tree</td>
      <td>0.898798</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Linear Regression</td>
      <td>0.834333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Lasso</td>
      <td>0.834332</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ridge</td>
      <td>0.834318</td>
    </tr>
    <tr>
      <th>8</th>
      <td>AdaBoost Regressor</td>
      <td>0.799796</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-19805419-bce8-435d-be15-dee4013dbdd2')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-19805419-bce8-435d-be15-dee4013dbdd2 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-19805419-bce8-435d-be15-dee4013dbdd2');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-f2d15803-07b5-47dc-9c0f-cd7c95c64403">
  <button class="colab-df-quickchart" onclick="quickchart('df-f2d15803-07b5-47dc-9c0f-cd7c95c64403')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-f2d15803-07b5-47dc-9c0f-cd7c95c64403 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




Getting default values to fill for Null Inputs


```python
X = df.drop(['selling_price'], axis=1)
default_values = {}
numerical_features = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
categorical_features = ['car_name', 'seller_type', 'fuel_type', 'transmission_type']
for feature in numerical_features:
    default_values[feature] = X[feature].median()  # Use median for numerical

for feature in categorical_features:
    default_values[feature] = X[feature].mode()[0]
for key, value in default_values.items():
    print(f"{key}: {value}\n")
```

    vehicle_age: 6.0
    
    km_driven: 50000.0
    
    mileage: 19.67
    
    engine: 1248.0
    
    max_power: 88.5
    
    seats: 5.0
    
    car_name: Hyundai i20
    
    seller_type: Dealer
    
    fuel_type: Petrol
    
    transmission_type: Manual
    


####Performing prediction on user input


```python
user_inputs = [
    {
        "car_name": "Hyundai Grand",
        "vehicle_age": 5,
        "km_driven": 20000,
        "seller_type": "Individual",
        "fuel_type": "Petrol",
        "transmission_type": "Manual",
        "mileage": 18.9,
        "engine": 1197.0,
        "max_power": 82.0,
        "seats": 5
    },
    {
        "car_name": "Maruti Swift",
        "vehicle_age": 3,
        "km_driven": 35000,
        "seller_type": "Dealer",
        "fuel_type": "Petrol",
        "transmission_type": "Automatic",
        "mileage": 20.4,
        "engine": 1197.0,
        "max_power": 90.0,
        "seats": 5
    },
    {
        "car_name": "Mahindra XUV500",
        "vehicle_age": 7,
        "km_driven": 80000,
        "seller_type": "Individual",
        "fuel_type": "Diesel",
        "transmission_type": "Manual",
        "mileage": 16.0,
        "engine": 2179.0,
        "max_power": 140.0,
        "seats": 7
    },
    {
        "car_name": "Tata Nexon EV",
        "vehicle_age": 1,
        "km_driven": 10000,
        "seller_type": "Dealer",
        "fuel_type": "Electric",
        "transmission_type": "Automatic",
        "mileage": None,  # Electric vehicles don't have mileage in km/l
        "engine": None,    # No engine, only motor
        "max_power": 129.0,
        "seats": 5
    }
]

for user_input in user_inputs:
  for i in user_input:
    if user_input[i]==None:
      user_input[i]=default_values[i]
  user_input=pd.DataFrame([user_input])
  preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", oh_transformer, onehot_columns),
         ("StandardScaler", numeric_transformer, num_features),
        ("BinaryEncoder", binary_transformer, binary_columns)

    ]
)
  X = df.drop(['selling_price'], axis=1)
  preprocessor.fit(X)
  X=preprocessor.transform(X)
  user_input=preprocessor.transform(user_input)
  predicted = model.predict(user_input)
  print(predicted)
```

    [558945.92914854]
    [723148.63921842]
    [933281.1434566]
    [996637.13080169]



