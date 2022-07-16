{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "lZVMoIAaCFbr"
      },
      "source": [
        "Contents:\n",
        "\n",
        "1. Module Imports\n",
        "2. Reading Data into the Notebook\n",
        "3. Train - Valid - Test split\n",
        "4. Data Pre-processing\n",
        "5. Hyperparameter tuning\n",
        "6. Evaluating test set accuracy with the trained model\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 47,
      "metadata": {
        "id": "SZjIO__cCFbv"
      },
      "outputs": [],
      "source": [
        "'''Import all necessary packages...pandas for data munging, \n",
        "sklearn's data preprocessing module, seaborn for data visualization\n",
        "%matplotlib for making any plots show up inside of this notebook.\n",
        "joblib for saving models (serialize/deserialize module). We will discuss\n",
        "each of these packages as we use them for our analysis'''\n",
        "\n",
        "import pandas as pd\n",
        "from pandas.api.types import is_string_dtype,is_numeric_dtype\n",
        "from sklearn import preprocessing\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.metrics import confusion_matrix,accuracy_score,matthews_corrcoef,f1_score\n",
        "# from sklearn.externals import joblib\n",
        "from matplotlib import pyplot as plt\n",
        "import seaborn as sns\n",
        "import joblib\n",
        "\n",
        "# magic command using % for unix to have plots within the notebook\n",
        "%matplotlib inline "
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "g1QnwFdpCFby"
      },
      "source": [
        "# II. Reading data into the notebook"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 48,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 161
        },
        "id": "herT99mACFby",
        "outputId": "02dcdeb5-6301-4b80-8049-ad2c92a98eb7"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(1309, 12)\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   PassengerId  Survived  Pclass  \\\n",
              "0            1         0       3   \n",
              "1            2         1       1   \n",
              "2            3         1       3   \n",
              "\n",
              "                                                Name     Sex   Age  SibSp  \\\n",
              "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
              "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
              "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
              "\n",
              "   Parch            Ticket     Fare Cabin Embarked  \n",
              "0      0         A/5 21171   7.2500   NaN        S  \n",
              "1      0          PC 17599  71.2833   C85        C  \n",
              "2      0  STON/O2. 3101282   7.9250   NaN        S  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-196cab6a-775c-443f-a0ad-cad84da041a2\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Name</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Ticket</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Cabin</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Braund, Mr. Owen Harris</td>\n",
              "      <td>male</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>A/5 21171</td>\n",
              "      <td>7.2500</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
              "      <td>female</td>\n",
              "      <td>38.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>PC 17599</td>\n",
              "      <td>71.2833</td>\n",
              "      <td>C85</td>\n",
              "      <td>C</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>Heikkinen, Miss. Laina</td>\n",
              "      <td>female</td>\n",
              "      <td>26.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>STON/O2. 3101282</td>\n",
              "      <td>7.9250</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-196cab6a-775c-443f-a0ad-cad84da041a2')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-196cab6a-775c-443f-a0ad-cad84da041a2 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-196cab6a-775c-443f-a0ad-cad84da041a2');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 48
        }
      ],
      "source": [
        "'''We will use the Titanic dataset, as supplied here (Titanic_full.csv).\n",
        "The same dataset can be downloaded from the Kaggle website. Check the\n",
        "shape and header of the data you just read in as a dataframe. The use of\n",
        "f and {} is specific to the recent versions of Python (3.x). You can as \n",
        "well type the full path here, and that works too!'''\n",
        "\n",
        "# my_df = pd.read_csv(f'{my_path}/Titanic_full.csv')\n",
        "my_df = pd.read_csv(\"https://raw.githubusercontent.com/adityamulik/INFO-6105---Data-Science-Engineering-And-Tools/main/Data/Titanic_full.csv\")\n",
        "print(my_df.shape)\n",
        "my_df.head(3)\n",
        "\n",
        "# 12 dimensions dataset\n",
        "# survived col is called as the y var / target / dependent var \n",
        "# rest of the variables are x var / independent\n",
        "# survived is what we are trying to predict"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "APxWmCSRCFbz"
      },
      "source": [
        "# III. Train - Valid- Test Split"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 49,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "w5pacN3SCFb0",
        "outputId": "776dbe55-a372-44b6-9188-dbe9f6b4bb24"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(1100, 12) (209, 12)\n"
          ]
        }
      ],
      "source": [
        "'''As we have seen, it's important to avoid the Texas sharp shooter logical\n",
        "fallacy. So, we plit the data into three sets in a 70-15-15 manner. \n",
        "This means 70% of the data rows go into building or training the model. \n",
        "This 70% is often called a training set. 15% of the data goes into \n",
        "evaluating model performance as you manually change or set the model \n",
        "hyperparameters (e.g. Value of K is a hyperparameter,in K-NN algorithm). \n",
        "This dataset is sometimes called a holdout set or the validation set. \n",
        "Finally, the last 15% of the data is the test set.  This dataset is \n",
        "never \"seen\" by the model for model building or hyperparamter tuning. After\n",
        "hyperparameter tuning and model selection, which we will discuss later,\n",
        "the model's final performance before sneidng it to production, will be\n",
        "evalauted on this test data set.\n",
        "The way the datarows are distributed will depend on the type of problem. \n",
        "Here, assuming the datarows of my_df are randomly arranged, and there's\n",
        "no time component, we will simply simply do the split, using a split \n",
        "function that we define. Also, this 15% split \n",
        "for the test set may change, if the actual test set size is predefined. \n",
        "In any case we will make sure the test set we choose from our dataset \n",
        "is as similar as it can get to the actual data that it will see in \n",
        "production'''\n",
        "\n",
        "#Calculate 15% of 1309. This is about 209 rows of data. \n",
        "#So, 1309 - 209 = 1100 rows of data will remain for the\n",
        "#train and valid sets which we will separate later. 200 rows of data will\n",
        "#go as test set data\n",
        "\n",
        "def mydf_splitter(my_df,num_rows):\n",
        "    return my_df[:num_rows].copy(),my_df[num_rows:]\n",
        "\n",
        "\n",
        "mydf_train_valid,mydf_test = mydf_splitter(my_df,1100)\n",
        "\n",
        "print(mydf_train_valid.shape,mydf_test.shape)\n",
        "\n",
        "#We are going to put away the mydf_test for now. \n",
        "#We will return to it later.\n",
        "\n",
        "# millions of data for actual ML model"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "dFIT094FCFb3"
      },
      "source": [
        "# IV. Data Pre-processing"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 50,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 446
        },
        "id": "bzfuh4_SCFb4",
        "outputId": "a90ceb9a-04a3-4fa7-bd5a-9eb88a536261"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7f8a3fca1850>"
            ]
          },
          "metadata": {},
          "execution_count": 50
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 576x432 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAfEAAAGbCAYAAADUXalBAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3debgkZXX48e+ZGZB9HwEBZRXFBZQBxR0QZRMQUHHBQdHRRDZBBPNL1BiNxribRJ2IcUxUXIIBV0AElxiXATHggiiBKGEZg4A7IOf3x3lvprnemekZbnXfmvl+nuc+t6t6qdPd1XXerd6KzESSJPXPrHEHIEmSVo1JXJKknjKJS5LUUyZxSZJ6yiQuSVJPmcQlSeqpOV2+eES8HHgRkMAVwAuArYGzgc2BS4FjM/OO5b3OFltskdtvv32XoUqSNGNceumlP8/MuSt6XHR1nnhEbAN8DdgtM38bER8HPgccDJyTmWdHxHuB72bme5b3WvPmzcvFixd3EqckSTNNRFyamfNW9Lium9PnAOtGxBxgPeAGYD/gk+3+RcARHccgSdJqqbMknpnXA28B/ptK3rdRzee3ZuZd7WE/A7bpKgZJklZnnSXxiNgUOBzYAbgfsD5w4Eo8f0FELI6IxUuWLOkoSkmS+qvL5vQnA/+VmUsy807gHOCxwCateR1gW+D6qZ6cmQszc15mzps7d4V9+5IkrXG6TOL/DTw6ItaLiAD2B74PXAwc3R4zHzi3wxgkSVptddkn/k1qANtl1Olls4CFwBnAqRHxY+o0s7O6ikGSpNVZp+eJZ+ZrgNdMWn0NsHeX25UkaU3gjG2SJPWUSVySpJ4yiUuS1FMmcUmSesokLklST5nEJUnqqU5PMZMkaXmufceNI9/m9qdsNfJtdsWauCRJPWUSlySpp0zikiT1lElckqSeMolLktRTJnFJknrKJC5JUk+ZxCVJ6imTuCRJPWUSlySpp0zikiT1lElckqSeMolLktRTJnFJknrKJC5JUk+ZxCVJ6imTuCRJPWUSlySpp0zikiT1lElckqSeMolLktRTJnFJknrKJC5JUk+ZxCVJ6imTuCRJPdVZEo+IXSPi8oG/2yPilIjYLCIujIir2/9Nu4pBkqTVWWdJPDOvysw9MnMPYE/gN8CngDOBizJzF+CitixJklbSqJrT9wd+kpnXAYcDi9r6RcARI4pBkqTVyqiS+DHAR9vtLTPzhnb7RmDLqZ4QEQsiYnFELF6yZMkoYpQkqVc6T+IRsTZwGPCJyfdlZgI51fMyc2FmzsvMeXPnzu04SkmS+mcUNfGDgMsy86a2fFNEbA3Q/t88ghgkSVrtjCKJP5ulTekA5wHz2+35wLkjiEGSpNVOp0k8ItYHDgDOGVj9JuCAiLgaeHJbliRJK2lOly+emb8GNp+07n+p0eqSJOlecMY2SZJ6yiQuSVJPmcQlSeopk7gkST1lEpckqadM4pIk9ZRJXJKknjKJS5LUUyZxSZJ6yiQuSVJPmcQlSeopk7gkST1lEpckqadM4pIk9ZRJXJKknjKJS5LUUyZxSZJ6yiQuSVJPmcQlSeopk7gkST1lEpckqadM4pIk9ZRJXJKknjKJS5LUUyZxSZJ6yiQuSVJPmcQlSeopk7gkST1lEpckqadM4pIk9ZRJXJKknuo0iUfEJhHxyYj4YUT8ICL2iYjNIuLCiLi6/d+0yxgkSVpddV0Tfyfwhcx8ELA78APgTOCizNwFuKgtS5KkldRZEo+IjYEnAGcBZOYdmXkrcDiwqD1sEXBEVzFIkrQ667ImvgOwBPiniPhORLw/ItYHtszMG9pjbgS2nOrJEbEgIhZHxOIlS5Z0GKYkSf3UZRKfAzwSeE9mPgL4NZOazjMzgZzqyZm5MDPnZea8uXPndhimJEn91GUS/xnws8z8Zlv+JJXUb4qIrQHa/5s7jEGSpNVWZ0k8M28EfhoRu7ZV+wPfB84D5rd184Fzu4pBkqTV2ZyOX/9E4MMRsTZwDfACquDw8Yg4HrgOeGbHMUiStFrqNIln5uXAvCnu2r/L7UqStCZwxjZJknrKJC5JUk+ZxCVJ6imTuCRJPWUSlySpp0zikiT1lElckqSeMolLktRTJnFJknrKJC5JUk+ZxCVJ6imTuCRJPWUSlySpp0zikiT1lElckqSeMolLktRTJnFJknrKJC5JUk+ZxCVJ6imTuCRJPWUSlySpp0zikiT1lElckqSeMolLktRTJnFJknrKJC5JUk+ZxCVJ6imTuCRJPWUSlySpp0zikiT1lElckqSemtPli0fEtcAvgT8Ad2XmvIjYDPgYsD1wLfDMzPxFl3FIkrQ6GkVNfN/M3CMz57XlM4GLMnMX4KK2LEmSVtI4mtMPBxa124uAI8YQgyRJvdd1Ek/ggoi4NCIWtHVbZuYN7faNwJYdxyBJ0mpphX3iEXHkFKtvA67IzJtX8PTHZeb1EXFf4MKI+OHgnZmZEZHL2O4CYAHA/e9//xWFKUnSGmeYgW3HA/sAF7flJwGXAjtExOsy85+X9cTMvL79vzkiPgXsDdwUEVtn5g0RsTUwZUEgMxcCCwHmzZs3ZaKXJGlNNkxz+hzgwZl5VGYeBexGNZM/CjhjWU+KiPUjYsOJ28BTgCuB84D57WHzgXNXPXxJktZcw9TEt8vMmwaWb27rbomIO5fzvC2BT0XExHY+kplfiIhvAx+PiOOB64BnrmLskiSt0YZJ4pdExGeAT7Tlo9q69YFbl/WkzLwG2H2K9f8L7L8KsUqSpAHDJPGXUYn7sW35Q8C/ZmYC+3YVmCRJWr4VJvGWrD/Z/iRJ0gyxwoFtEXFkRFwdEbdFxO0R8cuIuH0UwUmSpGUbpjn9zcDTMvMHXQcjSZKGN8wpZjeZwCVJmnmGqYkvjoiPAf8G/H5iZWae01lUkiRphYZJ4hsBv6Ema5mQgElckqQxGmZ0+gtGEYgkSVo5y0ziEfHKzHxzRLybqnnfQ2ae1GlkkiRpuZZXE58YzLZ4FIFIkqSVs8wknpmfbv8XTayLiFnABpnpeeKSJI3ZMJO9fCQiNmpzpV8JfD8iTu8+NEmStDzDnCe+W6t5HwF8HtgBOLbTqCRJ0goNk8TXioi1qCR+XmbeyRQD3SRJ0mgNk8TfB1wLrA98JSIeANgnLknSmA1znvi7gHcNrLouIrwEqSRJYzbMwLaT28C2iIizIuIyYL8RxCZJkpZjmOb0F7aBbU8BNqUGtb2p06gkSdIKDZPEo/0/GPjnzPzewDpJkjQmwyTxSyPiAiqJnx8RGwJ3dxuWJElakWGuYnY8sAdwTWb+JiI2B7woiiRJY7a8C6A8KDN/SCVwgB0jbEWXJGmmWF5N/FRgAfDWKe5LHKEuSdJYLe8CKAvaf88JlyRpBlphn3hEzAYOAbYffHxmvq27sCRJ0ooMM7Dt08DvgCtwVLokSTPGMEl828x8eOeRSJKklTLMeeKfj4indB6JJElaKcPUxL8BfCoiZgF3UrO1ZWZu1GlkkiRpuYZJ4m8D9gGuyEyvIy5J0gwxTHP6T4ErTeCSJM0sw9TErwEuiYjPA7+fWDnsKWbtFLXFwPWZeWhE7ACcDWwOXAocm5l3rHTkkiSt4Yapif8XcBGwNrDhwN+wTgZ+MLD8N8DbM3Nn4BfU3OySJGklrbAmnpl/uaovHhHbUhPFvAE4NWry9f2A57SHLAJeC7xnVbchSdKaapia+L3xDuCVLJ0kZnPg1sy8qy3/DNim4xgkSVotdZbEI+JQ4ObMvHQVn78gIhZHxOIlS5ZMc3SSJPXfCpN4RDx2mHVTeCxwWERcSw1k2w94J7BJREw0428LXD/VkzNzYWbOy8x5c+fOHWJzkiStWYapib97yHX3kJmvysxtM3N74BjgS5n5XOBi4Oj2sPnAuUPGKkmSBixzYFtE7AM8BpgbEacO3LURMPtebPMM4OyIeD3wHeCse/FakiStsZY3On1tYIP2mMFTym5naU16KJl5CXBJu30NsPfKPF+SJP2xZSbxzPwy8OWI+GBmXjfCmCRJ0hCGmbHtPhGxENh+8PGZuV9XQUmSpBUbJol/Angv8H7gD92GI0mShjVMEr8rM51RTZKkGWaYU8w+HRF/GhFbR8RmE3+dRyZJkpZrmJr4/Pb/9IF1Cew4/eFIkqRhDXMBlB1GEYgkSVo5w0y7ul5E/HkboU5E7NLmRZckSWM0TJ/4PwF3ULO3Qc11/vrOIpIkSUMZJonvlJlvBu4EyMzfANFpVJIkaYWGSeJ3RMS61GA2ImIn4PedRiVJklZomNHprwG+AGwXER+mLjF6XJdBSZKkFRtmdPqFEXEZ8GiqGf3kzPx555FJkqTlGmZ0+tOpWds+m5mfAe6KiCO6D02SJC3PMH3ir8nM2yYWMvNWqoldkiSN0TBJfKrHDNOXLkmSOjRMEl8cEW+LiJ3a39uAS7sOTJIkLd8wSfxEarKXjwFnA78DXtZlUJIkacWW2yweEbOBz2TmviOKR5IkDWm5NfHM/ANwd0RsPKJ4JEnSkIYZoPYr4IqIuBD49cTKzDyps6gkSdIKDZPEz2l/kiRpBhlmxrZFbe70+2fmVSOISZIkDWGYGdueBlxOzZ9OROwREed1HZgkSVq+YU4xey2wN3ArQGZeDuzYYUySJGkIwyTxOwenXW3u7iIYSZI0vGEGtn0vIp4DzI6IXYCTgK93G5YkSVqRYWdsewjwe+AjwG3AKV0GJUmSVmyZNfGIWAd4KbAzcAWwT2beNarAJEnS8i2vJr4ImEcl8IOAt4wkIkmSNJTl9YnvlpkPA4iIs4BvjSYkSZI0jOXVxO+cuGEzuiRJM8/yauK7R8Tt7XYA67blADIzN1reC7c+9a8A92nb+WRmviYidqAuabo5dV3yYzPzjnv5PiRJWuMssyaembMzc6P2t2Fmzhm4vdwE3vwe2C8zdwf2AA6MiEcDfwO8PTN3Bn4BHD8db0SSpDXNMKeYrZIsv2qLa7W/BPYDPtnWLwKO6CoGSZJWZ50lcYCImB0RlwM3AxcCPwFuHehj/xmwzTKeuyAiFkfE4iVLlnQZpiRJvdRpEs/MP2TmHsC21PzrD1qJ5y7MzHmZOW/u3LmdxShJUl91msQnZOatwMXAPsAmETExoG5b4PpRxCBJ0uqmsyQeEXMjYpN2e13gAOAHVDI/uj1sPnBuVzFIkrQ6G+YCKKtqa2BRRMymCgsfz8zPRMT3gbMj4vXAd4CzOoxBkqTVVmdJPDP/E3jEFOuvofrHJUnSvTCSPnFJkjT9TOKSJPWUSVySpJ4yiUuS1FMmcUmSesokLklST5nEJUnqKZO4JEk9ZRKXJKmnTOKSJPWUSVySpJ4yiUuS1FMmcUmSesokLklST5nEJUnqKZO4JEk9ZRKXJKmnTOKSJPWUSVySpJ4yiUuS1FMmcUmSesokLklST5nEJUnqKZO4JEk9ZRKXJKmnTOKSJPWUSVySpJ4yiUuS1FMmcUmSesokLklST5nEJUnqqc6SeERsFxEXR8T3I+J7EXFyW79ZRFwYEVe3/5t2FYMkSauzLmvidwGnZeZuwKOBl0XEbsCZwEWZuQtwUVuWJEkrqbMknpk3ZOZl7fYvgR8A2wCHA4vawxYBR3QVgyRJq7OR9IlHxPbAI4BvAltm5g3trhuBLZfxnAURsTgiFi9ZsmQUYUqS1Ctzut5ARGwA/CtwSmbeHhH/d19mZkTkVM/LzIXAQoB58+ZN+RhpdXbIp/525Nv87NNPH/k2Ja26TmviEbEWlcA/nJnntNU3RcTW7f6tgZu7jEGSpNVVl6PTAzgL+EFmvm3grvOA+e32fODcrmKQJGl11mVz+mOBY4ErIuLytu7PgDcBH4+I44HrgGd2GIMkSautzpJ4Zn4NiGXcvX9X25UkaU3hjG2SJPWUSVySpJ4yiUuS1FMmcUmSesokLklST5nEJUnqKZO4JEk9ZRKXJKmnTOKSJPWUSVySpJ4yiUuS1FMmcUmSesokLklST5nEJUnqKZO4JEk9ZRKXJKmnTOKSJPWUSVySpJ4yiUuS1FMmcUmSesokLklST5nEJUnqKZO4JEk9ZRKXJKmnTOKSJPWUSVySpJ4yiUuS1FMmcUmSesokLklST5nEJUnqKZO4JEk91VkSj4gPRMTNEXHlwLrNIuLCiLi6/d+0q+1LkrS667Im/kHgwEnrzgQuysxdgIvasiRJWgWdJfHM/Apwy6TVhwOL2u1FwBFdbV+SpNXdqPvEt8zMG9rtG4EtR7x9SZJWG2Mb2JaZCeSy7o+IBRGxOCIWL1myZISRSZLUD6NO4jdFxNYA7f/Ny3pgZi7MzHmZOW/u3LkjC1CSpL4YdRI/D5jfbs8Hzh3x9iVJWm10eYrZR4H/AHaNiJ9FxPHAm4ADIuJq4MltWZIkrYI5Xb1wZj57GXft39U2JUlakzhjmyRJPWUSlySpp0zikiT1lElckqSeMolLktRTJnFJknrKJC5JUk+ZxCVJ6imTuCRJPWUSlySpp0zikiT1lElckqSeMolLktRTJnFJknrKJC5JUk+ZxCVJ6imTuCRJPWUSlySpp0zikiT1lElckqSeMolLktRTJnFJknrKJC5JUk+ZxCVJ6imTuCRJPWUSlySpp0zikiT1lElckqSeMolLktRTJnFJknrKJC5JUk+NJYlHxIERcVVE/DgizhxHDJIk9d3Ik3hEzAb+HjgI2A14dkTsNuo4JEnqu3HUxPcGfpyZ12TmHcDZwOFjiEOSpF6bM4ZtbgP8dGD5Z8CjxhCHJEl/5KZ3/sdYtrvlyfus9HPGkcSHEhELgAVt8VcRcdU0vOwWwM+n4XWm20yMy5iGs1rFFLxymkO5h9Xqs+qQMQ1n1WN6+fQGMsmqx3XKPZYeMMxTxpHErwe2G1jetq27h8xcCCyczg1HxOLMnDedrzkdZmJcxjQcYxreTIzLmIZjTMMbdVzj6BP/NrBLROwQEWsDxwDnjSEOSZJ6beQ18cy8KyJOAM4HZgMfyMzvjToOSZL6bix94pn5OeBzY9j0tDbPT6OZGJcxDceYhjcT4zKm4RjT8EYaV2TmKLcnSZKmidOuSpLUUybxHomIGHcMkqSZwyS+kiJis4hYd0ybn9VimD2m7UsaA3/zWhaT+EqIiPWA04DXjDKRR9kN+FFEbJyZfxjHj9qWgKn14XPpQ4zjNlMTZURsAjyy3X5SROw05pBmpKn28T7s9xGxS0Q8clWfbxJfOXcAFwFrA6dGxDqj2GiW7wNfBL4aERuMOpFHRGQbBRkRT4yIjUe17eXFNO4YoL4fgIh4fETsMu54Jj6XiNg2ItaNiHUzM2fK5wX3iHGLiLjvuOMByMw/AETEoyNibkRsNO6Ymm2A/SLik8DfATeMI4jB/adVaGaMScenrSNiFtRvcybt95O1HHIKMH9VE7lJfEgRMTsz76LObd8MOAw4qeududXCJ76ndwMBfCUiNhxlIh/4gZxAXYVu7Ae4gZgWRMTrI+J1EbHzqLY/kIjmRMRmwKuBsSekduA6CPhX4FXAv7SC34w5FaXFeBjwaeDCiDgjIh4yjlgiYl5EvL3dPh74KPB+4E9HuT8tS5tHY2PgYODszPwNjL4QO/h7A94WEadFxNivexERswZiOxm4gIrvhTBzE3mL+3fUJLDrAc+IiAev7OuYxIfUEuZjqB/3R4EvUFPGvqLLGnmrhd8dES+nkvgbgJuAyyJio1Em8og4ADgeeGJm/jQido+InSNibHPwR8SJwDOp7+OpwPNHte2BpDgrM28B/o3aJyZiG8vvKyIeDvw1cCzwO2ArqvA5cf/YD2gtYZ9C7U/PB+4PHDGmGt7/Ao+NiH8B5gF7AO8ANgCeP45EPsV39G7q+9woIk6MiM1actp0xHHNp76vdwMnAvuOcvtTycy7oVoIgV2BFwOXAfu0SseMTOQTcQPPo6Yifz5wRkQ8YmVexyS+HBGxXUQcHBH3aat2AD6RmedTB8nzgcdQibzrPvI9gL/JzLMz8yDgy8AlEzXyLjY4xU5/C3AuVUP5a+AjwF8Cj+5i+yvS4rsfcCCwF3Uwfl1rQt5gRDE8Bfh2RFwAPAc4NiIeEBFbUaXrcbgbeA8tMQLHZuZtEfGYiFhrHDXy9pkc1m5vBZxMJcn/yszvAu8CjgIOGmFMAZCZ/0UVBDcG9szM2zLzYuBCqvDz0ojYcVRxtZgmapYnRcSbgNcDnwW+DuwMHBURpwKnR01f3YlJTej3adv+E+p4dDXwlnbfyLvXImLPiNi1tVY+FLgYuDkzv0FNJvZpYPeIOB3uUeieMSLiccBJwCHUcfQ3wPMi4kHDvoZJfPn2pH48B7bla4CnRsSjMvP3mflZqqbzIKrfaloso8T4e+CBA8t/QyWwz3VRwpzUx/TSiDic6otLYEfqgPIY4BcMebWd6Yhp0qrZVM3334HHAk9rXR7PAw7q6nMZXM7MC6jE8yrgW1RrwIupAt5fRcSG0x3DcmLbJiLuB9xONe2fRbWaXNNqKSdTXUEj1Q5InwG2johNMvNGqhB6I/DCVqu8Cvg40/g7WkFMg/v346iD54nA2hHxDoDM/DKVGG4HfjmKuCbFuAA4HHgrcCjwqsz8N+ArwE5Uze2jmXlHR9sf/IxOarH8FPgQcFxmHtCm0T4VOLKLGFZgb+A2YMPMvBI4EzgtInbMzJ9Tn9MFwANG3WKxLFMck+6iKkfrZeZPqXzzZKoyssdQL5qZ/k3xx9LZ7I6lBrMdCaxPHQgXUol9N+BrwEOne7vt9tFUP9jjgV2ohHl0u+8Y4M+AHTr+HE4D/gPYY4r4jqRd0GbE380hVNJeG3gYcDnwJ+2+44AfADt1HMMC4HVUsrx/W7cDVbjZCHgwsMUI99NHURcSejVwH6pW+3XgWVRt/HLg8FF+Ty2u7YDvUgf9yfvPEcA/AB9s8f4Y2H/E8Z0IfAd4QFvenhpA+raBx6w7olhi0vLrqALyyVTN8j6T7t9oRHEdCZzdvsu9qBruccC6bf+6HHjwCL+zwX1oN6pAuGdbPgNYAjxo4jMCNhjlPjVk3Fu049cW1BijpwGbtfv+girQbj7U6477jc3Ev4ED44FtJ30+8KWWPHZrO/C3qeR+VEcxnEDVMJ9B1cJ3Bg6g+no+BPyk6x8OsGU7eKxH1eCOAF4BbELVDC4CHjaq76Pdng/8kOoDfxOwH/A44ApqrMK3gN06iGG9gdsntQP9E4DFwGupVq1NqRr4JiPeXw+mah0LgauAl1GtJU+mEvs/AAdP/ixHFNsjgbe327PbfvMa4J1tP3pK+219BDikPW7WiPalvdr3t+Wkxzyg7UdvGvFnNWvg+9wZeHvbnz4CrNPuOx142Qhj2hz4HvDFgXXHt+/vS+130PkxYOK7m2rfaPvTp4FHtOVTqS6lkVYuVuJ9nNa+009SLXentt/oQqoQ/i1WohIy9jc0U//aD/x9wOPb8jEtaT2tLa8PbDyxc03D9iYKDrOoJuJPUTWq04DPA2u3+zehSm9bdfCeJ9cE7kvVwt8HfIwqMV7RksTawNwRfA+DB92Ngb9q739L4M+BNwK7t89qE4Ysva5kDAdTA522a4nob9v/U6lCzhxgHaqw81Y6bh2ZFNvmbf/Yry0/tX1XZ0zxfY40gbdt7kk1Rx9BNal/Avgw1WLxnfaYY9rn9hKqaXQU+9IjqUFQf9+W1xv4DW5IjSe4/4g+o/2Bx7Xb6wBfbfvy3sCtwJHtvue0398DO4xlqiT5GOA64IyBdesCcxlRa0Db5lYDt4+hWlH2b8untN/iRIvhiV1+TvfiPTwDuKjdvpTW4kO1tr6k/Q5WqnI29jc10/7aAXk9qv/3223dxI/7GVQp6egOt79e+7+wJahzB9a9hA5qmYPvsd1+AlWTW4tqrj5hYsei+pvfB8wewXcxGNPpVE37GmCvtu6BVNPTe4DHdBTDoVRz8BETMbU4vgqcA6zV1i+gEtWcjj+TXdsBbLuBdQupgtXstvx8qu/y2LbcWc12Wd/bpO/uWKrW8UHgIQOf2QW0Ag81juBNjKAVg2pJ+wI1svpHtKbYgc/u9FF+ZsALqZrjE9ry11naRXME8A1gEdVsPG1dd1N9bwO3n0XVuCcKh3u1Y98rRrkvDcQzl2oROIRqIf0ONRjyg8A/srRQ/VXg4eOIccj3cSzVdXRC2wcnWli2aP9Xer8b+5uaKX8sTdQTB8Kdqb6VMyc97hjgUV1sH9gH+D5VG38f8LuB+5/TfkT36/hzOIVqxn8rdTrdwwfu+xPgSuAhI/5unkq1gjyEag346cTnQA0qPAO4bwfb3Yoa2DRRaFhnIJ4fs7SGdBzVxN91P3y07+VOqiDxTmqE95+2z+CJ7XG7tbivpPUNjuMPeCjVNP1HtTWqC+Q/Bz+zqR7XQUyPpVouHt6WX0ydsnka1Qd9RZeJclIsswZuP5capPUwqkl1J5YW3h9Ctfxt2mEsg7XcE6iCw7OprrzntfV7UiPSTxjT/nQ08E3gElohlmodextwclv+fwwUcMf5xx+PbzqKGl/w720fnMg1ZwDvpQoiK91aNvY3OhP+WJrA96fOfzyeGuCyI/Bz4LSOtzv4Y34vcEC7/Zl2MH4X1Qc/7X1PLVFN7EyHsLSp531UjeDDVOFifaqZdtQJ/JHUpCV/P7DuzVSNfNu2vFZH296Uqi0+jGrmfG37Ps6jClQ/p2oB36ajFpIpYjqAGky5E9VK84YWw/uppP4RamDfjlTCf/IIv6v7Ax9qtx9HNcF+lDpV66iBxxxOJcuR9YFTBeM51CCxxVTCvk+77xAqib8B2HVEn9XgAf5h7f9xVNK8ux2HvkjNPfABBsZkdBDLIW1/vi/VNXUx1Zz/MqrJ9xrgT9tj92C03UWTu4QOoc4Imkjas6jCxntGFdOQce9F5ZEN2/JCqsl8k/advpIaz3MsNTBwlY+rY3+zM+WvHRy/1w4wXwPe1dbvCvyWSTXyadje5gO3dxi4fQLw3oHlZ1CFix07eM/bUQWE57cfw8OpwstLqZrvLlQz6IVU7a7TpuIW0+Qf7ZbUqSOfAg4dWP8PVE1zlUqvw8bSDu7nAz+jmu5eRPURvrnd3pwRjA2YFNe/AX/Rbr8AuL7tu2+lpv2TWW4AABK+SURBVOXcHngS1VS8/Yhj+xHVTPgWasT8etQpeN+lumm2bfEfOoJYBhPllgO3j22f03PoqAC4EnGdRvXlTjSfH04l8sOo7qxtaYXVjmI5kGqCPnBg3SyqGf+Sgc/rblrL05g+pw1pZwlQSftq4Ji2fHw7Rq3f1bFgFWLflyrcH9+OUf/M0q6JR1AVgrOoitG9avkZ+5udCX9UCf1VVCl0b6r0ue3A/bsyjTWadpD983aA2wa4meqHe1RLHF8DFozgfa9LTfn3Dqq5Z2J07NtZ2gf+Nuqc9G1GEM/k04+eQNWC51Al17fTam/tMdPehD5FTBtQLRHPZOAUHyqhP2/E++lErXIvaoDf7lRB5kVUQe/V7fPalUqao2oW3o6BEdNU68XN3LOJ9jRaQZjW793VAZc/7pN/WYvpb4H5bd0LqZaLFzCC8R3LiPNAqtl600nrn0clzcd3vP3N2nYmxnvsTPW9r0M17y9s64+iWlRGVgOfFOcrqFPcvsrSrq2jqHP3P99iG8m+Psy+N3D72dQI/kPbvnYCNbfH1iwdSX+fe73Ncb/pmfBht+UTqOT9LWDrtu5ptPNbp3rOvdj21tRI6z2oQsNOVCHik1QN82VUDbmrZuLBHW0O1S/4LpaWbN9H9T39aUsSI+ljGkhSJ1NN+adSTXsHU6XZV1BNx0+dzu9jFeJ8RttXOu0DX87270u1DvwWeMnA+sHT4LYcYSyPpro97kdrrWm/o88MPO4k4APtdqeDxhhoMaJamb5GFZw/ThVuXtnuexnVojKq860PAl40sHw08O52ex2qBjx74L7OxzNQzdOXUa1wFwGntvWPpxL6OVQrz7S3BC4npj3bcXEdajDvRe33/xWqi+Yp7XHPpgb7PWBUsa0g7sHj6kupguvTWozXUS1QH6JaXs5hms4wGvsbH+eHTfXbfbDtIA9sH+7ED3xvapDZUzqKYTOqdvnPLJ2oYC1q4NaXqf7WjTve0V4APJ06XWyi4HBou++1VA2985GeVD/p+gPfyfntgPbG9llcwtLmxZMYUYKaIs6tqYF/32PMJX+qNv4fLC1wzhr8P6IYHkQN6DuC6lb4XNuHJmL5DlXYeUH7HkfRhH4AVTN7FTXC+kiqwHwi1cz/RKr2+4r2+Gn/jS0jroOpU+12Z+lpUE9sB/a1Bh73bFpheoTf44FUjfzMgXVzqGbfZzOicQIDsXybKnzt1I5L21EJ8eNUxeIWlo6n6GyswL14D4dRYxp2bst7t/3/xe24P4dpnIBm7G94jB/0flSt7rfURAHrUqXfRVTJ/VvAYR1t+1BqEMt6VK3z/QwUFqgRvdM6Cp0/bnk4pSWBiUE167YfyLuommZMfk5Hn8WWVFPTK1oMW7Qf7fPajr8OderRf9HO0R/jPrMuVXPZeZxxtFjWavvNMxnxKWRt+9tTrTTHD6zboiWl17L0NLLF1HiCh061H05zTAe23+2JVHfD+6gJbzanBgFOnMZzDjVodLMRfVZPpQp+j2rLn2DpBDgfoAodz20J62rGcH4zVfj5ISMq1Cwjhie297/XwLpox8MvTSS+dty6kBHNpLcS8c9qv8vL2ve9I0srjE+iCrXzp3u7a+Tc6RGxOzWo4L3UB70+1cxxXmbOp/rLnp6Z53V05ZvfUbMM3Y8aTfyfwNMj4lCAzLwuM/9nmrf5f1cai7pO8n7Ue7wiItbOzN8C/0SNRN2b+sHkNMcwlSVUyft+1CCQX2fNIbwVNbjwd1SrxL9Qn9PYZOZvM/OzmfnjccbRYrmTSlI/y6VXQxqlfakzGc6KiFkRsSc1huEr1CxsJ7f9ah61n13Z4u5kn4q6FOzngL/KzHdTBfSNqQFRv2//HxgRz6dqncdlXXmuU+0COR+izhiY2N5fARtHxF9m5guploGJ8TiHZeaPuo5rssy8kBof8632WY7DnsDfZea3J66M2PaXJVRB8MiIeAHVQvqidswaq0n5Ye32u3widdriyyf298y8hCpcXjLdMYztEpJjdifwlcy8tC3vFxE/pPqkD8vMHw1c4WjaDjrtkqGZmV+MiIlTyd4TEZ+mapxPjogvZbte8DRu9wDqQhPfpfoEv0DVgOdR/ZYTF1DYLjPf0S5S0ekFHyJiF6oGeVVEfJg6R/apwIKIeB/VQvKqVuB6LjWy87ouY+qbzPz2GDd/DfCiiHgq1Wy9LpWIPk19d0dR+9jpo4gzM2+JiKcBb46IL2fmf0fEXdTgx19FxOepJtltqXEEP+86pojYnxoFfypVKH1xRJyTmd+IiLdSl518XWa+uj1+7ezoYibDyMzPtyuifTEi5tGuhNz1dgcutLIDdRwAGLwy413Ucevx1CDTZ82EY8GkC8S8CHh4RFxBFdoOA86PiHdm5skAmfm1TuIYTWVrvCY+7IH/21HN5mdMHGAi4niqGfDCVjqetu222/OpgWyXUadL7UkdVI7OzDsi4gHA7Zn5i+nY9kAMB1LnxP4zNQhp4rSyR1GzIH0uMxdHxHOpfqhnZeat0xnDFDFtTpWuf05dyvQP1HmUz6F+yD9vhZtnU4ngwsz8XpcxaeVEXfd7AdUt9GOqS+RKqpn9GKqpf8PMvGzEcR1E7d/nU607x2bmr9t9GwN3TSyPIJa9qG6Fr0fErlQX0VrApzLzmxGxG/Xb/HFmnjl4vBiniNggM381hu3uR13U6YzMvDQiZlE56g/tKmpfAf57FC0oKyMiXkp9t2dSo+jPoc6EuIXq3vlsZr6ys+3PgH1mJFpT9bOoZq1/ogZQvYoagDCHOvD8NZXIXjqdTTWtCehJ1En921KlyZe1bV+SmX8+XduatN3NqER5eGZ+uhVe3kIVYH5EveenU301j6fOAx1Jsmw/2C9SYwIeRk2s8ivgDqpv9XxqNPNdo4hHq6ZdRvSWgeUnUWMYDsnM/x1TTE+mTinbKjNvjoj1prt1ayXjmZWZd7fWp2OpgaTnZOa32mVab8vMG8YV30wREetTp9quB3xsoqW0FeZfQZ0K99MxhngPrbV2O6pr9HTqePocamzBLOo04lup1qBrO4tjdU7iAzXvB1J94J+kksU+VA3iwdT5tbtS13G9L1Ubf8p0lUQj4vFUafuAiYQUESdQNZaDqfNpD2x9v9MuIg6hTqPZJzNvb03XX87Mhe1a19tTA39+MuofSGvmfxfVDLsl1U9/DNU3eCM1H/pty34FzRQRsRY1OOqNwJ9l5mfHHM9BVIF138y8eZyxDGqJ/DlUQXVRZi4ec0gzSkRsQ42N2Z8aFPlbasDx0RPjKsZtcotJa5V6EPDmzHxyyzdfpVqn/rb1k3cXz+qcxAEiYm+qaeNDbRDOJtQpL0+lzon8fuur3pc6vevozLximra9GTWL1t5UM/WVA/dtQp3W9svM/MF0bG85cUxuYnxeZv5mJjTftULG24FHt37NTakmx/W6LL1q+rQEvjfVNfLOzPz0mEMCICIOp2pJI+vfHUarfT8dOGsmFTBmiohYl+pufDJ1IaqLxzHYb7KIuH9m/ne7fQw1KPqb1BTCm1LTQ+9BFWZfAvzJKFpY1oQkvhE1kOuWzDy0rduYOhn/IGoAwu+oneaWzLxqmrZ7CHWO7PpUX8k1VBPRNe3+WTnCUcVTNDGu01Xtf2W1QsY7qdaCsTTB6t5piXzzzLxxJhQOJ4yrf3dFImKtrmtomj5tHM8HqEvo/i/VFTtxydhfU5WkF1KVwfWoc/1H0zU5Q35r02agCX0vKoFeR80tfRF1/eKT2uM2pqY7vLaDGO7D0mkxj6Wa6V8M/A/VF3b1dG9zyLhmZBMj/F+t6bXUxDfjOGVKkqYUEetQLQPHUJPQPKudAfEwakKhJdTg3EcC12fm9aOKbbU7T7wl8MOo6UsnZmR7NDVJxyMj4v3tcbdNVwJvfSKDMfye6of+OlV6u4k6b/WBwKHRzoEctcz8PDX68wtR5/Z2cQ78KsnMc6m5ok3gkmaEiWNka7X8InWWzwOogcm0rtefUMeuuzLzW6NM4LAaJvHWD/1SlvanrEWdwnE71Xz+8Ih4yDRu7xDgryPifhHx3Ih4I0DWeajvps5v/IcWy1uBj4xzxHVLlk/IzLtnSpPnhJnY7ClpzTTpFOGdqasVnk+1qj4kIk5tD70D2KwNFB651S6JUxMD3EyNcDyOmubuf1pTclD9rtPSV9FOW/trauDF/1AXez8qIv4MagIKqh96J6rP94eZedN0bPveMFlK0vINJPBXUud/XxIRL2pnXvw9cEJEfIPqMj0tO54ga1l6P2PbQB/4Y6nP/esRcRN13t4hmXl1O3f1zUzjYIOI2IqarOVFWdMErp+Z10bN3PPBiLg9M/+OGg1+ATWFqE3FkjSDTaqBr02NQn8csBvwsXb/P7aa+GHA68Z5Jk3vk3hL4AdSJaPj2uoPUU0c74qIT1G18jOmebTg76npW3/XBj2c3goLNwE/pUppj6Sa9Q/M6Z8LXZI0jQbPGmrzeTyY6gPfKDMvi4jnAYsiYt3MfFdEXDDOiYSg56PT26CD+1FXJHpJmwFpd2pWtJ9SA8lmA/+TmV+dzlNf2rZPpS728BBq0MPXqBnhDmv/LwHuMIFLUn9ExBOprtJPULNZ/ohqTb0hIh5HnVK2X3Y8RfUwepnEp5gx543UdVrvZukk+pdn5hs7jmMDasrQ7YBz26h0IuKD1HzfH+5y+5Kkey/qKnxrZV2c5lnASdQV8b4QEU+gzm66G/j7zPzZTJpno3cD2wb6wJ8WEe9o52R/nZoj/BOZeSBwHpXMO5WZv8rM/8jMjw8k8GdQif0bXW9fknTvtEHPC4GJZvFvUpeuPRIgM79CXZ1vI+rKfbOp7toZoa818UOoyVRe1Yb8D973GOryf3+emZ8bYUxbUxdYeTGTpliVJM08bTzVXwB/mZkXRMR9qQsxbU7N9PmxzHxde+w+1DUmZtZEWX1L4lGXp3sH1Q/+HeqSmk+nTsK/DTiFmhXt3FFO/9jm+90PuCozfzyKbUqSVk0svcrjkZn5bxGxE3WFx9dk5kURsQN12egLssNLid5bfUziAbwB2JmazvRiYBfgduqylhtk5i9m0vzNkqSZZ6BV9zhqMq4vZOZbI2J21nXMt6cqiEdm5pKxBbocMz6JD/SBP46aC/1m6qox+wI3ZOaVEfFganrToxwJLkkaVmtS/xx1Cd03DSTwQ6kLV/1wJs/xMaMHtg0k8MOpfu6HUhfweFFmXtgS+OHUdcLfaAKXJK2MzPwCdWnq4yJik5bAjwNeDfxuJidwmKGTvbTZ0NYDftmaz19KnXu9L3Xpt8Pbh/0mYHvglMy80CZ0SdLKavnj5cBXI+IfgOcAL8h26eiZbMYl8Yh4ENUHcStwLXA5sICa1OXlwOHUALJXR8Qdmfm2ieeawCVJqyIzP99OHzsHeMQ0z/DZmRnVnB4RuwEfBV4JvIBK4LtRg9a2BT6cmddRo9DPoS44IknSvZaZnwE26UsCh5lXE98M2D0zLwaIiIuAp1Fx3gm8uF3e9RTqYibfHFegkqTVz7jnQl9ZMyqJZ+bXIuLgiLgmM3ekJp+fTc0/fl6bnW0u1Vfx1bEGK0nSmM3IU8wi4mDg48APgSf0rWQkSdIozKg+8QltutTDgK0mEnhErDXeqCRJmllmZBIHyMwvUZPN3xwRm2bmneOOSZKkmWRGNqcPak3rv8nMS8YdiyRJM8mMT+ITnMhFkqR76k0SlyRJ9zRj+8QlSdLymcQlSeopk7gkST1lEpdWQxHxh4i4fOBv+1V4jSPa9QwkzVAzatpVSdPmt5m5x718jSOAzwDfH/YJETEnM++6l9uVNCRr4tIaIiL2jIgvR8SlEXF+RGzd1r84Ir4dEd+NiH+NiPUi4jHUrIl/22ryO0XEJRExrz1ni4i4tt0+LiLOi4gvARdFxPoR8YGI+FZEfCciDh/Xe5ZWdyZxafW07kBT+qfatMXvBo7OzD2BDwBvaI89JzP3yszdgR8Ax2fm14HzgNMzc4/M/MkKtvfI9tpPBP4f8KXM3BvYlyoIrN/Be5TWeDanS6unezSnR8RDgYcCF7bL+c4Gbmh3PzQiXg9sAmwAnL8K27swM29pt58CHBYRr2jL6wD3pwoIkqaRSVxaMwTwvczcZ4r7PggckZnfjYjjgCct4zXuYmnr3TqT7vv1pG0dlZlXrXK0koZic7q0ZrgKmBsR+0BdFTAiHtLu2xC4oTW5P3fgOb9s9024Ftiz3T56Ods6HzgxWpU/Ih5x78OXNBWTuLQGyMw7qMT7NxHxXeBy4DHt7r8Avgn8O/DDgaedDZzeBqftBLwF+JOI+A6wxXI291fAWsB/RsT32rKkDjh3uiRJPWVNXJKknjKJS5LUUyZxSZJ6yiQuSVJPmcQlSeopk7gkST1lEpckqadM4pIk9dT/B3tjdDCKTh3VAAAAAElFTkSuQmCC\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ],
      "source": [
        "''' Deal with missing values. First, calculate the percentage of\n",
        "missing values for every column, and plot them as a bar chart'''\n",
        "\n",
        "# data processing that is done on the training dataset is similarly applied to the test dataset, as test dataset is something that will keep coming.\n",
        "\n",
        "# all null vals which is summed and performed in percent format\n",
        "null_vals = mydf_train_valid.isnull().sum()/len(mydf_train_valid)*100\n",
        "\n",
        "null_vals = pd.DataFrame(null_vals)\n",
        "null_vals.reset_index(inplace = True)\n",
        "null_vals.columns = [\"Feature\",\"Percent missing\"]\n",
        "plt.figure(figsize = (8,6))\n",
        "plt.xticks(rotation=45)\n",
        "\n",
        "# ploting the figure with seaborn\n",
        "sns.barplot(x = \"Feature\",y =\"Percent missing\",data = null_vals)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "xEm1YFOaCFb4"
      },
      "source": [
        "From the above plot, it looks like Cabin has ~80%missing values.\n",
        "It would be meaningless to impute or fill in 80% values, so we drop the column.\n",
        "We will impute age (which has ~ 20% missing, but we'll try to impute),\n",
        "Fare, and Embarked column.These have very little missing values\n",
        "\n",
        "We are going to preprocess this dataset in these steps--\n",
        "\n",
        "1. Convert the entire dataframe to an array of numbers. This itself is going to happen in two steps -- (a) Convert object types and string types to category type (b) map and convert cateogries of numbers.\n",
        "\n",
        "2. Impute or \"fill in\" missing values or NaNs. Here, continuous (e.g. Fare column) and categorical values are treated separately. For filling up missing continuous values, we use the median value of that column, and filling up missing categorical values, we use 0. Also, we add a separate \"marker\" column for both that notes whether a value has been imputed or not.\n",
        "\n",
        "3. Finally, for a lot of the algorithms like k-NN, we scale the data to lie between 0 and 1 with mean zero and unit variance.\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# categorical cols: survived, pclass, sex, cabin -> filled with various categories\n",
        "# continous cols: age, fare  "
      ],
      "metadata": {
        "id": "X8H48GwoYbJ7"
      },
      "execution_count": 51,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 52,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 130
        },
        "id": "MlU8qPtkCFb5",
        "outputId": "026fd3c3-a3e2-4e8c-e290-64a2217d429b"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(1100, 11)\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   PassengerId  Survived  Pclass  \\\n",
              "0            1         0       3   \n",
              "1            2         1       1   \n",
              "\n",
              "                                                Name     Sex   Age  SibSp  \\\n",
              "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
              "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
              "\n",
              "   Parch     Ticket     Fare Embarked  \n",
              "0      0  A/5 21171   7.2500        S  \n",
              "1      0   PC 17599  71.2833        C  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-21b5f872-7baf-4e3c-885e-dfb8c8f8749b\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Name</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Ticket</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Braund, Mr. Owen Harris</td>\n",
              "      <td>male</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>A/5 21171</td>\n",
              "      <td>7.2500</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
              "      <td>female</td>\n",
              "      <td>38.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>PC 17599</td>\n",
              "      <td>71.2833</td>\n",
              "      <td>C</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-21b5f872-7baf-4e3c-885e-dfb8c8f8749b')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-21b5f872-7baf-4e3c-885e-dfb8c8f8749b button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-21b5f872-7baf-4e3c-885e-dfb8c8f8749b');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 52
        }
      ],
      "source": [
        "mydf_train_valid_2 = mydf_train_valid.drop(\"Cabin\",axis = 1)\n",
        "print(mydf_train_valid_2.shape)\n",
        "mydf_train_valid_2.head(2)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 53,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4lDqYcaICFb6",
        "outputId": "e1fb4a73-f0b3-4660-b47e-aee25d5c04f1"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 1100 entries, 0 to 1099\n",
            "Data columns (total 11 columns):\n",
            " #   Column       Non-Null Count  Dtype  \n",
            "---  ------       --------------  -----  \n",
            " 0   PassengerId  1100 non-null   int64  \n",
            " 1   Survived     1100 non-null   int64  \n",
            " 2   Pclass       1100 non-null   int64  \n",
            " 3   Name         1100 non-null   object \n",
            " 4   Sex          1100 non-null   object \n",
            " 5   Age          881 non-null    float64\n",
            " 6   SibSp        1100 non-null   int64  \n",
            " 7   Parch        1100 non-null   int64  \n",
            " 8   Ticket       1100 non-null   object \n",
            " 9   Fare         1099 non-null   float64\n",
            " 10  Embarked     1098 non-null   object \n",
            "dtypes: float64(2), int64(5), object(4)\n",
            "memory usage: 94.7+ KB\n"
          ]
        }
      ],
      "source": [
        "#Check types of each column with the dataframe info () method\n",
        "mydf_train_valid_2.info()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 54,
      "metadata": {
        "id": "uitveBSpCFb6"
      },
      "outputs": [],
      "source": [
        "'''You can see that several of the columns or features are \"object\" type\n",
        "These need to be changed to category before we can convert those to \n",
        "mappings and numbers'''\n",
        "#1 (a) Define a function to convert object types and string types to category type\n",
        "\n",
        "def str_to_cat(my_df):\n",
        "    for p,q in my_df.items(): #my_df.items() is a generator in Python\n",
        "        if is_string_dtype(q): \n",
        "            my_df[p] = q.astype('category').cat.as_ordered()\n",
        "    return my_df"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 55,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kdx5Rf17CFb6",
        "outputId": "c18784ec-88e5-4b55-9c3c-2a4657e51759"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n"
          ]
        }
      ],
      "source": [
        "mydf_train_valid_3 = str_to_cat(mydf_train_valid_2)\n",
        "print()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 56,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "W-INC6P9CFb7",
        "outputId": "73e16705-89fc-456d-b175-637a2f3da22c"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 1100 entries, 0 to 1099\n",
            "Data columns (total 11 columns):\n",
            " #   Column       Non-Null Count  Dtype   \n",
            "---  ------       --------------  -----   \n",
            " 0   PassengerId  1100 non-null   int64   \n",
            " 1   Survived     1100 non-null   int64   \n",
            " 2   Pclass       1100 non-null   int64   \n",
            " 3   Name         1100 non-null   category\n",
            " 4   Sex          1100 non-null   category\n",
            " 5   Age          881 non-null    float64 \n",
            " 6   SibSp        1100 non-null   int64   \n",
            " 7   Parch        1100 non-null   int64   \n",
            " 8   Ticket       1100 non-null   category\n",
            " 9   Fare         1099 non-null   float64 \n",
            " 10  Embarked     1098 non-null   category\n",
            "dtypes: category(4), float64(2), int64(5)\n",
            "memory usage: 146.5 KB\n"
          ]
        }
      ],
      "source": [
        "#Check dtypes of columns after this operation\n",
        "mydf_train_valid_3.info()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 57,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Q9xRnFXaCFb7",
        "outputId": "bff48dcf-bcc8-4042-8ee7-45c12766e759"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Index(['C', 'Q', 'S'], dtype='object')\n",
            "Index(['female', 'male'], dtype='object')\n"
          ]
        }
      ],
      "source": [
        "'''Check the cateogry mapping for Embarked and Sex column. We need this\n",
        "later'''\n",
        "print(mydf_train_valid_3.Embarked.cat.categories)\n",
        "print(mydf_train_valid_3.Sex.cat.categories)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Q5jfJVPZCFb8"
      },
      "source": [
        "All object categories like Name, Sex, and Ticket have been converted to\n",
        "Category dtype !"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 58,
      "metadata": {
        "id": "IEXWSgtcCFcC"
      },
      "outputs": [],
      "source": [
        "'''Define impute functions. Impute categorical NaNs with -1, \n",
        "where we add 1 to make it 0. For each \n",
        "continuous variables, we impute missing values with median values of that\n",
        "column, and for every variable\n",
        "where any rows were imputed, add a separate 'imputed or not' column'''\n",
        "\n",
        "def mydf_to_nums(my_df, feature, null_status):\n",
        "    if not is_numeric_dtype(feature):\n",
        "        my_df[null_status] = feature.cat.codes + 1\n",
        "        \n",
        "def mydf_imputer(my_df, feature, null_status, null_table):\n",
        "    if is_numeric_dtype(feature):\n",
        "        if pd.isnull(feature).sum() or (null_status in null_table):\n",
        "            my_df[null_status+'_na'] = pd.isnull(feature)\n",
        "            filler = null_table[null_status] if null_status in null_table else feature.median()\n",
        "            my_df[null_status] = feature.fillna(filler)\n",
        "            null_table[null_status] = filler\n",
        "    return null_table   \n",
        "\n",
        "def mydf_imputer_mean(my_df, feature, null_status, null_table):    \n",
        "    if is_numeric_dtype(feature):\n",
        "        if pd.isnull(feature).sum() or (null_status in null_table):\n",
        "            my_df[null_status+'_na'] = pd.isnull(feature)\n",
        "            filler = null_table[null_status] if null_status in null_table else feature.mean()\n",
        "            my_df[null_status] = feature.fillna(filler)\n",
        "            null_table[null_status] = filler\n",
        "    return null_table \n",
        "\n",
        "def mydf_preprocessor(my_df, null_table):\n",
        "    '''null_table  = your table or None'''\n",
        "    \n",
        "    if null_table is None: \n",
        "        null_table = dict()\n",
        "    for p,q in my_df.items(): \n",
        "        null_table = mydf_imputer(my_df, q, p, null_table)\n",
        "    for p,q in my_df.items(): \n",
        "        mydf_to_nums(my_df, q, p)\n",
        "    my_df = pd.get_dummies(my_df, dummy_na = True)\n",
        "    res = [my_df, null_table]\n",
        "    return res"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install impyute\n",
        "from impyute.imputation.cs import fast_knn"
      ],
      "metadata": {
        "id": "wTWDLWPuEeG9",
        "outputId": "df2f376a-7105-4541-affc-3f3d6fdb7e27",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "execution_count": 66,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/\n",
            "Collecting impyute\n",
            "  Downloading impyute-0.0.8-py2.py3-none-any.whl (31 kB)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from impyute) (1.21.6)\n",
            "Requirement already satisfied: scipy in /usr/local/lib/python3.7/dist-packages (from impyute) (1.7.3)\n",
            "Requirement already satisfied: scikit-learn in /usr/local/lib/python3.7/dist-packages (from impyute) (1.0.2)\n",
            "Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from scikit-learn->impyute) (3.1.0)\n",
            "Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.7/dist-packages (from scikit-learn->impyute) (1.1.0)\n",
            "Installing collected packages: impyute\n",
            "Successfully installed impyute-0.0.8\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import sys\n",
        "sys.setrecursionlimit(100000)\n",
        "imputed_training=fast_knn(mydf_train_valid_3.values, k=30)"
      ],
      "metadata": {
        "id": "6JFJMjJeEkCZ",
        "outputId": "dd21805b-b29d-4f82-d0d4-54a7c91eb1fe",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 353
        }
      },
      "execution_count": 69,
      "outputs": [
        {
          "output_type": "error",
          "ename": "BadInputError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mBadInputError\u001b[0m                             Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-69-b835c39a2493>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0msys\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0msys\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msetrecursionlimit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m100000\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 3\u001b[0;31m \u001b[0mimputed_training\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mfast_knn\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmydf_train_valid_3\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mvalues\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mk\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m30\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/impyute/util/preprocess.py\u001b[0m in \u001b[0;36mwrapper\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m     53\u001b[0m             \u001b[0;32mreturn\u001b[0m \u001b[0mpd_DataFrame\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfn\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     54\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 55\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mfn\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     56\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     57\u001b[0m     \u001b[0;32mreturn\u001b[0m \u001b[0mwrapper\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/impyute/util/checks.py\u001b[0m in \u001b[0;36mwrapper\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m     31\u001b[0m             \u001b[0;32mraise\u001b[0m \u001b[0mBadInputError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Not a np.ndarray.\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     32\u001b[0m         \u001b[0;32melif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0m_dtype_float\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 33\u001b[0;31m             \u001b[0;32mraise\u001b[0m \u001b[0mBadInputError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Data is not float.\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     34\u001b[0m         \u001b[0;32melif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0m_nan_exists\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     35\u001b[0m             \u001b[0;32mraise\u001b[0m \u001b[0mBadInputError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"No NaN's in given data\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mBadInputError\u001b[0m: Data is not float."
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# from sklearn.impute import SimpleImputer\n",
        "# imp_mean = SimpleImputer(strategy='most_frequent')\n",
        "# imp_mean.fit(mydf_train_valid_3)\n",
        "# imputed_train_df = imp_mean.transform(mydf_train_valid_3)\n",
        "# mydf_train_valid_test = pd.DataFrame(imputed_train_df, columns = mydf_train_valid_3.columns).astype(mydf_train_valid_3.dtypes.to_dict())\n",
        "# # my_table = None"
      ],
      "metadata": {
        "id": "u15ef0Cb-K1N"
      },
      "execution_count": 59,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 60,
      "metadata": {
        "id": "VZD2nZFpCFcD"
      },
      "outputs": [],
      "source": [
        "mydf_train_valid_4,my_table = mydf_preprocessor(mydf_train_valid_3,null_table = None)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# mydf_train_valid_test"
      ],
      "metadata": {
        "id": "twOcmtMuCX5L",
        "outputId": "4fdaca06-62fd-4940-e901-6c0358410d22",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 424
        }
      },
      "execution_count": 61,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "      PassengerId  Survived  Pclass  \\\n",
              "0               1         0       3   \n",
              "1               2         1       1   \n",
              "2               3         1       3   \n",
              "3               4         1       1   \n",
              "4               5         0       3   \n",
              "...           ...       ...     ...   \n",
              "1095         1096         1       2   \n",
              "1096         1097         0       1   \n",
              "1097         1098         0       3   \n",
              "1098         1099         1       2   \n",
              "1099         1100         1       1   \n",
              "\n",
              "                                                   Name     Sex   Age  SibSp  \\\n",
              "0                               Braund, Mr. Owen Harris    male  22.0      1   \n",
              "1     Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
              "2                                Heikkinen, Miss. Laina  female  26.0      0   \n",
              "3          Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
              "4                              Allen, Mr. William Henry    male  35.0      0   \n",
              "...                                                 ...     ...   ...    ...   \n",
              "1095                           Andrew, Mr. Frank Thomas    male  25.0      0   \n",
              "1096                          Omont, Mr. Alfred Fernand    male  24.0      0   \n",
              "1097                           McGowan, Miss. Katherine  female  35.0      0   \n",
              "1098                       Collett, Mr. Sidney C Stuart    male  24.0      0   \n",
              "1099                      Rosenbaum, Miss. Edith Louise  female  33.0      0   \n",
              "\n",
              "      Parch            Ticket     Fare Embarked  \n",
              "0         0         A/5 21171   7.2500        S  \n",
              "1         0          PC 17599  71.2833        C  \n",
              "2         0  STON/O2. 3101282   7.9250        S  \n",
              "3         0            113803  53.1000        S  \n",
              "4         0            373450   8.0500        S  \n",
              "...     ...               ...      ...      ...  \n",
              "1095      0        C.A. 34050  10.5000        S  \n",
              "1096      0        F.C. 12998  25.7417        C  \n",
              "1097      0              9232   7.7500        Q  \n",
              "1098      0             28034  10.5000        S  \n",
              "1099      0          PC 17613  27.7208        C  \n",
              "\n",
              "[1100 rows x 11 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-60560957-7b6f-4393-ac78-2c8201860fe7\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Name</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Ticket</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Braund, Mr. Owen Harris</td>\n",
              "      <td>male</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>A/5 21171</td>\n",
              "      <td>7.2500</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
              "      <td>female</td>\n",
              "      <td>38.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>PC 17599</td>\n",
              "      <td>71.2833</td>\n",
              "      <td>C</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>Heikkinen, Miss. Laina</td>\n",
              "      <td>female</td>\n",
              "      <td>26.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>STON/O2. 3101282</td>\n",
              "      <td>7.9250</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
              "      <td>female</td>\n",
              "      <td>35.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>113803</td>\n",
              "      <td>53.1000</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Allen, Mr. William Henry</td>\n",
              "      <td>male</td>\n",
              "      <td>35.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>373450</td>\n",
              "      <td>8.0500</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1095</th>\n",
              "      <td>1096</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>Andrew, Mr. Frank Thomas</td>\n",
              "      <td>male</td>\n",
              "      <td>25.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>C.A. 34050</td>\n",
              "      <td>10.5000</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1096</th>\n",
              "      <td>1097</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>Omont, Mr. Alfred Fernand</td>\n",
              "      <td>male</td>\n",
              "      <td>24.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>F.C. 12998</td>\n",
              "      <td>25.7417</td>\n",
              "      <td>C</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1097</th>\n",
              "      <td>1098</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>McGowan, Miss. Katherine</td>\n",
              "      <td>female</td>\n",
              "      <td>35.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>9232</td>\n",
              "      <td>7.7500</td>\n",
              "      <td>Q</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1098</th>\n",
              "      <td>1099</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>Collett, Mr. Sidney C Stuart</td>\n",
              "      <td>male</td>\n",
              "      <td>24.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>28034</td>\n",
              "      <td>10.5000</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1099</th>\n",
              "      <td>1100</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Rosenbaum, Miss. Edith Louise</td>\n",
              "      <td>female</td>\n",
              "      <td>33.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>PC 17613</td>\n",
              "      <td>27.7208</td>\n",
              "      <td>C</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>1100 rows × 11 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-60560957-7b6f-4393-ac78-2c8201860fe7')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-60560957-7b6f-4393-ac78-2c8201860fe7 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-60560957-7b6f-4393-ac78-2c8201860fe7');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 61
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "mydf_train_valid_3.info()"
      ],
      "metadata": {
        "id": "C8nSkotE6TeM",
        "outputId": "4b700117-5f49-4c04-b86e-b32169ea08e9",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "execution_count": 63,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 1100 entries, 0 to 1099\n",
            "Data columns (total 13 columns):\n",
            " #   Column       Non-Null Count  Dtype  \n",
            "---  ------       --------------  -----  \n",
            " 0   PassengerId  1100 non-null   int64  \n",
            " 1   Survived     1100 non-null   int64  \n",
            " 2   Pclass       1100 non-null   int64  \n",
            " 3   Name         1100 non-null   int16  \n",
            " 4   Sex          1100 non-null   int8   \n",
            " 5   Age          1100 non-null   float64\n",
            " 6   SibSp        1100 non-null   int64  \n",
            " 7   Parch        1100 non-null   int64  \n",
            " 8   Ticket       1100 non-null   int16  \n",
            " 9   Fare         1100 non-null   float64\n",
            " 10  Embarked     1100 non-null   int8   \n",
            " 11  Age_na       1100 non-null   bool   \n",
            " 12  Fare_na      1100 non-null   bool   \n",
            "dtypes: bool(2), float64(2), int16(2), int64(5), int8(2)\n",
            "memory usage: 68.9 KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "HL8RBh50CFcD"
      },
      "outputs": [],
      "source": [
        "mydf_train_valid_4.head(3) # All numerical values and labels for True/False"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ia33uojnCFcD"
      },
      "outputs": [],
      "source": [
        "'''Please store the null_table, category mapping separately.\n",
        "We will need to process the test dataset using these values'''\n",
        "my_table"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "XnuQ-GdSCFcE"
      },
      "outputs": [],
      "source": [
        "'''Now, let's separate the X and Y variables (vertical split of the \n",
        "dataframe). Here the Y column is the variable we are trying to predict, \n",
        "survived or not(0 = No, 1 = Yes)'''\n",
        "\n",
        "# as said, y is something we wanted to predict, survived in this case and the rest of the data in x variable\n",
        "\n",
        "Y = mydf_train_valid_4[\"Survived\"]\n",
        "X = mydf_train_valid_4.drop([\"Survived\"],axis = 1)\n",
        "\n",
        "print(X.shape,Y.shape)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "jczJCEBsCFcH"
      },
      "outputs": [],
      "source": [
        "# Scaling\n",
        "# KnN cannot determine values with different range of continuous value, so we try\n",
        "# to scale it between 0 and 1 or normalizing\n",
        "\n",
        "'''Note that the different continuous variable columns of this dataframe \n",
        "have numbers in different ranges. For example, the Fare and age columns. \n",
        "For some machine learning algorithms like Decision Trees and \n",
        "their ensembles (Random Forests, for example) the above X and Y\n",
        "can be directly used as input. However, for a lot of other ML algorithms \n",
        "like K nearest neighbors (KNN), we need to scale the continuous variables \n",
        "so that their values are mapped to a number between 0 and 1. \n",
        "Let's split this dataframe into continuous variable \n",
        "columns and those with categorical variables. We will leave \n",
        "the categorical variables untouched because their \n",
        "values are treated as different levels and its kind of meaningless to\n",
        "scale them'''\n",
        "\n",
        "X_cat = X[['PassengerId', 'Pclass', 'Name', 'Sex', 'SibSp', 'Parch',\n",
        "       'Ticket', 'Embarked', 'Age_na', 'Fare_na']]\n",
        "# X_cat = X[['PassengerId', 'Pclass', 'Name', 'Sex', 'SibSp', 'Parch',\n",
        "#        'Ticket', 'Embarked']]\n",
        "X_con = X.drop(X_cat,axis = 1)\n",
        "print(X_cat.shape,X_con.shape) # Categorical and continuous"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "M5i0yPatCFcH"
      },
      "outputs": [],
      "source": [
        "# Applicable for continous only! \n",
        "# Bell curve applicable which gives the standard deviation\n",
        "# this was done using sklearn\n",
        "'''Scale the continuous variables. To standardize (includes scaling), \n",
        "we subtract mean of that column from every value, then divide the results \n",
        "by the variable's standard deviation. There are different ways to \n",
        "standardize. Please see preprocessing under scikit-leanr page'''\n",
        "\n",
        "scaler = preprocessing.StandardScaler().fit(X_con)\n",
        "X_con_sc = pd.DataFrame(scaler.transform(X_con))\n",
        "X_con_sc.columns = [\"Age\",\"Fare\"]\n",
        "print(X_con_sc.shape)\n",
        "X_con_sc.head(2)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "A6PmTxtQCFcI"
      },
      "outputs": [],
      "source": [
        "'''Store this scaler variable or its mean and SD, by pickling or something;\n",
        "we need to use the same mean and SD scaler later while pre-processing \n",
        "the test set. Now, let's join the cateogrical and scaled continuous \n",
        "variables, back together into one dataframe'''\n",
        "\n",
        "df_list = [X_cat,X_con_sc]\n",
        "X_full = pd.concat(df_list,axis = 1)\n",
        "print(X_full.shape)\n",
        "X_full.head(2)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "gK5tCjoSCFcI"
      },
      "outputs": [],
      "source": [
        "# Initially we just split 15% out for test, while validation 15% was pending,\n",
        "# it will be done as per below\n",
        "\n",
        "'''Then, split into train and valid sets for model building \n",
        "and hyperparameter tuning, respectively !Remember, we need to \n",
        "split (horizontally the rows) X_full into train and validation sets.\n",
        "We use the dataframe splitter function we defined previously.\n",
        "Strictly for later use in another module, merge X and Y and store.\n",
        "Save it as train data. Reason explained in module III'''\n",
        "\n",
        "X_train,X_valid = mydf_splitter(X_full,900)\n",
        "Y_train,Y_valid = mydf_splitter(Y,900)\n",
        "\n",
        "print(X_train.shape,X_valid.shape,Y_train.shape,Y_valid.shape)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "gz67zC0yCFcI"
      },
      "outputs": [],
      "source": [
        "'''Time for training the model and evaluating it on the validation set. \n",
        "At first, let's use the default values for the kNN hyperparameters -\n",
        "n_neighbors = 3,weights = 'uniform'). KNN has more hyperparameters such as\n",
        "leaf_size, metric, etc. But, these two are key hyperparamters'''\n",
        "\n",
        "my_knn_model = KNeighborsClassifier(n_neighbors = 5,weights = 'uniform') # args are hyperparamters\n",
        "my_knn_model.fit(X_train,Y_train)\n",
        "\n",
        "#Predict on the validation set\n",
        "Y_pred = my_knn_model.predict(X_valid)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "qWAOWAORCFcJ"
      },
      "outputs": [],
      "source": [
        "# Plot confusion matrix\n",
        "from sklearn.metrics import confusion_matrix\n",
        "\n",
        "my_knn_cmatrix = confusion_matrix(Y_valid,Y_pred)\n",
        "\n",
        "my_knn_df = pd.DataFrame(my_knn_cmatrix)\n",
        "plt.figure(figsize = (8,8))\n",
        "sns.heatmap(my_knn_df, xticklabels = [\"Unlucky\",\"Survived\"],\n",
        "            yticklabels = [\"Unlucky\",\"Survived\"],annot = True)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "5lKxMbtXCFcJ"
      },
      "outputs": [],
      "source": [
        "print(accuracy_score(Y_valid,Y_pred),\n",
        "      matthews_corrcoef(Y_valid,Y_pred),f1_score(Y_valid,Y_pred))\n",
        "\n",
        "#An MCC of -0.0474 looks bad !We need to do model tuning or \n",
        "#hyperparameter tuning to try to make it better\n",
        "\n",
        "# accuracy goes from 0-1 \n",
        "# less than 0.5 is useless\n",
        "\n",
        "# above methods are being used to evaluate the model"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ceJmuFm2CFcJ"
      },
      "source": [
        "# V. Hyperparameter tuning"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "xpiMKpVlCFcJ"
      },
      "outputs": [],
      "source": [
        "'''We chose a value of K = 5 here. But how do we know if that's the right \n",
        "value? We need to do hyper parameter tuning.That is, we need to check \n",
        "different values of K and find out performance scores for each on the \n",
        "validation set! We will pick the value of K that gives the best \n",
        "validation set accuracy and use that value of K to predict on the\n",
        "test set, which we have kept aside'''\n",
        "\n",
        "from sklearn.metrics import accuracy_score\n",
        "\n",
        "num_neighs = list()\n",
        "accuracy_list = list()\n",
        "\n",
        "\n",
        "# iteratively for each value of k, we are checking how score is improving\n",
        "for neighbor in range(1,20): \n",
        "    my_knn_model = KNeighborsClassifier(n_neighbors = neighbor,weights = 'uniform')\n",
        "    my_knn_model.fit(X_train,Y_train)\n",
        "    Y_pred = my_knn_model.predict(X_valid)\n",
        "    accuracy = accuracy_score(Y_valid,Y_pred)\n",
        "    num_neighs.append(neighbor)\n",
        "    accuracy_list.append(accuracy)\n",
        "    # Use of another for loop if there will be another list of or range of args \n",
        "    # given for hyperparameter to use multiple combinations\n",
        "    # Part of Assignment 1\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Fm5mgKiNCFcJ"
      },
      "outputs": [],
      "source": [
        "eval_df =  pd.DataFrame({\"Num of neighbors\": num_neighs,\"Valid accuracy Score\": accuracy_list})\n",
        "eval_df"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "zRNvfxEDCFcK"
      },
      "outputs": [],
      "source": [
        "#Plot accuracy Vs validation set accuracy of the model\n",
        "sns.set_style(\"whitegrid\")\n",
        "sns.pairplot(eval_df,x_vars = \"Num of neighbors\",\n",
        "             y_vars = \"Valid accuracy Score\",plot_kws = {'s': 60},height = 4.0)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "f6nxFRqsCFcK"
      },
      "outputs": [],
      "source": [
        "'''Congrats, K = 14 seems to give the best validation set accuracy (= 0.6) !!! So, let's turn to the test set\n",
        "and use K = 14 for that !OK, so how do we save this trained and \n",
        "hyperparameter tuned model for later use? First, we club together, the\n",
        "train and valid set. We already have this dataframe. Then, we make and \n",
        "train a model with K = 14. Then save it with joblib, which we imported \n",
        "earlier'''\n",
        "\n",
        "\n",
        "knn_model_fin = KNeighborsClassifier(n_neighbors = 14,weights = 'uniform')\n",
        "knn_model_fin.fit(X_full,Y)\n",
        "\n",
        "# !mkdir knn_model\n",
        "knn_model_name = 'knn_model_final.sav'\n",
        "joblib.dump(knn_model_fin,knn_model_name)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "dyyMmdXZCFcK"
      },
      "outputs": [],
      "source": [
        "#Make sure your model has been saved !\n",
        "# !ls knn_model"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "m736-kOACFcK"
      },
      "outputs": [],
      "source": [
        "# Congrats! You have saved your model!Now, let's read it back in!\n",
        "knn_model_loaded = joblib.load(knn_model_name)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "TVrLxzFCCFcL"
      },
      "source": [
        "# VI. Evaluating test set accuracy with the trained model"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0Oqk4s0ECFcL"
      },
      "outputs": [],
      "source": [
        "'''Before we can apply this on the test set, we\n",
        "need to pre-process the test set in exactly the same way we did the\n",
        "train_valid set !!!'''\n",
        "\n",
        "print(mydf_test.shape)\n",
        "mydf_test.head(3)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "DCoZXJG9CFcL"
      },
      "outputs": [],
      "source": [
        "#get rid of the \"cabin\" column as we did before with the train_valid set\n",
        "mydf_test1 = mydf_test.drop(\"Cabin\",axis = 1)\n",
        "print(mydf_test1.shape)\n",
        "mydf_test1.head(3)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "lEouExZJCFcL"
      },
      "outputs": [],
      "source": [
        "'''Make sure the category codes for train and test sets are the same as \n",
        "the ones we used previously! Here, we have coded, Name, Sex, Ticket and \n",
        "embarked. Because Name and Ticket ids will not be repeated, \n",
        "we will check Sex and embarked. Checking category codes for the test set...'''\n",
        "\n",
        "mydf_test2 = str_to_cat(mydf_test1)\n",
        "mydf_test2.Sex.cat.categories"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "LoYMRMJtCFcL"
      },
      "outputs": [],
      "source": [
        "#Check for the Embarked column\n",
        "mydf_test2.Embarked.cat.categories"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "nzT4iFpuCFcL"
      },
      "outputs": [],
      "source": [
        "#Cool, this means the category codes are the same. We can proceed.\n",
        "#Make sure you use the same impute values of median.\n",
        "mydf_test3,my_table1 = mydf_preprocessor(mydf_test2,\n",
        "                                         null_table = my_table)\n",
        "print(mydf_test3.shape)\n",
        "mydf_test3.head(3)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "jlNBOszOCFcM"
      },
      "outputs": [],
      "source": [
        "my_table1"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "bHospzOgCFcM"
      },
      "outputs": [],
      "source": [
        "# Now, let's split out the X and Y variables (vertical split of the dataframe)\n",
        "#Remember we did this previously!\n",
        "\n",
        "\n",
        "Y_t = mydf_test3[\"Survived\"]\n",
        "X_t = mydf_test3.drop([\"Survived\"],axis = 1)\n",
        "\n",
        "print(X_t.shape,Y_t.shape)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "wGlVb54YCFcM"
      },
      "outputs": [],
      "source": [
        "#Separate continuous and categorical variables/columns for scaling\n",
        "\n",
        "X_cat_t = X_t[['PassengerId', 'Pclass', 'Name', 'Sex', 'SibSp', 'Parch',\n",
        "       'Ticket', 'Embarked', 'Age_na', 'Fare_na']]\n",
        "X_con_t = X_t.drop(X_cat_t,axis = 1)\n",
        "print(X_cat_t.shape,X_con_t.shape)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "e1NGF2zYCFcM"
      },
      "outputs": [],
      "source": [
        "'''Scale using the training set mean and SD. This is already captured in\n",
        "the scaler object we made. Else, save that in a joblib dump too to reload'''\n",
        "\n",
        "X_con_sct = pd.DataFrame(scaler.transform(X_con_t))\n",
        "X_con_sct.columns = [\"Age\",\"Fare\"]\n",
        "print(X_con_sct.shape)\n",
        "X_con_sct.head(2)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "lU2CCasSCFcM"
      },
      "outputs": [],
      "source": [
        "print(X_cat_t.shape,X_con_sct.shape)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "LJng7vfjCFcM"
      },
      "outputs": [],
      "source": [
        "#Re-index before merging\n",
        "X_cat_t.reset_index(inplace = True,drop = False)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0s2g5trnCFcN"
      },
      "outputs": [],
      "source": [
        "X_cat_t.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "wiFF6RgZCFcN"
      },
      "outputs": [],
      "source": [
        "X_cat_t.drop(\"index\",inplace = True,axis = 1)\n",
        "X_cat_t.head(2)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "OnWjdFAiCFcN"
      },
      "outputs": [],
      "source": [
        "#Merge the two sets of columns\n",
        "df_list_I = [X_cat_t,X_con_sct]\n",
        "X_test_I = pd.concat(df_list_I,axis = 1)\n",
        "print(X_test_I.shape)\n",
        "X_test_I.head(2)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "_GN8dnDmCFcN"
      },
      "outputs": [],
      "source": [
        "#Now we are ready to test it out. Let's load the saved model first.\n",
        "kNN_loaded = joblib.load(knn_model_name)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "eLpGl0MKCFcN"
      },
      "outputs": [],
      "source": [
        "#Testing...\n",
        "Y_test_pred = kNN_loaded.predict(X_test_I)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "iDdSTsfiCFcN"
      },
      "outputs": [],
      "source": [
        "print(accuracy_score(Y_t,Y_test_pred),\n",
        "      matthews_corrcoef(Y_t,Y_test_pred),f1_score(Y_t,Y_test_pred))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "EKOMA719CFcO"
      },
      "outputs": [],
      "source": [
        "'''We are done! Our kNN model is not doing great on this dataset but\n",
        "we learnt how to properly use machine learning. Soon, we will learn\n",
        "how to use other algorithms to get better performance'''"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "z3bxP2XuCFcO"
      },
      "outputs": [],
      "source": [
        "'''Write out full train_valid and test dataframes for later use\n",
        "in module III'''\n",
        "X_full[\"Survived\"] = Y\n",
        "X_test_I[\"Survived\"] = Y_t\n",
        "\n",
        "print(X_full.shape)\n",
        "print(X_test_I.shape)"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.6.4"
    },
    "colab": {
      "name": "Module_III_code.ipynb",
      "provenance": []
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}