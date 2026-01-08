{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPoyTQvivcqmpFVFynTdltw",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/rishiramanuja10/ML/blob/main/7.%20Logistic%20Regression.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.datasets import load_iris\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.metrics import accuracy_score\n",
        "import numpy as np\n",
        "# Load the iris dataset\n",
        "X, y = load_iris(return_X_y=True)\n",
        "# Add some random noise to labels\n",
        "np.random.seed(42)\n",
        "y_noisy = (y + np.random.choice([1, -1], size=len(y), p=[0.1, 0.9])) % 3\n",
        "# Split the dataset into training and testing sets\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y_noisy, test_size=0.2, random_state=42)\n",
        "# Initialize and train the Logistic Regression model with increased max_iter\n",
        "logistic_regression_model = LogisticRegression(max_iter=1000).fit(X_train, y_train)\n",
        "# Make predictions on the test set\n",
        "y_pred = logistic_regression_model.predict(X_test)\n",
        "# Display accuracy\n",
        "print(f\"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "dbtb4K9-_wIV",
        "outputId": "ff7346d8-d046-4788-8d63-8aeaede6c8f7"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy: 80.00%\n"
          ]
        }
      ]
    }
  ]
}