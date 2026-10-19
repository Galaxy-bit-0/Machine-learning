# Machine-learning
Here I am posting machine learning algorithms from scratch

1#Logistic regression
# Logistic Regression From Scratch (NumPy)

A from-scratch implementation of binary logistic regression using only NumPy —
no `sklearn.linear_model` used for training. Built to understand the math
behind gradient descent, not just call a library function.

## Dataset
Uses the classic **Iris dataset** (`sklearn.datasets.load_iris`), reframed as
binary classification: predicting whether a flower is **Iris-Virginica**
(class 2) or not.

## What's implemented
- Sigmoid activation function
- Binary cross-entropy (log loss) cost function
- Gradient computation (for weights **and** bias) derived manually
- Batch gradient descent training loop
- Train/test split and accuracy evaluation
- Cost-over-epochs plot (to visualize convergence)
- Decision boundary visualization (2 features, for plotting purposes)

## Math

Prediction (sigmoid):
ŷ = 1 / (1 + e^(-(Xθ + b)))
Cost function (binary cross-entropy):
J(θ,b) = -(1/m) * Σ [ y·log(ŷ) + (1-y)·log(1-ŷ) ]
Gradients:
∂J/∂θ = (1/m) * Xᵀ(ŷ - y) #weight/theta
∂J/∂b = (1/m) * Σ(ŷ - y) #bias/theta_not
Parameter update:
θ := θ - α * ∂J/∂θ #weight/theta
b := b - α * ∂J/∂b #bias/theta_not


## Results
- Trained for 1000 epochs, learning rate α = 0.01
- Test accuracy: ~96–100% (Iris Virginica vs not Iris Virginica
-linearly separable problem, so high accuracy is expected, not evidence of overfitting on this specific dataset)

## Files
- `logistic_regression.py` — main script (training, evaluation, plots)

## How to run
```bash
pip install numpy matplotlib scikit-learn
python logistic_regression.py
```

## Notes
- The decision boundary plot uses only 2 of the 4 features (petal length,
  petal width) since boundaries can only be visualized in 2D. A separate
  mini version of the model is trained on just those 2 features for that
  plot; the main model uses all 4 features.
- This project is for learning purposes — `sklearn.linear_model.LogisticRegression`
  would be the standard choice for real-world use. #this is the standard model .fit() 

## Author
Balagani Shanmukh — built as a hands-on ML algorithm to understand logistic regression
and gradient descent at the implementation level.
