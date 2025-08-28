## Runtime Complexity Analysis from `experiments.py`

For Fitting, the time complexity for fitting a dataset into a Decision Tree is **O(n * m * log(n))** where **n** is the number of rows in the dataset and **m** is the number of features used for fitting.

For predicting, the time complexity is of the order **O(log(n))** or **O(h)**, where **h** is the maximum depth of the decision tree, whichever is lower.

Here is a plot from our analysis:

![Decision Tree Fit Time vs N (M=10)](images\Fit_time_vs_N_(M=10).png)
![Decision Tree Predict Time vs N (P=10)](images\Predict_time_vs_N_(M=10).png)
![Decision Tree Fit Time vs P (N=500)](images\Fit_time_vs_M_(N=500).png)
![Decision Tree Predict Time vs M (N=500)](images\Predict_time_vs_M_(N=500).png)


The plot for both fitting and predicting is matching the shape of theoretical graphs in essence.

