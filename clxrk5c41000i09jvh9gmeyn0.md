---
title: "Understanding Batch and Stochastic Gradient Descent and Normal Equation for Linear Regression"
seoTitle: "Gradient Descent and Normal Equation in Regression"
seoDescription: "Learn Batch and Stochastic Gradient Descent, and the Normal Equation for minimizing the cost function in linear regression"
datePublished: Sun Jun 23 2024 13:01:31 GMT+0000 (Coordinated Universal Time)
cuid: clxrk5c41000i09jvh9gmeyn0
slug: understanding-batch-and-stochastic-gradient-descent-and-normal-equation-for-linear-regression
cover: https://cdn.hashnode.com/res/hashnode/image/stock/unsplash/xG8IQMqMITM/upload/defd9c2cbba47d798c6c706512f5cc0d.jpeg
tags: gradient-descent, stochastic-gradient-descent, batch-gradient, machine-learning-math

---

### *In this article I'll be discussing about 3 things:*

* ### *Batch Gradient Descent*
    
* ### *Stochastic Gradient Descent*
    
* ### *Normal Equation*
    

## *Gradient Descent*

Gradient descent in machine learning is used to find the values of a function's parameters (coefficients), in our case θ, that minimise a cost function J(θ) as much as possible. To understand this better, read my previous [blog](https://vamshi.study/linear-regression-from-scratch) where I build a linear regression model from scratch and explained gradient descent in detail.

There are 3 types in gradient descent:

* Batch Gradient Descent
    
* Stochastic Gradient Descent
    
* Mini-Batch Gradient Descent
    

### *Batch Gradient Descent:*

In batch gradient descent, we use the entire training dataset for each iteration to update our parameters θ. In this method, we adjust our theta values to reduce the error from a chosen error function J(θ).

Let's say our error function is

$$\displaylines{ J(\theta) = \frac{1}{2}\sum(\hat{y}_i - y_i)^2 \\ here \;\; \hat{y}_i - predicted\;\; value \\ \quad \; y_i - actual\;\; value }$$

Here, **ŷ** is h(x), which is our hypothesis or prediction function that we use to predict our y values given an x value.

So our gradient descent algorithm would turn out to be

$$\displaylines{\frac{\partial}{\partial \; \theta_j} J(\theta) = \frac{\partial}{\partial \; \theta_j}\;\frac{1}{2}\sum [(h(x^i) - y^i)^2] \\ = \sum2\frac{1}{2}(h(x^i) - y^i)(\frac{\partial}{\partial \; \theta_j} ((θ_0x_0 + θ_1x_1 + θ_2x_2 + ... + θ_nx_n ) - y^i)) \\ Simplifying \;\; (\frac{\partial}{\partial \; \theta_j} ((θ_0x_0 + θ_1x_1 + θ_2x_2 + ... + θ_nx_n ) - y^i)) \;\; gives \;\; us\;\; x^i \\ This \;\; is \;\; because \;\; everything \;\; else\;\; is \;\; a \;\; constant\;\; which\;\; equates\;\; to \;\; 0\\ \\ = \sum(h(x) - y) x^j }$$

$$\displaylines{\theta_j := \theta_j - \alpha\sum(h_θ(x^i) - y^j)x_j^i) \\ here \;\; \alpha - learning \;\; rate }$$

Here, the **Σ** makes this "***batch***" gradient descent because it means we are subtracting the sum of errors from all the data samples, multiplying it by alpha, and updating theta.

### *Stochastic Gradient Descent*

Now imagine we have a large dataset. Batch gradient descent gives us accurate results, but it's quite slow. On the other hand, **stochastic gradient descent** is faster, though it provides less accurate results.

However, the results produced by stochastic gradient descent are "good enough," so we can use this method for large datasets.

The equation for stochastic gradient descent is

$$\displaylines{\theta_j := \theta_j - \alpha(h_θ(x^i) - y^j)x_j^i) \\ here \;\; \alpha - learning \;\; rate }$$

Here, we remove the **Σ** and update our parameter values for only one data sample or data point.

### *Normal Equation for Linear Regression*

For linear regression, there is a way to find the optimal parameter values in a single step without any iterations, called the Normal Equation. The Normal Equation for linear regression is a method to determine θ without needing iterations.

Let's get familiar with a few notations before we derive the normal equation.**Matrix Derivatives**

For a function f : R→ R mapping from n-by-d matrices to the real numbers, we define the derivative of f with respect to A to be:

$$\displaylines{ \nabla_Af(A) = \begin{bmatrix} \frac{\partial f}{\partial A_{11}} && ... &&\frac{\partial f}{\partial A_{1d}} \\ : && : && :\\ \frac{\partial f}{\partial A_{n1}} && ... &&\frac{\partial f}{\partial A_{nd}} \\ \end{bmatrix} }$$

for example

$$\displaylines{ if \;\; f(x) = A_{11} + A_{12}^2 \\ \\ and \;\; A = \begin{bmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \\ \end{bmatrix} \\ \\ then \;\; \nabla_A f(A) = \begin{bmatrix} 1 & 2A_{12} \\ 0 & 0 \end{bmatrix} }$$

Also here are a few things to keep in mind

$$\displaylines{ X = \begin{bmatrix} -(X^{(1)})^T- \\ -(X^{(2)})^T- \\ : \\ -(X^{(n)})^T- \\ \end{bmatrix} \\ \\ \hat{y} = \begin{bmatrix} y^{(1)} \\ y^{(2)} \\ : \\ y^{(n)} \\ \end{bmatrix} \\ \\ X\theta - \hat{y} = \begin{bmatrix} (X^{(1)})^T\theta - y^{(1)} \\ (X^{(2)})^T\theta - y^{(2)} \\ : \\ (X^{(n)})^T\theta - y^{(n)} \\ \end{bmatrix} \\ \\ ZᵀZ = ∑ Zᵢ² \\ => \frac{1}{2}(X\theta - \hat{y})^T(X\theta - \hat{y}) =\frac{1}{2} \sum(X\theta - \hat{y})^2 \\ => J(\theta) }$$

Now, to minimise our J(θ) let's find the derivative with respect to θ

$$\displaylines{ \nabla_{\theta}J(\theta) = \frac{1}{2}(X\theta - \hat{y})^T(X\theta - \hat{y}) \\ = \frac{1}{2}\nabla_\theta ((X\theta)^TX\theta - (X\theta)^T\hat{y} - \hat{y}^T(X\theta)+ \hat{y}^T\hat{y} \\ = \frac{1}{2}(X^TX\theta + X^TX\theta - X^TY - X^TY) \\ = (X^TX\theta - X^TY) \\ Now, set\;\;this\;\;equal\;\;to\;\;0 \\ => X^TX\theta = X^TY \\ => \theta = (X^TX)^{-1}X^TY }$$

Now that is the equation of θ

$$\theta = (X^TX)^{-1}X^TY$$

That is the formula for normal equation for linear regression.

That concludes this blog about understanding gradient descent and normal equation for linear regression

I'll be teaching about Logically Weighted Regression and Probabilistic Interpretation in my next article, so stay tuned!

If you enjoyed this article, please consider [buying me a coffee](https://buymeacoffee.com/vamshi6).

Follow me on my social media and consider subscribing, as I'll be covering more Machine Learning algorithms "from scratch" and their math.