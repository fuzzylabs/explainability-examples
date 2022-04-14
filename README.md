# Explainability Examples

## What is explainability?
Explainability tools break down model inferences into how each input parameter influenced the final result.
This has two main uses:
* Feedback on individual inferences. If a user is rejected for a mortgage what can they do to improve? Increase their
credit score, up their deposit, etc...
* Monitoring for systematic bias in the model, by gathering values across a set of inferences. We can verify that our
model is making reasonable decisions, and that we're not making incorrect or unjustified decisions, for example
rejecting mortgage applicants users based on their gender.

it's also notable that their is a tradeoff between accuracy and explainability of models, see the `Algorithmic Choice
and Code` section of [this article](Algorithmic Choice and Code) for more details.

## What is shap
[Shap](https://shap.readthedocs.io/en/latest/index.html) is an academic python library that uses a game theory approach
to graph model explainability.
[This article](https://towardsdatascience.com/shap-explained-the-way-i-wish-someone-explained-it-to-me-ab81cc69ef30)
gives a breakdown of how specifically this is done.

## Articles
[Simplifying the Role of Explainability in the MLOps Cycle](https://www.persistent.com/blogs/simplifying-the-role-of-explainability-in-the-mlops-cycle/)

[Shap values explained](https://towardsdatascience.com/shap-explained-the-way-i-wish-someone-explained-it-to-me-ab81cc69ef30)

## Installation and running

Run `pip install -r requirements.txt` to install project dependencies.
Run `python main.py` to train an iris model and update the saved explainability graphs.

A force plot is saved to `shap.html` and a summary plot is saved to `shap.png`.
