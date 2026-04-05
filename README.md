# Adversarial Attacks for CV Models with Foolbox

## Overview

This material was prepared as part of a study on data poisoning at Central University, under the mentorship of experts from AIRI Safe AI. The notebook includes:

- CUB-200-2011 data loading and holdout split
- ResNet18 fine-tuning on bird species classification
- Foolbox evaluation with PGD, Boundary Attack, and FGSM
- Adversarial training with a lightweight PGD inner loop
- Randomized smoothing experiments
- Benign and adversarial accuracy reporting
- Confusion matrix and example visualizations

## Dataset

The notebook uses the CUB-200-2011 dataset, which contains 11.8k images of 200 bird species, and keeps a separate test split for evaluation.

<img width="648" height="329" alt="image" src="https://github.com/user-attachments/assets/91eb973b-613d-4282-9c06-616f8fd7fd92" />

## Model

The model is based on ResNet-18 and was fine-tuned on CUB-200-2011, achieving 73% accuracy on the held-out test set. The entire training process was conducted on a T4 GPU with 16 GB VRAM in Google Colab.

## Benign and adversarial accuracy

For a classifier $f$, benign accuracy measures performance on clean images:

$$Acc_{\text{benign}} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\bigl(f(x_i) = y_i\bigr)$$

Adversarial accuracy measures performance after an attack has been applied:

$$Acc_{\text{adv}} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\bigl(f(x_i^{\text{adv}}) = y_i\bigr)$$

Here, $x_i^{\text{adv}}$ is the adversarially perturbed version of the original image. 

## PGD attack

Projected Gradient Descent is the main white-box attack used in the notebook. It repeatedly updates the input in the direction of the loss gradient and projects the result back into the allowed $\varepsilon$-ball:

$$x_{t+1} = \Pi_{x+\mathcal{S}}\Bigl(x_t + \alpha \cdot \operatorname{sign}\bigl(\nabla_{x_t} L(\theta, x_t, y)\bigr)\Bigr)$$

On the holdout subset, the baseline model drops sharply under PGD, which confirms that standard fine-tuning alone is not enough to make the classifier robust. The adversarial nose remains invisible for human's eye:

<img width="449" height="300" alt="Untitled" src="https://github.com/user-attachments/assets/cca7d044-c91c-47cf-a5c8-7888f4bbb153" />

## Boundary attack

Boundary Attack is a decision-based black-box attack. It starts from an adversarial point and then moves along the decision boundary while keeping the input adversarial:

$$x_{t+1} = \operatorname{proj}\bigl(x_t + \eta \cdot \vec{d}\bigr) \quad \text{s.t.} \quad f(x_{t+1}) \neq f(x)$$

Unlike PGD, the perturbations are not constrained by a simple gradient step, so the attacked images remain visually meaningful while the added noise is still visible:

<img width="676" height="390" alt="Untitled" src="https://github.com/user-attachments/assets/cbf78c0c-bcdd-4e81-a225-d5e76911b306" />

## FGSM attack

Fast Gradient Sign Method is the one-step version of the gradient-based attack family:

$$x_{\text{adv}} = x + \varepsilon \cdot \operatorname{sign}\bigl(\nabla_x L(\theta, x, y)\bigr)$$

We compare baseline model and the adversarially trained model on the same FGSM sweep. The baseline curve drops much earlier, while the adversarially trained model stays stable for noticeably larger values of $\varepsilon$. 

<img width="529" height="350" alt="Untitled" src="https://github.com/user-attachments/assets/0b86e197-e409-46b4-8d91-afefaa96413f" />

## Randomized smoothing

Randomized smoothing evaluates the classifier under Gaussian noise and uses majority vote over multiple noisy copies of the same input:

$$g(x) = \arg\max_{c \in \mathcal{Y}} \mathbb{P}(f(x + \varepsilon) = c),
\quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$

The adversarially trained model stays more stable as $\sigma$ grows, while the baseline degrades much faster. It is a sanity check to compare the models' robustness to noise.

<img width="515" height="350" alt="Untitled" src="https://github.com/user-attachments/assets/b37e7b37-bcc3-40a2-90c7-d544b6058a89" />
