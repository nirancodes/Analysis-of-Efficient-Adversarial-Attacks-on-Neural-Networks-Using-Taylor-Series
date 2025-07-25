## 📌 Disclaimer

This project was created for academic purposes only. The goal is to demonstrate the application of calculus to adversarial machine learning—not to promote or facilitate malicious use.

---


# Taylor Series & Adversarial Attacks on Neural Networks

This project is a Calculus II (MAT187) mini-exploration into how fundamental mathematical concepts—specifically Taylor Series approximations—can be used by adversaries to fool neural networks. Through both theoretical analysis and practical implementation, we examine how small, precisely crafted perturbations can lead machine learning models to make incorrect predictions while remaining undetectable to humans.
Taylor approximations aren't just a calculus exercise—they're powerful tools that attackers can use to break deep learning systems with surgical precision. This project demonstrates how math and machine learning intersect in both innovative and adversarial ways.

---

## ✨ Project Highlights

- **Mathematical Foundation**  
  Applies first and second-order Taylor expansions to model how perturbations alter loss functions in neural networks.

- **Adversarial Attack Methods**  
  Compares Fast Gradient Sign Method (FGSM, 1st-order) with TEAM (Taylor Expansion-based Attack Method, 2nd-order), demonstrating their impact on model predictions and visual detectability.

- **Robustness Quantification**  
  Uses trapezoidal numerical integration to approximate model robustness as a function of perturbation size (ϵ).

- **Black-Box Implementation**  
  Developed using PyTorch, leveraging a small dataset of real-world autonomous driving cone images to simulate attacks on a pretrained model.

---

## 📘 Learning Objectives

- Understand how hackers exploit gradients to maximize neural network loss functions.
- Apply Taylor series to estimate perturbation effects on model outputs.
- Analyze how curvature (via Hessians) enables stealthier second-order attacks.
- Use numerical integration techniques from calculus to measure attack robustness.

---

## 🔍 Structure

- `Part 1`: Building the mathematical approximation
- `Part 2`: Optimizing perturbations under ε-constraints
- `Implementation`: Comparing FGSM vs. TEAM attacks on vision tasks
- `Quantification`: Using trapezoidal rule to estimate robustness degradation

---

## 🛠️ Tech Stack

- **Python, PyTorch, Matplotlib** — for neural network modeling and perturbation generation + visualization
- **Desmos** — for visualizing Taylor polynomials  
- **MAT187 Calculus** — core mathematical tools for analysis

---

## 👩‍🏫 Authors

- Niranjana Naveen Nambiar  
- Michelle Xu  
- Jade Boongaling  
- Harshita Srikanth  
Lecturer: Prof. Armanpreet Pannu

---

