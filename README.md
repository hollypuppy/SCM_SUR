# Generalized Synthetic Control Methods with Multiple Outcomes

This model is a replication and extension of Pinkney (2021). Pinkney's model is developed upon Xu's Generalized synthetic control method (2017). It uses an interactive fixed effects (IFE) model as Xu does. However, rather than fitting the latent factors only on control group data (for all periods) while fitting the loadings in a separate procedure, in only pre-treatment periods, Pinkney's model uses more of the data for estimation while simultaneously estimating the latent factors.

### 1. Framework of Pickney's Model
Suppose there are $J$ units, and each unit is observed for $T$ periods.

<div align="center">
  <img width="233" alt="image" src="https://github.com/user-attachments/assets/bb6e5974-46dd-4ab0-96c4-fcd6d3833ba5">
</div> 

where
- $y_j$ is the outcome, where $j\in J$
- $F$ is a $T\times L$ matrix, representing the latent factors in this model
- $\beta_j$ is a $L\times 1$ vector, which is the factor loading of the latent factors
- $X$ is a matrix of observed covariates with coefficients $\gamma$
- $\Delta$ is a size $T$ vector that is the same across all $J$ series, which can be viewed as a unit-fixed effect.

### 2. Model with Multiple Outcomes

As an extension of this model, suppose for each unit $j$ at each time period $t$, we observe $K$ outcomes. 
i.e., $J$ units, $T$ time periods, $K$ outcomes, $L$ latent factors.

The model becomes

<div align="center">
  <img width="207" alt="image" src="https://github.com/user-attachments/assets/8e033200-74a9-42b4-94fa-7aaefb511d32">
</div> 

where
- $\mathbf{y}_{jt}$ is a $K$-vector
- $\boldsymbol{\Sigma}_{\epsilon}$ is a $K\times K$ matrix
- <img width="124" alt="image" src="https://github.com/user-attachments/assets/16247695-6463-4ef6-833b-23ed3e7b6f43"> where $\psi_{jtk} = f_t\Sigma_k\beta_j$, with $f_t$ a size $L$ row vector, $\Sigma_k$ a $L\times L$ matrix, and $\beta_j$ a $L$-vector.

### 2.1 Priors

$F \sim N(0,1)$

$\gamma \sim N(0,1)$

$\Delta \sim N(0,2)$

$\kappa \sim N(0,1)$

$\sigma \sim N(0,1)$
    
$(\beta_{l,j}|\lambda_l,\eta_j,\tau)\sim N(0,\lambda_l)$

$(\lambda_l|\eta_j,\tau)\sim Cauchy^+(0,\tau\eta_j)$

$\eta_j \sim Cauchy^+(0,1)$

$\tau \sim Cauchy^+(0,1)$

### 3. Application

I use the German Reunification as the case study. The German reunification data is from (Hainmueller, 2014). The data consist of GDP measured in Purchasing Power Parity (PPP)-adjusted 2002 USD of 17 OECD member countries from 1960 - 2003. The pre-intervention period is from 1960 to 1990 - at which point German reunification occurred. 

The results are shown as follows. It aligns relatively closely with the observed outcome values in the pre-intervention period, especially for log(GDP) and 1/GDP.

![SUR_SCM_3outcomes](https://github.com/user-attachments/assets/2ef5930d-5965-4e83-973a-1263d4e8740d)

### 4. References

Hainmueller, J. (2014). Replication data for: Comparative politics and the synthetic control method. https://doi.org/10.7910/DVN/24714

Pinkney, S. (2021). An improved and extended Bayesian synthetic control. arXiv. USA. Retrieved from http://arxiv.org

Xu, Y. (2017). Generalized synthetic control method: Causal inference with interactive fixed effects models. Political Analysis, 25(1), 57-76. https://doi.org/10.1017/pan.2016.2
