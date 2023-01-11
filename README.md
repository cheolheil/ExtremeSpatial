# ExtremeSpatial
Extreme Spatial Model for Python Users

Please refer to `demo.ipynb` to check how to use the model. Currently, it has the following models:
- `LatentExtreme`: estimates marginal distribution for observed locations.
- `SmithModel`: max-stable model from [Smith (1990)](https://www.researchgate.net/profile/Stilian-Stoev-2/publication/271095588_Upper_bounds_on_value-at-risk_for_the_maximum_portfolio_loss/links/564b824b08aeab8ed5e7694a/Upper-bounds-on-value-at-risk-for-the-maximum-portfolio-loss.pdf)
- `SchlatherModel`: max-stable model from [Schlather (2002)](https://link.springer.com/article/10.1023/A:1020977924878)
Max-stable models use pairwise composite likelihood estimation.

The current version is very CRUDE because of my laziness. More (e.g., Reich-Shaby, selective likelihood estimation, extremal coefficient, etc.) will be verified and added soon. Any question, assitance, discussion is welcome.
