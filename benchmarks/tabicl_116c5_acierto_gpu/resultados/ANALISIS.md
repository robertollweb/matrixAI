# Kit 116-C5: TabICL (GPU) frente a TabM y a LightGBM

Regla: `REGLA.md`, sha256 `a7fa88a988f7cb08e01f5509167f338b4ca55bab17425f342125f7d55713f259`. Margen de equivalencia ±0.005. Diferencia = TabICL GPU − rival (métrica cruda, más es mejor), media de pliegues emparejados con su intervalo bootstrap al 95 %.

| conjunto | métrica | pliegues | TabICL GPU | vs TabM | vs LightGBM | control GPU−CPU |
|---|---|---|---|---|---|---|
| balance-scale | accuracy | 15/15 | 0.9799 | +0.0143 [+0.0095, +0.0196] GANA | +0.1074 [+0.0989, +0.1159] GANA | +0.0000 (máx abs Δ 0.0000, n=15) |
| dresses-sales | auroc | 15/15 | 0.5517 | -0.0219 [-0.0428, -0.0030] PIERDE | -0.0288 [-0.0467, -0.0120] PIERDE | +0.0000 (máx abs Δ 0.0000, n=15) |
| diabetes | auroc | 15/15 | 0.8079 | +0.0105 [+0.0079, +0.0131] GANA | +0.0441 [+0.0357, +0.0541] GANA | +0.0000 (máx abs Δ 0.0000, n=15) |
| climate-model-simulation-crashes | auroc | 15/15 | 0.9741 | +0.0444 [+0.0343, +0.0550] GANA | +0.0248 [+0.0187, +0.0307] GANA | +0.0000 (máx abs Δ 0.0000, n=15) |
| kc2 | auroc | 15/15 | 0.9293 | +0.0098 [+0.0079, +0.0115] GANA | +0.0414 [+0.0294, +0.0538] GANA | +0.0000 (máx abs Δ 0.0000, n=15) |
| yeast | accuracy | 12/12 | 0.6093 | +0.0210 [+0.0148, +0.0268] GANA | +0.0299 [+0.0179, +0.0425] GANA | +0.0000 (máx abs Δ 0.0000, n=12) |
| Moneyball | r2 | 15/15 | 0.9437 | +0.0042 [+0.0022, +0.0064] GANA | +0.0107 [+0.0097, +0.0118] GANA | +0.0000 (máx abs Δ 0.0000, n=15) |
| pc1 | auroc | 15/15 | 0.8695 | +0.0724 [+0.0463, +0.1038] GANA | +0.0881 [+0.0764, +0.0994] GANA | +0.0000 (máx abs Δ 0.0000, n=15) |
| wilt | auroc | 15/15 | 0.9993 | +0.0002 [+0.0001, +0.0004] empate (equivalente) | +0.0033 [+0.0028, +0.0039] empate (equivalente) | — |
| pc4 | auroc | 15/15 | 0.9556 | +0.0145 [+0.0100, +0.0186] GANA | +0.0249 [+0.0224, +0.0275] GANA | — |
| wine_quality | r2 | 15/15 | 0.5524 | +0.0567 [+0.0525, +0.0608] GANA | +0.1222 [+0.1188, +0.1255] GANA | — |
| sick | auroc | 15/15 | 0.9985 | +0.0005 [+0.0001, +0.0009] empate (equivalente) | +0.0041 [+0.0015, +0.0079] GANA | — |
| house_prices_nominal | r2 | 15/15 | 0.9308 | +0.0020 [-0.0004, +0.0042] empate (equivalente) | +0.0292 [+0.0258, +0.0327] GANA | — |
| pendigits | accuracy | 15/15 | 0.9965 | +0.0046 [+0.0041, +0.0050] GANA | +0.0074 [+0.0066, +0.0082] GANA | — |
| ozone-level-8hr | auroc | 15/15 | 0.9444 | +0.0120 [+0.0064, +0.0183] GANA | +0.0104 [+0.0066, +0.0146] GANA | — |
| jm1 | auroc | 15/15 | 0.7755 | +0.0430 [+0.0389, +0.0467] GANA | +0.0329 [+0.0296, +0.0362] GANA | — |
| us_crime | r2 | 15/15 | 0.7330 | +0.0171 [+0.0141, +0.0198] GANA | +0.0179 [+0.0127, +0.0233] GANA | — |
| PhishingWebsites | auroc | 15/15 | 0.9980 | +0.0015 [+0.0013, +0.0017] empate (equivalente) | +0.0037 [+0.0035, +0.0038] empate (equivalente) | — |
| optdigits | accuracy | 15/15 | 0.9971 | +0.0059 [+0.0052, +0.0066] GANA | +0.0094 [+0.0081, +0.0106] GANA | — |
| house_16H | r2 | 5/5 | 0.7558 | +0.0332 [+0.0232, +0.0441] GANA | +0.0936 [+0.0910, +0.0964] GANA | — |
| mfeat-factors | accuracy | 15/15 | 0.9943 | +0.0128 [+0.0102, +0.0153] GANA | +0.0172 [+0.0150, +0.0195] GANA | — |
| house_sales | r2 | 5/5 | 0.8913 | -0.0193 [-0.0257, -0.0127] PIERDE | +0.0233 [+0.0181, +0.0275] GANA | — |
| SAT11-HAND-runtime-regression | r2 | 15/15 | 0.7122 | -0.0923 [-0.0997, -0.0859] PIERDE | -0.0226 [-0.0272, -0.0177] PIERDE | — |
| micro-mass | accuracy | 15/15 | 0.8590 | -0.0265 [-0.0419, -0.0106] PIERDE | -0.0047 [-0.0189, +0.0100] empate (no concluyente) | — |

## TabICL frente a TabM y frente a LightGBM: gana / pierde / empate dentro del margen

| rival | gana | pierde | empate equivalente | empate no concluyente | no comparable |
|---|---|---|---|---|---|
| TabM (119-C3) | 16 | 4 | 4 | 0 | 0 |
| LightGBM (113-C4) | 19 | 2 | 2 | 1 | 0 |
