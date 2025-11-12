# Retail Rocket EDA Summary

- Dataset size: 2.76M events with 1.41M visitors and 235k items.
- Event imbalance: ~96.7% views, 2.5% add-to-cart, 0.8% transactions.
- Item 461686 is the top product across add-to-cart and purchase counts (highlighted in Kaggle notebooks).
- Missing values: transactionid absent for ~2.73M rows (expected for non-purchase events).
- Duplicate rows: 460 repeated logs (~0.017% of data); negligible impact.
- Daily activity shows steady engagement with small weekend dips and a mid-May spike, matching Kaggle analyses.
- Conclusion: feedback is sparse and popularity-driven; models must handle rare positive signals and potential cold starts.
