Financial Disclosure Anomaly Detection Using Jraph-based GNNs
This repository contains the code and data for the paper:

“AI-Enhanced Data Mining in the Financial Domain: Methods and Applications — Leveraging Deep Learning and NLP to Detect Anomalies and Combat Financial Crime”
Submitted to the Workshop on AI for Financial Crime Fight (AI4FCF) at the IEEE International Conference on Data Mining (ICDM).

Project Overview
This example demonstrates the use of JAX, Haiku, and Google DeepMind’s Jraph library to detect anomalies in financial disclosures. The project applies deep learning to identify patterns of fraudulent or misleading reporting behavior by analyzing both structured financial data (ratios) and unstructured narrative disclosures (text).

This repository is part of a multi-stage research project involving SEC filings submitted by South American firms cross-listed in the U.S.

Project Stages
Stage 1 (Current): ~130 filings used to develop and evaluate baseline models
Ratio-only and Text-only models complete
Combined input models in progress

Stage 2 (Planned): Expand to ~6,000 filings (2013–2023) for large-scale modeling

Included in This Release
This initial contribution focuses on Stage 1 and includes 6 fully working models:

Model Type	  Input Type
GNN        	  Ratio-only
CNN	          Ratio-only
GRU	          Ratio-only
GNN	          Text-only
CNN	          Text-only
GRU	          Text-only

All models are implemented in pure JAX and use the functional programming patterns encouraged in Jraph-based modeling.

Repository Contents


├── Text.csv                  # Labeled unstructured financial text
├── ratio.csv                # Labeled structured financial ratios
├── dataset.py               # Data preprocessing and graph construction
├── model.py                 # Model definitions (GNN/CNN/GRU)
├── run_gnn_ratio.py         # GNN model using financial ratios
├── run_gnn_text.py          # GNN model using text
├── run_cnn_ratio.py         # CNN model using ratios
├── run_cnn_text.py          # CNN model using text
├── run_gru_ratio.py         # GRU model using ratios
├── run_gru_text.py          # GRU model using text
├── train.py (optional)      # [Planned] Unified runner interface
├── README.md
├── requirements.txt



Install required packages:

pip install -r requirements.txt


Run any model:

python run_gnn_ratio.py
python run_cnn_text.py


Sample Results (Stage 1)

| Model | Input Type | Accuracy | F1 Score | Notes                       |
| ----- | ---------- | -------- | -------- | --------------------------- |
| GNN   | Ratio-only | 1.00     | 1.00     | Small dataset, overfit      |
| CNN   | Ratio-only | 1.00     | 1.00     | Same as above               |
| GRU   | Ratio-only | 1.00     | 1.00     | Same as above               |
| GNN   | Text-only  | 1.00     | 0.00     | Predicted only one class    |
| CNN   | Text-only  | 0.40     | 0.57     | Struggled with sparse text  |
| GRU   | Text-only  | 0.60     | 0.00     | Text boundaries not learned |

Combined input models are under development and will be released in Stage 1.2.


Notes on Data

This example uses real-world data from 130 cross-listed firms.

Text.csv contains labeled narrative financial statements

ratio.csv contains 6 calculated financial ratios

These are provided here for academic demonstration only.

Combined data files have been intentionally withheld to protect project integrity. They will be released as part of Stage 1.2 or upon request.

Note on Current Limitations and Future Work:
The current models exhibit limitations such as overfitting due to small sample size, difficulty learning text boundaries, poor performance on sparse textual data, and class imbalance (e.g., predicting only one class in some cases).
These challenges will be addressed in Stage 2, which will expand the dataset from approximately 150 to over 6,000 SEC filings. Stage 2 will also introduce combined input models (text + ratio) and implement further architectural and training refinements.
This work is expected to progress over the next three months.

Acknowledgments
We gratefully acknowledge the open-source contributions of Google DeepMind, whose development of the Jraph library has been foundational to this research. This implementation follows Jraph’s functional modeling style using GraphsTuple and GraphNetwork.
While not officially affiliated, this example aspires to extend the Jraph ecosystem with a real-world application in financial anomaly detection.

License
All code is shared for academic, non-commercial use. Dependencies are used under their respective open-source licenses.

Contact / Access
For questions or access to additional data (e.g., combined input models or full Stage 2 data), please contact:

Wesley Leeroy
GitHub: github.com/wesleyleeroy
