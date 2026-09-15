# Rest-Mex 2025: Research on Sentiment Analysis Task for Mexican Tourist Texts (FrogCode Team) 🇲🇽 🇨🇴

In this repository we address the task of joint classification of sentiment polarity, destination type, 
and Magic Town from Spanish-language tourist reviews, as proposed by the 
[Rest-Mex 2025 challenge](https://sites.google.com/cimat.mx/rest-mex-2025/sentiment-analysis-task?authuser=0), 
part of the [IberLEF 2025](https://sites.google.com/view/iberlef-2025/tasks?authuser=0). 

We hereby propose a lightweight, hierarchical attention-based model that leverages
the inherent structure of review data. Specifically, the model posits that words form reviews, and reviews are
grouped by town. The proposed approach is predicated on the Hierarchical Attention Network (HAN), with
extensions to a multi-output setting that are concomitant with the preservation of efficiency and interpretability

## Project structure

The project is structured as follows:

```
NLP_Rest_Mex2025/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── data/
├── images/
├── notebooks/
│   ├── data_analysis_ezau.ipynb
│   ├── data_analysis.ipynb
│   ├── exploratoryAnalysis.ipynb
│   └── train_model.ipynb
└── src/
    ├── inference/
    │   └── predict.py
    ├── models/
    │   └── hierarchical_model.py
    ├── pipelines/
    │   ├── data_loader.py
    │   ├── data_pipeline.py
    │   ├── dataset.py
    │   └── preprocessor.py
    ├── train/
    │   └── train.py
    └── visualization/
        └── plot_generator.py
```

## Citing

This repo is part of the [research paper](https://ceur-ws.org/Vol-4098/RESTMEX2025_paper14.pdf). If you find this code useful in your research, please consider citing:

```
@inproceedings{torres2025hierarchical,
  title={Hierarchical Attention Networks for Multilabel Sentiment Analysis in Spanish Reviews of Mexican Magic Towns.},
  author={Torres, Ezau Faridh Torres and C{\'a}rdenas, Edison David Serrano},
  booktitle={IberLEF@ SEPLN},
  year={2025}
}
```

## License

This project is licensed under the GPL-3.0 license. See the LICENSE file for details.
