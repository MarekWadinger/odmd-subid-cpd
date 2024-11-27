# Change-Point Detection based on Online DMD with Control

[![Quality and Tests](https://github.com/MarekWadinger/odmd-subid-cpd/actions/workflows/code-quality-tests.yml/badge.svg)](https://github.com/MarekWadinger/odmd-subid-cpd/actions/workflows/code-quality-tests.yml)
[![codecov](https://codecov.io/gh/MarekWadinger/odmd-subid-cpd/branch/main/graph/badge.svg?token=BIS0A7CF1F)](https://codecov.io/gh/MarekWadinger/odmd-subid-cpd)

As the energy sector races toward radical climate action, scaling new solutions is essential. Automated control has been crucial to efficient operations, and detecting unforeseen critical shifts can be a game-changer for safety.

![ilustratory example](/publications/figures/web-app-chart.png)

This repository presents a novel approach to change-point detection (CPD) that leverages Online Dynamic Mode Decomposition with Control (ODMDwC). Designed specifically for complex industrial systems where timely detection of behavioral shifts is critical, this method captures both spatial and temporal system patterns and adapts dynamically to non-linear system changes due to factors like aging and seasonality.

Our ODMDwC-based method addresses real-world challenges of non-uniform data streams in safety-critical systems by providing reliable CPD without dependency on exhaustive physical models. It leverages control input to enhance change detection performance, yielding robust and intuitive results even in environments with high noise.

## 🚀 Features

- **Adaptive Linearization**: ODMDwC dynamically adapts to system behavior, maintaining a correspondence between detected changes and their actual extent.
- **Truncated ODMDwC with Time-Delay Embeddings**: Incorporates higher-order time-delay embeddings to improve noise robustness and capture broadband features.
- **Enhanced CPD Performance**: The method outperforms SVD-based and other common CPD methods on benchmark datasets, improving detection accuracy while reducing false positives.
- **Intuitive Hyperparameter Tuning**: Offers practical guidelines for hyperparameter selection to streamline model application.

## Benchmark Evaluation

We validated our approach on both synthetic and real-world datasets, including:

1. [SKAB](https://github.com/waico/SKAB) Laboratory Water Circulation System

    | **Algorithm**                  | **NAB (standard)** | **NAB (low FP)** | **NAB (low FN)** |
    |--------------------------------|--------------------|------------------|------------------|
    | Perfect detector               | 54.77              | 54.11            | 56.99            |
    | **CPD-DMD (\(t=0\))**          | **34.29**          | 23.21            | **42.54**        |
    | **CPD-DMD (\(t=0.0025\))**     | 33.43              | **23.28**        | 41.71            |
    | MSCRED                         | 32.42              | 16.53            | 40.28            |
    | Isolation forest               | 26.16              | 19.50            | 30.82            |
    | T-squared+Q (PCA)              | 25.35              | 14.51            | 31.33            |
    | Conv-AE                        | 23.61              | 21.54            | 27.55            |
    | LSTM-AE                        | 23.51              | 20.11            | 25.91            |
    | T-squared                      | 19.54              | 10.20            | 24.31            |
    | MSET                           | 13.84              | 10.22            | 17.37            |
    | Vanilla AE                     | 11.41              | 6.53             | 13.91            |
    | Vanilla LSTM                   | 11.31              | -3.80            | 17.25            |
    | Null detector                  | 0.00               | 0.00             | 0.00             |

2. [CATS](https://www.kaggle.com/datasets/patrickfleith/controlled-anomalies-time-series-dataset) Controlled Anomalies Dataset

    | **Algorithm**                  | **NAB (standard)** | **NAB (low FP)** | **NAB (low FN)** |
    |--------------------------------|--------------------|------------------|------------------|
    | Perfect detector               | 30.21              | 29.89            | 31.28            |
    | **MSCRED**                     | **37.19**          | 13.46            | **47.18**        |
    | **CPD-DMD (\(t=0\))**          | 25.66              | **20.62**        | 29.84            |
    | **CPD-DMD**                    | 17.84              | 15.01            | 20.06            |
    | Isolation forest (\(c=3.8\%\)) | 17.81              | 15.84            | 20.00            |
    | T-squared+Q (PCA)              | 11.80              | 11.40            | 12.30            |
    | LSTM-AE                        | 11.39              | 11.26            | 11.69            |
    | T-squared                      | 15.15              | 14.98            | 15.71            |
    | MSET                           | 14.48              | 13.43            | 15.60            |
    | Vanilla AE                     | 2.52               | 2.44             | 2.77             |
    | Vanilla LSTM                   | 0.73               | 0.70             | 0.82             |
    | Conv-AE                        | 0.15               | 0.14             | 0.18             |
    | Null detector                  | 0.00               | 0.00             | 0.00             |

## 📜 Citation

If you use this platform for academic purposes, please cite our publication:

```bibtex
@misc{wadinger2024changepointdetectionindustrialdata,
    author    ={Marek Wadinger and Michal Kvasnica and Yoshinobu Kawahara},
    note      = {Submitted to Applied Energy},
    title     ={Change-Point Detection in Industrial Data Streams based on Online Dynamic Mode Decomposition with Control},
    url       ={https://arxiv.org/abs/2407.05976},
    year      ={2024},
}
```

## 👐 Contributing

Feel free to contribute in any way you like, we're always open to new ideas and
approaches.

- Feel welcome to
[open an issue](https://github.com/MarekWadinger/odmd-subid-cpd/issues/new/choose)
if you think you've spotted a bug or a performance issue.

## 🛠 For Developers

### Installation (for Local Use)

If you wish to run the platform locally, follow the steps below:

1. Clone the repository:

    ```sh
    git clone https://github.com/MarekWadinger/odmd-subid-cpd.git
    ```

2. Navigate to the project folder:

    ```sh
    cd odmd-subid-cpd
    ```

3. Create a virtual environment:

    ```sh
    python -m venv --upgrade-deps .venv
    source .venv/bin/activate
    ```

4. Install the required dependencies:

    ```sh
    pip install -r requirements.txt
    ```
