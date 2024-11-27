"""
This module is modified part of evaluation from library [tsad](https://github.com/waico/tsad)
"""

from typing import Any

import numpy as np
import pandas as pd


def filter_detecting_boundaries(detecting_boundaries):
    """
    Filters out empty sublists from a list of detecting boundaries.

    Args:
        detecting_boundaries (list of list): A list containing sublists,
                                             where each sublist represents a boundary.

    Returns:
        list of list: A list containing only the non-empty sublists from the input.

    Examples:
        >>> filter_detecting_boundaries([[1, 2], [], [1, 2]])
        [[1, 2], [1, 2]]

        >>> filter_detecting_boundaries([[], []])
        []
    """
    _detecting_boundaries = []
    for couple in detecting_boundaries.copy():
        if len(couple) != 0:
            _detecting_boundaries.append(couple)
    detecting_boundaries = _detecting_boundaries
    return detecting_boundaries


def single_detecting_boundaries(
    true_series,
    true_list_ts,
    prediction,
    portion,
    window_width,
    anomaly_window_destination,
    intersection_mode,
):
    """
    Extract detecting_boundaries from series or list of timestamps
    """

    if (true_series is not None) and (true_list_ts is not None):
        raise Exception("Choose the ONE type")
    elif true_series is not None:
        true_timestamps = true_series[true_series == 1].index
    elif true_list_ts is not None:
        if len(true_list_ts) == 0:
            return [[]]
        else:
            true_timestamps = true_list_ts
    else:
        raise Exception("Choose the type")
    #
    detecting_boundaries = []
    td = (
        pd.Timedelta(window_width)
        if window_width is not None
        else pd.Timedelta(
            (prediction.index[-1] - prediction.index[0])
            / (len(true_timestamps) + 1)
            * portion
        )
    )
    for val in true_timestamps:
        if anomaly_window_destination == "lefter":
            detecting_boundaries.append([val - td, val])
        elif anomaly_window_destination == "righter":
            detecting_boundaries.append([val, val + td])
        elif anomaly_window_destination == "center":
            detecting_boundaries.append([val - td / 2, val + td / 2])
        else:
            raise RuntimeError("choose anomaly_window_destination")

    # block for resolving intersection problem:
    # important to watch right boundary to be never included to avoid windows intersection
    if len(detecting_boundaries) == 0:
        return detecting_boundaries

    new_detecting_boundaries = detecting_boundaries.copy()
    intersection_count = 0
    for i in range(len(new_detecting_boundaries) - 1):
        if (
            new_detecting_boundaries[i][1]
            >= new_detecting_boundaries[i + 1][0]
        ):
            # transform print to list of intersections
            # print(f'Intersection of scoring windows {new_detecting_boundaries[i][1], new_detecting_boundaries[i+1][0]}')
            intersection_count += 1
            if intersection_mode == "cut left window":
                new_detecting_boundaries[i][1] = new_detecting_boundaries[
                    i + 1
                ][0]
            elif intersection_mode == "cut right window":
                new_detecting_boundaries[i + 1][0] = new_detecting_boundaries[
                    i
                ][1]
            elif intersection_mode == "cut both":
                _a = new_detecting_boundaries[i][1]
                new_detecting_boundaries[i][1] = new_detecting_boundaries[
                    i + 1
                ][0]
                new_detecting_boundaries[i + 1][0] = _a
            else:
                raise Exception("choose the intersection_mode")
    # print(f'There are {intersection_count} intersections of scoring windows')
    detecting_boundaries = new_detecting_boundaries.copy()
    return detecting_boundaries


def check_errors(my_list):
    """
    Check format of input true data.

    Args:
        my_list (list): Uniform format of true data (See evaluate.evaluate).

    Returns:
        int: Depth of list, or variant of processing.

    Raises:
        Exception: If non-uniform data format is found at any level.
    """
    assert isinstance(my_list, list)
    mx = 1
    #     ravel = []
    level_list: dict[int, Any] = {}

    def check_error(my_list):
        return not (
            (all(isinstance(my_el, list) for my_el in my_list))
            or (all(isinstance(my_el, pd.Series) for my_el in my_list))
            or (all(isinstance(my_el, pd.Timestamp) for my_el in my_list))
        )

    def recurse(my_list, level=1):
        nonlocal mx
        nonlocal level_list

        if check_error(my_list):
            raise Exception(
                f"Non uniform data format in level {level}: {my_list}"
            )

        if level not in level_list.keys():
            level_list[level] = []  # for checking format

        for my_el in my_list:
            level_list[level].append(my_el)
            if isinstance(my_el, list):
                mx = max([mx, level + 1])
                recurse(my_el, level + 1)

    recurse(my_list)
    for level in level_list:
        if check_error(level_list[level]):
            raise Exception(
                f"Non uniform data format in level {level}: {my_list}"
            )

    if 3 in level_list:
        for el in level_list[2]:
            if not ((len(el) == 2) or (len(el) == 0)):
                raise Exception(
                    f"Non uniform data format in level {2}: {my_list}"
                )
    return mx


def extract_cp_confusion_matrix(
    detecting_boundaries, prediction, point=0, binary=False
):
    """
    Extracts the confusion matrix for change point detection.

    Args:
        detecting_boundaries (list of list of int): List of pairs of start and end times for detecting boundaries.
        prediction (pd.Series): Series containing the prediction results with timestamps.
        point (int, optional): Index of the predicted change point within the window. Defaults to 0.
        binary (bool, optional): Flag indicating whether to use binary classification. Defaults to False.

    Returns:
        dict: A dictionary containing:
            - 'TPs' (dict): Dictionary of true positives with window indices as keys and lists of [start, predicted, end] times as values.
            - 'FPs' (list): List of false positive timestamps.
            - 'FNs' (list): List of false negative window indices or timestamps.
    """
    _detecting_boundaries = []
    for couple in detecting_boundaries.copy():
        if len(couple) != 0:
            _detecting_boundaries.append(couple)
    detecting_boundaries = _detecting_boundaries

    times_pred = prediction[prediction.dropna() == 1].sort_index().index

    my_dict: dict[str, Any] = {}
    my_dict["TPs"] = {}
    my_dict["FPs"] = []
    my_dict["FNs"] = []

    if len(detecting_boundaries) != 0:
        my_dict["FPs"].append(
            times_pred[times_pred < detecting_boundaries[0][0]]
        )  # left
        for i in range(len(detecting_boundaries)):
            times_pred_window = times_pred[
                (times_pred >= detecting_boundaries[i][0])
                & (times_pred <= detecting_boundaries[i][1])
            ]
            times_prediction_in_window = prediction[
                detecting_boundaries[i][0] : detecting_boundaries[i][1]
            ].index
            if len(times_pred_window) == 0:
                if not binary:
                    my_dict["FNs"].append(i)
                else:
                    my_dict["FNs"].append(times_prediction_in_window)
            else:
                my_dict["TPs"][i] = [
                    detecting_boundaries[i][0],
                    times_pred_window[point]
                    if not binary
                    else times_pred_window,  # attention
                    detecting_boundaries[i][1],
                ]
                if binary:
                    my_dict["FNs"].append(
                        times_prediction_in_window[
                            ~times_prediction_in_window.isin(times_pred_window)
                        ]
                    )
            if len(detecting_boundaries) > i + 1:
                my_dict["FPs"].append(
                    times_pred[
                        (times_pred > detecting_boundaries[i][1])
                        & (times_pred < detecting_boundaries[i + 1][0])
                    ]
                )

        my_dict["FPs"].append(
            times_pred[times_pred > detecting_boundaries[i][1]]
        )  # right
    else:
        my_dict["FPs"].append(times_pred)

    if len(my_dict["FPs"]) > 1:
        my_dict["FPs"] = np.concatenate(my_dict["FPs"])
    elif len(my_dict["FPs"]) == 1:
        my_dict["FPs"] = my_dict["FPs"][0]
    if len(my_dict["FPs"]) == 0:  # not elif on purpose
        my_dict["FPs"] = []

    if binary:
        if len(my_dict["FNs"]) > 1:
            my_dict["FNs"] = np.concatenate(my_dict["FNs"])
        elif len(my_dict["FNs"]) == 1:
            my_dict["FNs"] = my_dict["FNs"][0]
        if len(my_dict["FNs"]) == 0:  # not elif on purpose
            my_dict["FNs"] = []
    return my_dict


def confusion_matrix(true, prediction):
    true_ = true == 1
    prediction_ = prediction == 1
    TP = (true_ & prediction_).sum()
    TN = (~true_ & ~prediction_).sum()
    FP = (~true_ & prediction_).sum()
    FN = (true_ & ~prediction_).sum()
    return TP, TN, FP, FN


def single_average_delay(
    detecting_boundaries,
    prediction,
    anomaly_window_destination,
    clear_anomalies_mode,
):
    """
    anomaly_window_destination: 'lefter', 'righter', 'center'. Default='right'
    """
    detecting_boundaries = filter_detecting_boundaries(detecting_boundaries)
    point = 0 if clear_anomalies_mode else -1
    dict_cp_confusion = extract_cp_confusion_matrix(
        detecting_boundaries, prediction, point=point
    )

    missing = 0
    detectHistory = []
    all_true_anom = 0
    FP = 0

    FP += len(dict_cp_confusion["FPs"])
    missing += len(dict_cp_confusion["FNs"])
    all_true_anom += len(dict_cp_confusion["TPs"]) + len(
        dict_cp_confusion["FNs"]
    )

    if anomaly_window_destination == "lefter":

        def average_time(output_cp_cm_tp):
            return output_cp_cm_tp[2] - output_cp_cm_tp[1]
    elif anomaly_window_destination == "righter":

        def average_time(output_cp_cm_tp):
            return output_cp_cm_tp[1] - output_cp_cm_tp[0]
    elif anomaly_window_destination == "center":

        def average_time(output_cp_cm_tp):
            return output_cp_cm_tp[1] - (
                output_cp_cm_tp[0]
                + (output_cp_cm_tp[2] - output_cp_cm_tp[0]) / 2
            )
    else:
        raise Exception("Choose anomaly_window_destination")

    for fp_case_window in dict_cp_confusion["TPs"]:
        detectHistory.append(
            average_time(dict_cp_confusion["TPs"][fp_case_window])
        )
    return missing, detectHistory, FP, all_true_anom


def my_scale(
    fp_case_window=None,
    A_tp=1,
    A_fp=0,
    koef=1,
    detalization=1000,
    clear_anomalies_mode=True,
    plot_figure=False,
):
    """
    ts - segment on which the window is applied
    """
    x = np.linspace(-np.pi / 2, np.pi / 2, detalization)
    x = x if clear_anomalies_mode else x[::-1]
    y = (
        (A_tp - A_fp)
        / 2
        * -1
        * np.tanh(koef * x)
        / (np.tanh(np.pi * koef / 2))
        + (A_tp - A_fp) / 2
        + A_fp
    )
    if not plot_figure and fp_case_window is not None:
        event = int(
            (fp_case_window[1] - fp_case_window[0])
            / (fp_case_window[-1] - fp_case_window[0])
            * detalization
        )
        if event >= len(x):
            event = len(x) - 1
        score = y[event]
        return score
    else:
        return y


def single_evaluate_nab(
    detecting_boundaries,
    prediction,
    table_of_coef=None,
    clear_anomalies_mode=True,
    scale_func="improved",
    scale_koef=1,
) -> np.ndarray:
    """
    Evaluate the NAB (Numenta Anomaly Benchmark) score for a given set of predictions.

    Args:
        detecting_boundaries (list of list of two float values):
            The list of lists of left and right boundary indices for scoring results of labeling.
            If empty, can be [[]], or [[],[t1,t2],[]].
        prediction (list):
            The list of predicted anomaly points.
        table_of_coef (pandas DataFrame, optional):
            Table of coefficients for NAB score function.
            Default is a 3x4 DataFrame with indices 'Standard', 'LowFP', 'LowFN' and columns 'A_tp', 'A_fp', 'A_tn', 'A_fn'.
        clear_anomalies_mode (bool, optional):
            If True, the left of Atp boundary is equal to the right of Afp.
            Otherwise, fault mode, when the left of Afp boundary is the right of Atp. Default is True.
        scale_func (str, optional):
            The scaling function to use. Default is "improved".
            If not "improved", an exception is raised.
        scale_koef (int, optional):
            The scaling coefficient. Default is 1.
            1 - depends on the relative step, which means that if there are too many points in the scoring window, the difference will be too large.
            too many points in the scoring window, the drop will be too
            stiff in the middle.
            2- the leftmost point is not equal to Atp and the right is not equal to Afp.
            (especially if you use a blurring multiplier).

    Returns:
        numpy.ndarray:
            A 3xN array where N is the number of profiles ('Standard', 'LowFP', 'LowFN').
            The first row contains the scores, the second row contains the null scores,
            and the third row contains the perfect scores.
    """
    if scale_func == "improved":
        scale_func = my_scale
    else:
        raise Exception("choose the scale_func")

    # filter
    detecting_boundaries = filter_detecting_boundaries(detecting_boundaries)

    if table_of_coef is None:
        table_of_coef = pd.DataFrame(
            [
                [1.0, -0.11, 1.0, -1.0],
                [1.0, -0.22, 1.0, -1.0],
                [1.0, -0.11, 1.0, -2.0],
            ]
        )
        table_of_coef.index = pd.Index(["Standard", "LowFP", "LowFN"])
        table_of_coef.index.name = "Metric"
        table_of_coef.columns = ["A_tp", "A_fp", "A_tn", "A_fn"]

    # GO
    point = 0 if clear_anomalies_mode else -1
    dict_cp_confusion = extract_cp_confusion_matrix(
        detecting_boundaries, prediction, point=point
    )

    Scores, Scores_perfect, Scores_null = [], [], []
    for profile in ["Standard", "LowFP", "LowFN"]:
        A_tp = table_of_coef["A_tp"][profile]
        A_fp = table_of_coef["A_fp"][profile]
        A_fn = table_of_coef["A_fn"][profile]

        score = 0
        score += A_fp * len(dict_cp_confusion["FPs"])
        score += A_fn * len(dict_cp_confusion["FNs"])
        for fp_case_window in dict_cp_confusion["TPs"]:
            set_times = dict_cp_confusion["TPs"][fp_case_window]
            score += scale_func(set_times, A_tp, A_fp, koef=scale_koef)

        Scores.append(score)
        Scores_perfect.append(len(detecting_boundaries) * A_tp)
        Scores_null.append(len(detecting_boundaries) * A_fn)

    return np.array(
        [np.array(Scores), np.array(Scores_null), np.array(Scores_perfect)]
    )


def chp_score(
    true,
    prediction,
    metric="nab",
    window_width=None,
    portion=0.1,
    anomaly_window_destination="lefter",
    clear_anomalies_mode=True,
    intersection_mode="cut right window",
    table_of_coef=None,
    scale_func="improved",
    scale_koef=1,
    verbose=False,
):
    """
    Calculates various metrics for evaluating anomaly or changepoint detection.

    Args:
        true (pd.Series or list): True labels of anomalies or changepoints.
            Can be in various formats:
            - pd.Series with binary int labels (1 is anomaly, 0 is not anomaly)
            - list of pd.Timestamp of true labels
            - list of list of t1, t2: left and right detection boundaries of pd.Timestamp
            - list of pd.Series with binary int labels for multiple datasets
            - list of list of pd.Timestamp of true labels for multiple datasets
            - list of list of list of t1, t2: left and right detection boundaries of pd.Timestamp for multiple datasets
        prediction (pd.Series or list): Predicted labels of anomalies or changepoints.
            Can be in various formats:
            - pd.Series with binary int labels (1 is anomaly, 0 is not anomaly)
            - list of pd.Series with binary int labels for multiple datasets
        metric (str): Metric to use for evaluation. Options are {'nab', 'binary', 'average_time', 'confusion_matrix'}. Default is 'nab'.
        window_width (str): Width of detection window as a pd.Timedelta string. Default is None.
        portion (float): Portion of the width of the length of prediction divided by the number of real CPs in the dataset. Default is 0.1.
        anomaly_window_destination (str): Location of the detection window relative to the anomaly. Options are {'lefter', 'righter', 'center'}. Default is 'lefter'.
        clear_anomalies_mode (bool): If True, only the first value inside the detection window is taken. If False, only the last value inside the detection window is taken. Default is True.
        intersection_mode (str): How to handle overlapping detection windows. Options are {'cut left window', 'cut right window', 'both'}. Default is 'cut right window'.
        table_of_coef (pd.DataFrame): Application profiles of NAB metric. Default is None.
        scale_func (str): Scoring function in NAB metric. Options are {'default', 'improved'}. Default is 'improved'.
        scale_koef (float): Smoothing factor for the scoring function. Default is 1.
        verbose (bool): If True, outputs useful information. Default is False.

    Returns:
        tuple or dict: Value of the metrics depending on the chosen metric.
            - 'nab': dict with keys 'Standard', 'LowFP', 'LowFN' and corresponding float values.
            - 'average_time': tuple with average time (float), missing changepoints (int), false positives (int), number of true changepoints (int).
            - 'binary': tuple with F1 metric (float), false alarm rate (float), missing alarm rate (float).
            - 'confusion_matrix': tuple with true positives (int), true negatives (int), false positives (int), false negatives (int).

    Examples:
        >>> y_true = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

        >>> y_pre1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        >>> y_pre2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        >>> y_pre3 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        >>> y_pre4 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        >>> y_pre5 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


        >>> def convert_comp(y):
        ...     y_ = pd.Series(y)
        ...     y_.index = pd.to_datetime(y_.index, unit="s")
        ...     return y_


        >>> y_true_ = convert_comp(y_true)
        >>> for y_pred in [y_pre1, y_pre2, y_pre3, y_pre4, y_pre5]:
        ...     print(f"{y_true}{y_pred}")
        ...     y_pred = convert_comp(y_pred)
        ...     print("=== Binary ===")
        ...     print(chp_score(
        ...         y_true_,
        ...         y_pred,
        ...         metric="binary",
        ...     ))
        ...     print("=== Average time ===")
        ...     print(chp_score(
        ...         y_true_,
        ...         y_pred,
        ...         metric="average_time",
        ...         window_width="3s",
        ...         anomaly_window_destination="righter",
        ...         portion=1,
        ...     ))
        ...     print("=== NAB ===")
        ...     print(chp_score(
        ...         y_true_,
        ...         y_pred,
        ...         metric="nab",
        ...         window_width="3s",
        ...         anomaly_window_destination="righter",
        ...         portion=1,
        ...     ))
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0][0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        === Binary ===
        (0.0, 0.0, 100.0)
        === Average time ===
        (nan, 1, 0, 1)
        === NAB ===
        {'Standard': 0.0, 'LowFP': 0.0, 'LowFN': 0.0}
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0][0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        === Binary ===
        (0.0, 16.67, 100.0)
        === Average time ===
        (Timedelta('0 days 00:00:03'), 0, 0, 1)
        === NAB ===
        {'Standard': 44.5, 'LowFP': 39.0, 'LowFN': 63.0}
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0][1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        === Binary ===
        (0.0, 16.67, 100.0)
        === Average time ===
        (nan, 1, 1, 1)
        === NAB ===
        {'Standard': -5.5, 'LowFP': -11.0, 'LowFN': -3.67}
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0][1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        === Binary ===
        (0.25, 100.0, 0.0)
        === Average time ===
        (Timedelta('0 days 00:00:00'), 0, 3, 1)
        === NAB ===
        {'Standard': 83.5, 'LowFP': 67.0, 'LowFN': 89.0}
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0][0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        === Binary ===
        (1.0, 0.0, 0.0)
        === Average time ===
        (Timedelta('0 days 00:00:00'), 0, 0, 1)
        === NAB ===
        {'Standard': 100.0, 'LowFP': 100.0, 'LowFN': 100.0}
    """

    assert isinstance(true, pd.Series) or isinstance(true, list)
    # checking prediction
    if isinstance(prediction, pd.Series):
        true = [true]
        prediction = [prediction]
    elif isinstance(prediction, list):
        if not all(isinstance(my_el, pd.Series) for my_el in prediction):
            raise Exception("Incorrect format for prediction")
    else:
        raise Exception("Incorrect format for prediction")

    # checking dataset length: Number of dataset unequal
    assert len(true) == len(prediction)

    # final check
    input_variant = check_errors(true)

    def check_sort(my_list, input_variant):
        for dataset in my_list:
            if input_variant == 2:
                assert all(np.sort(dataset) == np.array(dataset))
            elif input_variant == 3:
                assert all(
                    np.sort(np.concatenate(dataset)) == np.concatenate(dataset)
                )
            elif input_variant == 1:
                assert all(
                    dataset.index.values == dataset.sort_index().index.values
                )

    check_sort(true, input_variant)
    check_sort(prediction, 1)

    # part 2. To detected boundaries
    if (
        ((metric == "nab") or (metric == "average_time"))
        and (window_width is None)
        and (input_variant != 3)
    ):
        print(
            f"Since you didn't choose window_width and portion, portion will be default ({portion})"
        )

    if input_variant == 1:
        detecting_boundaries = [
            single_detecting_boundaries(
                true_series=true[i],
                true_list_ts=None,
                prediction=prediction[i],
                window_width=window_width,
                portion=portion,
                anomaly_window_destination=anomaly_window_destination,
                intersection_mode=intersection_mode,
            )
            for i in range(len(true))
        ]

    elif input_variant == 2:
        detecting_boundaries = [
            single_detecting_boundaries(
                true_series=None,
                true_list_ts=true[i],
                prediction=prediction[i],
                window_width=window_width,
                portion=portion,
                anomaly_window_destination=anomaly_window_destination,
                intersection_mode=intersection_mode,
            )
            for i in range(len(true))
        ]

    elif input_variant == 3:
        detecting_boundaries = true.copy()
        # Next anti fool system [[[t1,t2]],[]] -> [[[t1,t2]],[[]]]
        for i in range(len(detecting_boundaries)):
            if len(detecting_boundaries[i]) == 0:
                detecting_boundaries[i] = [[]]
    else:
        raise Exception("Unknown format for true data")

    if metric == "nab":
        matrix = np.zeros((3, 3))
        for i in range(len(prediction)):
            matrix_ = single_evaluate_nab(
                detecting_boundaries[i],
                prediction[i],
                table_of_coef=table_of_coef,
                clear_anomalies_mode=clear_anomalies_mode,
                scale_func=scale_func,
                scale_koef=scale_koef,
                # plot_figure=plot_figure,
            )
            matrix = matrix + matrix_

        results = {}
        desc = ["Standard", "LowFP", "LowFN"]
        for t, profile_name in enumerate(desc):
            results[profile_name] = round(
                100
                * (matrix[0, t] - matrix[1, t])
                / (matrix[2, t] - matrix[1, t]),
                2,
            )
            if verbose:
                print(profile_name, " - ", results[profile_name])
        return results

    elif metric == "average_time":
        missing, FP, all_true_anom = 0, 0, 0
        detectHistory: list[Any] = []
        for i in range(len(prediction)):
            missing_, detectHistory_, FP_, all_true_anom_ = (
                single_average_delay(
                    detecting_boundaries[i],
                    prediction[i],
                    anomaly_window_destination=anomaly_window_destination,
                    clear_anomalies_mode=clear_anomalies_mode,
                )
            )
            missing, detectHistory, FP, all_true_anom = (
                missing + missing_,
                detectHistory + detectHistory_,
                FP + FP_,
                all_true_anom + all_true_anom_,
            )
        add = np.mean(detectHistory)
        if verbose:
            print("Amount of true anomalies", all_true_anom)
            print(f"A number of missed CPs = {missing}")
            print(f"A number of FPs = {int(FP)}")
            print("Average time", add)
        return add, missing, int(FP), all_true_anom

    elif (metric == "binary") or (metric == "confusion_matrix"):
        if all(isinstance(my_el, pd.Series) for my_el in true):
            TP, TN, FP, FN = 0, 0, 0, 0
            for i in range(len(prediction)):
                TP_, TN_, FP_, FN_ = confusion_matrix(true[i], prediction[i])
                TP, TN, FP, FN = TP + TP_, TN + TN_, FP + FP_, FN + FN_
        else:
            print(
                "For this metric it is better if you use pd.Series format for true \nwith common index of true and prediction"
            )
            TP, TN, FP, FN = 0, 0, 0, 0
            for i in range(len(prediction)):
                dict_cp_confusion = extract_cp_confusion_matrix(
                    detecting_boundaries[i], prediction[i], binary=True
                )
                TP += np.sum(
                    [
                        len(dict_cp_confusion["TPs"][window][1])
                        for window in dict_cp_confusion["TPs"]
                    ]
                )
                FP += len(dict_cp_confusion["FPs"])
                FN += len(dict_cp_confusion["FNs"])
                TN += len(prediction[i]) - TP - FP - FN

        if metric == "binary":
            f1 = round(TP / (TP + (FN + FP) / 2), 2)
            far = round(FP / (FP + TN) * 100, 2)
            mar = round(FN / (FN + TP) * 100, 2)
            if verbose:
                print(f"False Alarm Rate {far} %")
                print(f"Missing Alarm Rate {mar} %")
                print(f"F1 metric {f1}")
            return f1, far, mar

        elif metric == "confusion_matrix":
            if verbose:
                print("TP", TP)
                print("TN", TN)
                print("FP", FP)
                print("FN", FN)
            return TP, TN, FP, FN
    else:
        raise Exception("Choose the performance metric")
