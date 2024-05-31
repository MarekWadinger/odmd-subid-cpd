import datetime
import os
import sys
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from river.compose import Pipeline
from river.decomposition import OnlineDMD, OnlineDMDwC
from river.preprocessing import Hankelizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from functions.chdsubid import SubIDChangeDetector, get_default_rank
from functions.metrics import chp_score
from functions.plot import plot_chd
from functions.preprocessing import hankel
from functions.rolling import Rolling


# --- Functions ---
def update_selection_X():
    st.session_state.selected_X = st.session_state.multiselect_X
    st.session_state.m = len(st.session_state.selected_X)
    st.session_state.m
    if st.session_state.m > 0:
        st.session_state.disable_params = False
    else:
        st.session_state.disable_params = True


def update_selection_U():
    st.session_state.selected_U = st.session_state.multiselect_U
    st.session_state.l = len(st.session_state.selected_U)


def progressive_val_predict(
    X: pd.DataFrame,
    U: pd.DataFrame,
    _model: SubIDChangeDetector | Pipeline,
    compute_alt_scores=False,
    _progress_bar=None,
):
    # CREATE REFERENCE TO LAST STEP OF PIPELINE (TRACK STATE OF MODEL)
    if isinstance(_model, Pipeline):
        model_ = _model._last_step
    else:
        model_ = _model

    y_pred = np.zeros(X.shape[0], dtype=float)
    meta: dict[str, np.ndarray] = {}
    if compute_alt_scores:
        meta["scores_dmd_diff"] = np.zeros(X.shape[0], dtype=float)

    if U.empty:
        U = pd.DataFrame(index=X.index, columns=[None])
    # Run pipeline
    for i, (x, u) in enumerate(
        zip(
            X.to_dict(orient="records"),
            U.to_dict(orient="records"),
        )
    ):
        y_pred[i] = _model.score_one(x)
        if compute_alt_scores:
            meta["scores_dmd_diff"][i] = (
                model_.distances[1] - model_.distances[0]
            ).real
        if None in u:
            _model.learn_one(x)
        else:
            _model.learn_one(x, **{"u": u})

        if _progress_bar:
            _progress_bar.progress(
                i / st.session_state.n, text="Running experiment ..."
            )
    return y_pred, meta


def export_fig(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="pdf")
    buf.seek(0)
    return buf.getvalue()


@st.cache_data
def concat_results(X, scores_dmd, scores_dmd_diff):
    df = pd.concat(
        [
            X,
            pd.Series(scores_dmd.real, index=X.index, name="DMD"),
            pd.Series(scores_dmd_diff.real, index=X.index, name="DMD (diff)"),
        ],
        axis=1,
    )
    return df


@st.cache_data
def plot(X, scores_dmd, scores_dmd_diff, Y_, test_size):
    fig, axs = plot_chd(
        [X.values, scores_dmd.real, scores_dmd_diff.real],
        Y_,
        labels=["X", "DMD", "DMD (diff)"],
        grace_period=test_size,
    )
    fig.set_size_inches(18, 10)  # Set the size of the figure
    return fig, axs


@st.cache_data
def compute_metrics(Y, scores_dmd, test_size):
    start_date = "2023-01-01 00:00:00"
    date_range = pd.date_range(start=start_date, periods=len(Y), freq="s")
    y_true = pd.Series(Y, index=date_range)
    y_true_cp = y_true.diff().abs().fillna(0.0)
    experiments: dict[str, pd.Series] = {
        "Perfect detector": y_true,
        "Random detector": pd.Series(
            np.random.randint(2, size=y_true.shape[0]), index=date_range
        ),
        "Null detector": pd.Series(
            np.zeros(y_true.shape[0]), index=date_range
        ),
        "Always positive": pd.Series(
            np.ones(y_true.shape[0]), index=date_range
        ),
        "Online DMD (t=0)": pd.Series(scores_dmd > 0.0, index=date_range),
        "Online DMD (t=0.5)": pd.Series(scores_dmd > 0.5, index=date_range),
        "Online DMD": pd.Series(scores_dmd > 0.25, index=date_range),
    }

    # TODO: Seems like the window width is not aligned correctly with index
    window_params = {
        "valid": {
            "window_width": f"{test_size}s",
            "anomaly_window_destination": "righter",
        },
    }

    metrics = [
        "F1",
        "FAR",
        "MAR",
        "Delay",
        "TP",
        "FN",
        "FP",
        "Standard",
        "LowFP",
        "LowFN",
    ]

    df_res = pd.DataFrame(
        columns=[m for m in metrics],
        index=list(experiments.keys()),
    )

    for name, po in experiments.items():
        pc = po.astype(int).diff().abs().fillna(0.0)
        res = {}
        for window_name, kwargs in window_params.items():
            binary = chp_score(
                y_true,
                po,
                metric="binary",
            )
            add = chp_score(
                y_true_cp,
                pc,
                metric="average_time",
                window_width=kwargs["window_width"],
                anomaly_window_destination=kwargs[
                    "anomaly_window_destination"
                ],
            )
            nab = chp_score(
                y_true_cp,
                pc,
                metric="nab",
                window_width=kwargs["window_width"],
                anomaly_window_destination=kwargs[
                    "anomaly_window_destination"
                ],
            )
            res_ = dict(zip(["F1", "FAR", "MAR"], binary))
            res_.update(dict(zip(["Delay", "FN", "FP", "TP"], add)))
            res_.update(nab)
            res_ = {k: v for k, v in res_.items()}
            res.update(res_)
        df_res[name] = res
    df_res.sort_values("Standard", ascending=False)

    return df_res


@st.cache_data
def export_df(df):
    return df.to_csv().encode("utf-8")


# --- Page Config ---

st.set_page_config(
    page_title="Streamlit cheat sheet",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Initialize session ---
if "n" not in st.session_state:
    st.session_state.n = 100

if "m" not in st.session_state:
    st.session_state.m = 1

if "l" not in st.session_state:
    st.session_state.l = 0

if "selected_X" not in st.session_state:
    st.session_state.selected_X = []

if "selected_U" not in st.session_state:
    st.session_state.selected_U = []

if "disable_params" not in st.session_state:
    st.session_state.disable_params = True

# === Sidebar ===
st.sidebar.title("Upload Data")

data = st.sidebar.file_uploader(
    "Upload a CSV file with snapshots", type=["csv"]
)
data_gt = st.sidebar.file_uploader(
    "Upload a CSV file with ground truth (Optional)", type=["csv"]
)

st.sidebar.title("Partition Data")
if data:
    df = pd.read_csv(data, index_col=0)
    df.index = pd.to_datetime(df.index)

    # Get the list of all columns
    all_columns = df.columns.tolist()

    # Create multiselect widgets
    st.sidebar.multiselect(
        "Select columns for X",
        [col for col in all_columns if col not in st.session_state.selected_U],
        key="multiselect_X",
        default=st.session_state.selected_X,
        on_change=update_selection_X,
    )
    st.sidebar.multiselect(
        "Select columns for U",
        [col for col in all_columns if col not in st.session_state.selected_X],
        key="multiselect_U",
        default=st.session_state.selected_U,
        on_change=update_selection_U,
    )

    st.session_state.n, st.session_state.m, st.session_state.l = (
        len(df),
        len(st.session_state.selected_X),
        len(st.session_state.selected_U),
    )

    if data_gt:
        df_gt = pd.read_csv(data_gt, index_col=0)
        if len(df_gt) != len(df):
            st.error(
                "The number of snapshots in the ground truth file is different from the number of snapshots in the data file."
            )
        df_gt.index = pd.to_datetime(df_gt.index)

st.sidebar.title("Parameters")

with st.sidebar.form(key="params_form", border=False):
    window_size: int = st.slider(
        "Learning windows size (Default: 1/10 snapshots)",
        1,
        st.session_state.n // 10,
        disabled=st.session_state.disable_params,
    )
    ref_size: int = st.slider(
        "Base windows size (Default: window_size)",
        1,
        st.session_state.n // 10,
        key="slider_ref_size",
        disabled=st.session_state.disable_params,
    )
    test_size = st.slider(
        "Test windows size (Default: window_size)",
        1,
        st.session_state.n // 10,
        key="slider_test_size",
        disabled=st.session_state.disable_params,
    )
    lag = st.slider(
        "Lag (Default: 0)",
        -st.session_state.n // 10,
        st.session_state.n // 10,
        0,
        disabled=st.session_state.disable_params,
    )
    if ref_size + lag + test_size > st.session_state.n:
        st.error(
            "The sum of the base windows size, lag, and test windows size should be less than the number of snapshots."
        )
    hm = st.slider(
        "Time-delays states (Default: 0)",
        0,
        60 // st.session_state.m if st.session_state.l > 0 else 60,
        disabled=st.session_state.disable_params,
    )

    hl = st.slider(
        "Time-delays control (Default: 0)",
        0,
        60 // st.session_state.l if st.session_state.l > 0 else 60,
        label_visibility="hidden" if st.session_state.l < 0 else "visible",
    )
    submit_params = st.form_submit_button("Run")

# === Main ===
st.title(
    "Change-Point Detection in Industrial Data Streams based on Online DMD with Control"
)

# === Enable after submitting parameters ===
if submit_params:
    if ref_size + lag + test_size > st.session_state.n:
        st.error(
            "The sum of the base windows size, lag, and test windows size should be less than the number of snapshots."
        )
    p = min(
        10,
        get_default_rank(
            hankel(
                df[:window_size][st.session_state.selected_X],
                hm,
                hm // 60 // st.session_state.m,
            )
        ),
    )

    q = min(
        10,
        get_default_rank(
            hankel(
                df[:window_size][st.session_state.selected_X],
                hm,
                hm // 60 // st.session_state.m,
            )
        ),
    )

    # --- Run the algorithm ---
    runtime_info = st.info("Preparing run")
    X = df[st.session_state.selected_X]
    U = df[st.session_state.selected_U]
    # TODO: enable hankelization of us on the fly
    U_ = pd.DataFrame(hankel(U, hn=hl))

    # Initialize Hankelizer
    hankelizer = Hankelizer(hm)

    # Initialize Transformer
    init_size = window_size
    if U.empty:
        odmd = Rolling(
            OnlineDMD(
                r=p,
                initialize=init_size,
                w=1.0,
                exponential_weighting=False,
                eig_rtol=1e-1,
            ),
            init_size + 1,
        )
    else:
        odmd = Rolling(
            OnlineDMDwC(
                p=p,
                q=q,
                initialize=init_size,
                w=1.0,
                exponential_weighting=False,
                eig_rtol=1e-1,
            ),
            init_size + 1,
        )

    # Initialize Change-Point Detector
    subid_dmd = SubIDChangeDetector(
        odmd,
        ref_size=ref_size,
        test_size=test_size,
        grace_period=init_size + test_size + 1,
    )

    # Build pipeline
    pipeline_dmd = hankelizer | subid_dmd

    # Run pipeline
    runtime_info.info(f"Detecting with {odmd} ...")
    runtime_prog = st.progress(0, text="Running experiment ...")
    scores_dmd, meta = progressive_val_predict(
        X,
        U_,
        pipeline_dmd,
        compute_alt_scores=True,
        _progress_bar=runtime_prog,
    )
    runtime_prog.empty()
    scores_dmd_diff = meta["scores_dmd_diff"]

    runtime_info.info("Plotting results ...")
    # Plot results
    if data_gt:
        Y: np.ndarray | None = df_gt.iloc[:, 0].values
        Y_: np.ndarray | None = np.where(Y == 1)[0]
    else:
        Y = None
        Y_ = None
    fig, axs = plot(X, scores_dmd, scores_dmd_diff, Y_, test_size)

    now = datetime.datetime.now().strftime("%Y%m%d-%H_%M_%S")
    tab1, tab2, tab3 = st.tabs(
        ["📈 **Chart**", "🗃 **Data**", "🏆 **Metrics**"]
    )
    tab1.write(fig)
    buf = export_fig(fig)
    tab1.download_button(
        "Download figure as PDF",
        buf,
        file_name=f"{now}-odmd-cpd-results.pdf",
        mime="application/pdf",
    )

    df_res = concat_results(X, scores_dmd, scores_dmd_diff)
    tab2.write(df_res)
    tab2.download_button(
        "Download results as CSV",
        export_df(df_res),
        file_name=f"{now}-odmd-cpd-results.csv",
        mime="text/csv",
    )

    if Y is not None:
        df_metrics = compute_metrics(Y, scores_dmd, test_size)
        tab3.write(df_metrics)
        tab3.download_button(
            "Download metrics as CSV",
            export_df(df_metrics),
            file_name=f"{now}-odmd-cpd-results.csv",
            mime="text/csv",
        )
    runtime_info.success("Done! 🎉")
