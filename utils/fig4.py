import os

import pandas as pd

from utils import plot_corr, plot_corr_double

if __name__ == "__main__":
    plot_dir = f"../plot/overall"
    os.makedirs(plot_dir, exist_ok=True)

    nonamor4plugin_gof_train = pd.read_csv(
        f"../plot/nonamor_calibration/nonamor4plugin_gof_train.csv"
    )["gof_means"].values
    nonamor4plugin_gof_test = pd.read_csv(
        f"../plot/nonamor_calibration/nonamor4plugin_gof_test.csv"
    )["gof_means"].values
    nonamor4amor_gof_train = pd.read_csv(
        f"../plot/nonamor_calibration/nonamor4amor_gof_train.csv"
    )["gof_means"].values
    nonamor4amor_gof_test = pd.read_csv(
        f"../plot/nonamor_calibration/nonamor4amor_gof_test.csv"
    )["gof_means"].values
    amor_gof_train = pd.read_csv(f"../plot/amor_calibration/amor_single_gof_train.csv")[
        "gof_means"
    ].values
    amor_gof_test = pd.read_csv(f"../plot/amor_calibration/amor_single_gof_test.csv")[
        "gof_means"
    ].values
    plugin_gof_train = pd.read_csv(
        f"../plot/plugin_regression/plugin_single_gof_train.csv"
    )["gof_means"].values
    plugin_gof_test = pd.read_csv(
        f"../plot/plugin_regression/plugin_single_gof_test.csv"
    )["gof_means"].values

    nonamor_ctt = pd.read_csv(
        f"../plot/nonamor_calibration/nonamor_calibration_corr_ctt.csv"
    )["corr_ctt_means"].values
    amor_ctt = pd.read_csv(f"../plot/amor_calibration/amor_calibration_corr_ctt.csv")[
        "corr_ctt_means"
    ].values

    nonamor_helm = pd.read_csv(
        f"../plot/nonamor_calibration/nonamor_calibration_corr_helm.csv"
    )["corr_helm_means"].values
    amor_helm = pd.read_csv(f"../plot/amor_calibration/amor_calibration_corr_helm.csv")[
        "corr_helm_means"
    ].values

    plot_corr(
        data1=nonamor_ctt,
        data2=amor_ctt,
        plot_path=f"{plot_dir}/ctt_nonamor_amor.png",
        title=r"$\theta$ correlation with CTT",
        # title=r'$\theta$ correlation with CTT. $\rho$ = {:.2f}',
        xlabel=r"Traditional amortization",
        ylabel=r"Joint amortization",
    )

    plot_corr(
        data1=nonamor_helm,
        data2=amor_helm,
        plot_path=f"{plot_dir}/helm_nonamor_amor.png",
        title=r"$\theta$ correlation with HELM",
        # title=r'$\theta$ correlation with HELM. $\rho$ = {:.2f}',
        xlabel=r"Traditional amortization",
        ylabel=r"Joint amortization",
    )

    plot_corr_double(
        data1_train=nonamor4amor_gof_train,
        data1_test=nonamor4amor_gof_test,
        data2_train=amor_gof_train,
        data2_test=amor_gof_test,
        plot_path=f"{plot_dir}/gof_nonamor_amor.png",
        xlabel=r"Traditional amortization",
        ylabel=r"Joint amortization",
    )

    plot_corr_double(
        data1_train=nonamor4plugin_gof_train,
        data1_test=nonamor4plugin_gof_test,
        data2_train=plugin_gof_train,
        data2_test=plugin_gof_test,
        plot_path=f"{plot_dir}/gof_nonamor_plungin.png",
        xlabel=r"Traditional amortization",
        ylabel=r"Plug-in amortization",
    )
