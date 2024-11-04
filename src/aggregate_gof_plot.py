import pandas as pd

if __name__ == "__main__":
    trad_gof = pd.read_csv(f'../plot/nonamor_calibration/nonamor_calibration_gof.csv')['gof_means'].values
    plugin_gof_train = pd.read_csv(f'../plot/plugin_regression/plugin_regression_gof_train.csv')['gof_means'].values
    plugin_gof_test = pd.read_csv(f'../plot/plugin_regression/plugin_regression_gof_test.csv')['gof_means'].values
    joint_gof_train = pd.read_csv(f'../plot/amor_calibration/amor_calibration_gof_train.csv')['gof_means'].values
    joint_gof_test = pd.read_csv(f'../plot/amor_calibration/amor_calibration_gof_test.csv')['gof_means'].values
    dim2_1pl_trad_gof = pd.read_csv(f'../plot/dim2_1pl/dim2_1pl_gof_con_True.csv')['gof_means'].values