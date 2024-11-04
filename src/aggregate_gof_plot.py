import pandas as pd
from utils import (
    DATASETS,
)

if __name__ == "__main__":
    traditional_gof = pd.read_csv(f'../plot/nonamor_calibration/nonamor_calibration_gof.csv').values
    
    plugin_gof_df_train = pd.DataFrame({
        'datasets': DATASETS,
        'gof_means': plugin_gof_train_means,
    })
    plugin_gof_df_train.to_csv(f'{plot_dir}/nonamor4plugin_gof_train.csv', index=False)
    
    plugin_gof_df_test = pd.DataFrame({
        'datasets': DATASETS,
        'gof_means': plugin_gof_test_means,
    })
    plugin_gof_df_test.to_csv(f'{plot_dir}/nonamor4plugin_gof_test.csv', index=False)
    
    amor_gof_df_train = pd.DataFrame({
        'datasets': DATASETS,
        'gof_means': amor_gof_train_means,
    })
    amor_gof_df_train.to_csv(f'{plot_dir}/nonamor4amor_gof_train.csv', index=False)
    
    amor_gof_df_test = pd.DataFrame({
        'datasets': DATASETS,
        'gof_means': amor_gof_test_means,
    })
    amor_gof_df_test.to_csv(f'{plot_dir}/nonamor4amor_gof_test.csv', index=False)
    
    gof_df = pd.DataFrame({
        'datasets': DATASETS,
        'gof_means': gof_means,
        'gof_stds': gof_stds
    })
    gof_df.to_csv(f'{plot_dir}/nonamor_calibration_gof.csv', index=False)
    