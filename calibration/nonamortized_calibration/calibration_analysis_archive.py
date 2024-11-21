#     plugin_train_indices = pd.read_csv(
#         f"../../data/plugin_regression/{dataset}/train_0.csv"
#     )["index"].values
#     plugin_test_indices = pd.read_csv(
#         f"../../data/plugin_regression/{dataset}/test_0.csv"
#     )["index"].values

#     plugin_gof_train_mean, _ = goodness_of_fit(
#         z=torch.tensor(item_parms[plugin_train_indices], dtype=torch.float32),
#         theta=torch.tensor(abilities, dtype=torch.float32),
#         y=torch.tensor(y[:, plugin_train_indices], dtype=torch.float32),
#     )
#     plugin_gof_train_means.append(plugin_gof_train_mean)

#     plugin_gof_test_mean, _ = goodness_of_fit(
#         z=torch.tensor(item_parms[plugin_test_indices], dtype=torch.float32),
#         theta=torch.tensor(abilities, dtype=torch.float32),
#         y=torch.tensor(y[:, plugin_test_indices], dtype=torch.float32),
#     )
#     plugin_gof_test_means.append(plugin_gof_test_mean)

#     amor_train_indices = pd.read_csv(
#         f"../../data/amor_calibration/{dataset}/z_train_0.csv"
#     )["index"].values
#     amor_test_indices = pd.read_csv(
#         f"../../data/amor_calibration/{dataset}/z_test_0.csv"
#     )["index"].values

#     amor_gof_train_mean, _ = goodness_of_fit(
#         z=torch.tensor(item_parms[amor_train_indices], dtype=torch.float32),
#         theta=torch.tensor(abilities, dtype=torch.float32),
#         y=torch.tensor(y[:, amor_train_indices], dtype=torch.float32),
#     )
#     amor_gof_train_means.append(amor_gof_train_mean)

#     amor_gof_test_mean, _ = goodness_of_fit(
#         z=torch.tensor(item_parms[amor_test_indices], dtype=torch.float32),
#         theta=torch.tensor(abilities, dtype=torch.float32),
#         y=torch.tensor(y[:, amor_test_indices], dtype=torch.float32),
#     )
#     amor_gof_test_means.append(amor_gof_test_mean)

# plugin_gof_df_train = pd.DataFrame(
#     {
#         "datasets": DATASETS,
#         "gof_means": plugin_gof_train_means,
#     }
# )
# plugin_gof_df_train.to_csv(f"{plot_dir}/nonamor4plugin_gof_train.csv", index=False)

# plugin_gof_df_test = pd.DataFrame(
#     {
#         "datasets": DATASETS,
#         "gof_means": plugin_gof_test_means,
#     }
# )
# plugin_gof_df_test.to_csv(f"{plot_dir}/nonamor4plugin_gof_test.csv", index=False)

# amor_gof_df_train = pd.DataFrame(
#     {
#         "datasets": DATASETS,
#         "gof_means": amor_gof_train_means,
#     }
# )
# amor_gof_df_train.to_csv(f"{plot_dir}/nonamor4amor_gof_train.csv", index=False)

# amor_gof_df_test = pd.DataFrame(
#     {
#         "datasets": DATASETS,
#         "gof_means": amor_gof_test_means,
#     }
# )
# amor_gof_df_test.to_csv(f"{plot_dir}/nonamor4amor_gof_test.csv", index=False)

# gof_df = pd.DataFrame(
#     {"datasets": DATASETS, "gof_means": gof_means, "gof_stds": gof_stds}
# )
# gof_df.to_csv(f"{plot_dir}/nonamor_calibration_gof.csv", index=False)

# ctt_df = pd.DataFrame(
#     {
#         "datasets": DATASETS,
#         "corr_ctt_means": corr_ctt_means,
#         "corr_ctt_stds": corr_ctt_stds,
#     }
# )
# ctt_df.to_csv(f"{plot_dir}/nonamor_calibration_corr_ctt.csv", index=False)

# helm_df = pd.DataFrame(
#     {
#         "datasets": [d for d in DATASETS if d != "airbench"],
#         "corr_helm_means": corr_helm_means,
#         "corr_helm_stds": corr_helm_stds,
#     }
# )
# helm_df.to_csv(f"{plot_dir}/nonamor_calibration_corr_helm.csv", index=False)

# error_bar_plot_single(
#     datasets=DATASETS,
#     means=gof_means,
#     stds=gof_stds,
#     plot_path=f"{plot_dir}/nonamor_calibration_summarize_gof",
#     xlabel=r"Goodness of Fit",
# )

# error_bar_plot_single(
#     datasets=DATASETS,
#     means=corr_ctt_means,
#     stds=corr_ctt_stds,
#     plot_path=f"{plot_dir}/nonamor_calibration_summarize_theta_corr_ctt",
#     xlabel=r"$\theta$ correlation with CTT",
# )

# error_bar_plot_single(
#     datasets=[d for d in DATASETS if d != "airbench"],
#     means=corr_helm_means,
#     stds=corr_helm_stds,
#     plot_path=f"{plot_dir}/nonamor_calibration_summarize_theta_corr_helm",
#     xlabel=r"$\theta$ correlation with HELM",
# )
