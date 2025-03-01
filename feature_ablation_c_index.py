# تابع برای انجام حذف ویژگی با استفاده از شاخص c-index
def feature_ablation_c_index(model, train_x, test_x, df_test, feature_names, device):
    feature_importances = {}

    # محاسبه baseline c-index قبل از حذف ویژگی‌ها
    model.eval()
    with torch.no_grad():
        test_predict = model(test_x)  # پیش‌بینی پارامترهای Weibull با استفاده از متغیرهای مستقل
        test_predict = pd.DataFrame(test_predict.cpu().numpy(), columns=["pred_alpha", "pred_beta", "pred_gamma", "pred_landa"])

        test_result = df_test.copy()
        test_result.reset_index(inplace=True)
        test_result = pd.concat([test_result, test_predict], axis=1)
        test_result.set_index("index", drop=True, inplace=True)

        t_max = df_test["time"].max()
        num_vals = max(math.ceil(t_max), 50)
        t_vals = np.linspace(0, t_max, num_vals)

        surv = weibull_surv(t_vals, test_result["pred_alpha"].to_numpy(),
                            test_result["pred_beta"].to_numpy(),
                            test_result["pred_gamma"].to_numpy(),
                            test_result["pred_landa"].to_numpy())
        surv = pd.DataFrame(data=surv, index=t_vals)

        test_time = df_test['time'].values
        test_status = df_test['death'].values

        ev = EvalSurv(surv, test_time, test_status, censor_surv='km')
        baseline_c_index = ev.concordance_td()
        print(f"Baseline c-index: {baseline_c_index:.4f}")

    for feature_idx, feature_name in enumerate(feature_names):
        # حذف ویژگی با قرار دادن آن به مقدار صفر
        X_train_ablated = train_x.clone()
        X_test_ablated = test_x.clone()
        X_train_ablated[:, feature_idx] = 0
        X_test_ablated[:, feature_idx] = 0

        # بازآموزی مدل با داده‌های حذف‌شده
        model.train()
        optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)  # تنظیم مجدد بهینه‌ساز
        for epoch in range(5):  # آموزش مختصر برای داده‌های حذف‌شده
            optimizer.zero_grad()
            pred_y = model(X_train_ablated)
            loss = deep_MWD_loss(train_y, pred_y)

            loss.backward()
            optimizer.step()

        # ارزیابی مدل بر روی داده‌های حذف‌شده
        model.eval()
        with torch.no_grad():
            test_predict_ablated = model(X_test_ablated)
            test_predict_ablated = pd.DataFrame(test_predict_ablated.cpu().numpy(), columns=["pred_alpha", "pred_beta", "pred_gamma", "pred_landa"])

            test_result_ablated = df_test.copy()
            test_result_ablated.reset_index(inplace=True)
            test_result_ablated = pd.concat([test_result_ablated, test_predict_ablated], axis=1)
            test_result_ablated.set_index("index", drop=True, inplace=True)

            surv_ablated = weibull_surv(t_vals, test_result_ablated["pred_alpha"].to_numpy(),
                                        test_result_ablated["pred_beta"].to_numpy(),
                                        test_result_ablated["pred_gamma"].to_numpy(),
                                        test_result_ablated["pred_landa"].to_numpy())
            surv_ablated = pd.DataFrame(data=surv_ablated, index=t_vals)

            # اندازه‌گیری عملکرد با استفاده از c-index
            ev_ablated = EvalSurv(surv_ablated, test_time, test_status, censor_surv='km')
            ablated_c_index = ev_ablated.concordance_td()
            feature_importances[feature_name] = baseline_c_index - ablated_c_index
            print(f"Feature: {feature_name}, Ablated c-index: {ablated_c_index:.4f}, Impact: {feature_importances[feature_name]:.4f}")

    return feature_importances

# لیست نام ستون‌های ویژگی‌ها
feature_names = df_train.drop(['death', 'time'], axis=1).columns

# انجام حذف ویژگی با استفاده از c-index
feature_importances_c_index = feature_ablation_c_index(model, train_x, test_x, df_test, feature_names, device)
