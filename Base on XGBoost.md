# data:
## 基础配置
    start_date = '2010-01-01'
    split_date = '2015-01-01'
    end_date = '2018-01-01'
    instrument = D.instruments(start_date=start_date, end_date=end_date, market='CN_STOCK_A')

## 获取每年年报公告后的第一个交易日历
    trading_days = D.trading_days(market='CN', start_date=start_date, end_date=end_date)
    trading_days['month'] = trading_days.date.map(lambda x:x.month)
    trading_days['year'] = trading_days.date.map(lambda x:x.year)
    groupby_td = trading_days.groupby(['year','month']).apply(lambda x:x.head(1))
    first_date_after_financial_report = list(groupby_td[groupby_td['month']==5].date) # 5月第一个交易日
    first_date_after_financial_report = [i.strftime('%Y-%m-%d') for i in first_date_after_financial_report] # date转换为str 

## 特征列表
    financial_features_fields = ['date','fs_roe_0','fs_bps_0','fs_operating_revenue_ttm_0','fs_current_assets_0','fs_non_current_assets_0',
                  'fs_roa_0','fs_total_profit_0','fs_free_cash_flow_0','adjust_factor_0','fs_eps_0','pe_ttm_0','close_0',
                  'fs_common_equity_0','fs_net_income_0','market_cap_0','fs_eps_yoy_0','beta_szzs_90_0','fs_net_profit_margin_ttm_0',   
                 ]

## 按年获取财务特征数据
    def get_financial_features(date,instrument=instrument,fields=financial_features_fields):
        assert type(date) == str  
        df = D.features(instrument, date, date, fields)
        return df

## 获取财务特征数据，采取缓存的形式，可以节省运行时间
    def get_financial_features_cache():
        print('获取财务特征数据，并缓存！')
        financial_features  = pd.DataFrame()
        for dt in first_date_after_financial_report:
            df = get_financial_features(dt)
            financial_features = financial_features.append(df)   
        return Outputs(financial_features=DataSource.write_df(financial_features))
    m1 = M.cached.v2(run=get_financial_features_cache)
    financial_features_df = m1.financial_features.read_df()  
 
## 获取日线特征数据
    daily_history_features_fields = ['close','amount','pb_lf']  # 标注也在这里获取
    def get_daily_history_features(start_date=start_date,end_date=end_date,instrument=instrument,fields=daily_history_features_fields):
        df = D.history_data(instrument,start_date,end_date,fields)
        return df 

## 按股票groupby 计算日线特征
    def calcu_daily_history_features(df):
        df['mean_amount'] = pd.rolling_apply(df['amount'], 22, np.nanmean)/df['amount']
        df['month_1_mom'] = df['close']/df['close'].shift(22)
        df['month_12_mom'] = df['close']/df['close'].shift(252)
        df['volatity'] = pd.rolling_apply(df['close'], 90, np.nanstd)/df['close']
        return df 

## 获取日线特征，采取缓存的形式，可以节省运行时间
    def get_daily_features_cache():
        print('获取日线特征数据，并缓存！')
        daily_history_features = get_daily_history_features().groupby('instrument').apply(calcu_daily_history_features)
        return Outputs(daily_features=DataSource.write_df(daily_history_features))
    m2 = M.cached.v2(run=get_daily_features_cache)
    daily_features_df = m2.daily_features.read_df()  
 
## 财务特征和日线特征合并
    result=financial_features_df.merge(daily_features_df, on=['date', 'instrument'], how='inner') 

# 抽取衍生特征
## 资产周转率
    result['asset_turnover'] = result['fs_operating_revenue_ttm_0']/(result['fs_non_current_assets_0'] + result['fs_current_assets_0']) 
## 总盈利/总资产
    result['gross_profit_to_asset'] = result['fs_total_profit_0']/(result['fs_non_current_assets_0'] + result['fs_current_assets_0']) 
## 自营现金流/总资产
    result['cash_flow_to_assets'] = result['fs_free_cash_flow_0']/(result['fs_non_current_assets_0'] + result['fs_current_assets_0'])
## 总收入/价格 
    result['sales_yield'] = result['fs_operating_revenue_ttm_0']/result['close_0']
# 现金流/股数/股价
    result['cash_flow_yield'] = result['fs_free_cash_flow_0']/(result['fs_common_equity_0']/result['close'])/result['close']
# 营业收入 Sales to EV
    result['sales_to_ev'] = result['fs_operating_revenue_ttm_0']/result['fs_common_equity_0']
#  EBITDA to EV 
    result['ebitda_to_ev'] = result['fs_net_income_0']/result['fs_common_equity_0']

    def judge_positive_earnings(df):   
        if df['adjust_factor_0'] > df['adjust_factor_0_forward']:
            return 1
        else:
            return 0
    
# 构建时序衍生特征函数（# 一年前的pe # 一年前的总收入/价格 # 复权因子哑变量）
    def construct_derivative_features(tmp):
        tmp['pe_forward'] = tmp['pe_ttm_0'].shift(1) 
        tmp['sales_yield_forward'] = tmp['sales_yield'].shift(1) 
        tmp['adjust_factor_0_forward'] = tmp['adjust_factor_0'].shift(1)
        tmp['positive_earnings'] = tmp.apply(judge_positive_earnings,axis=1)
        # 标注数据构建
        tmp['label'] = tmp['pb_lf'].shift(-1)
        return tmp 

    features_df = result.groupby('instrument').apply(construct_derivative_features)

# 去极值和标准化
## 哪些特征需要进行 去极值和标准化处理
    need_deal_with_features = ['fs_free_cash_flow_0',  'fs_eps_0',
           'fs_operating_revenue_ttm_0', 'market_cap_0', 'fs_roe_0',
           'fs_current_assets_0', 'fs_roa_0', 'pe_ttm_0',
           'fs_non_current_assets_0', 'fs_eps_yoy_0', 'fs_bps_0', 'close_0',
           'adjust_factor_0', 'fs_common_equity_0', 'fs_net_income_0',
           'fs_total_profit_0', 'amount', 'pb_lf', 'mean_amount',
           'asset_turnover', 'gross_profit_to_asset', 'cash_flow_to_assets',
           'sales_yield', 'cash_flow_yield', 'sales_to_ev', 'ebitda_to_ev',
           'pe_forward', 'sales_yield_forward', 'adjust_factor_0_forward','label']

## 去极值
    def remove_extremum(df,features=need_deal_with_features):
        factor_list = features
        for factor in factor_list:
            df[factor][df[factor] >= np.percentile(df[factor], 95)] = np.percentile(df[factor], 95)
            df[factor][df[factor] <= np.percentile(df[factor], 5)] = np.percentile(df[factor], 5)
        return df

## 标准化
    def standardization(df,features=need_deal_with_features):
        factor_list = features
        for factor in factor_list:
            df[factor] = (df[factor] - df[factor].mean()) / df[factor].std()
        return df 

    def deal_with_features(df):
        return standardization(remove_extremum(df))

    features_df_after_deal_with = features_df.groupby('date').apply(deal_with_features)
 
## 整理因子和标注
    key_attr = ['instrument', 'date',]
    explained_features = [ 'fs_eps_0','fs_net_profit_margin_ttm_0','fs_bps_0','asset_turnover','fs_roa_0','fs_roe_0',
           'gross_profit_to_asset','cash_flow_to_assets', 'positive_earnings', 'pe_forward','sales_yield', 'sales_yield_forward',
            'cash_flow_yield', 'sales_to_ev', 'ebitda_to_ev','market_cap_0',
           'beta_szzs_90_0', 'month_1_mom', 'month_12_mom', 'volatity','mean_amount', 'fs_eps_yoy_0']
    label = ['label']
    final_data = features_df_after_deal_with[label+key_attr+explained_features +['pb_lf']]

    al:
    import xgboost as xgb # 导入包

## 样本内的数据同样 划分训练数据和测试数据
    assert len(final_data.columns) == 26
    data = final_data[final_data['date'] <= '2015-01-01'] # 样本内数据
    data = data[~pd.isnull(data[label[0]])] # 删除标注为缺失值的
    data = data[key_attr+label+explained_features].dropna()  # 删除 特征为缺失值的
    data.index = range(len(data))
    train_data = data.ix[:int(len(data)*0.8)]  # 80%的数据拿来训练
    test_data = data.ix[int(len(data)*0.8):]   # 剩下的数据拿来验证

## 数据按 特征和标注处理，便于xgboost构建对象
    X_train = train_data[explained_features]
    y_train = train_data[label[0]]
    X_test = test_data[explained_features]
    y_test = test_data[label[0]]

## xgboost 构建对象
    dtrain = xgb.DMatrix(X_train.values,label=y_train.values) # 这个地方如果是X_train 其实不影响结果
    dtest = xgb.DMatrix(X_test.values,label=y_test.values)

## 设置参数，参数的格式用map的形式存储
    param = {'max_depth': 3,                  # 树的最大深度
             'eta': 0.1,                        # 一个防止过拟合的参数，默认0.3
             'n_estimators':100,                 # Number of boosted trees to fit 
             'silent': 1,                     # 打印信息的繁简指标，1表示简， 0表示繁
             'objective': 'reg:linear'}  # 使用的模型，分类的数目 

    num_round = 100 # 迭代的次数
## 看板，每次迭代都可以在控制台打印出训练集与测试集的损失
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]

## 训练模型
    bst = xgb.train(param, dtrain, num_round, evals=watchlist)

## 测试集上模型预测
    preds = bst.predict(dtest)  
# preds

## 样本外数据 
    out_of_sample_data = final_data[final_data['date'] > '2015-01-01'] # 样本内数据
    out_of_sample_data = out_of_sample_data[key_attr+explained_features+['pb_lf']]  #  取出 特征数据
    assert len(out_of_sample_data.columns) == 25

X_TEST = out_of_sample_data[explained_features]
    out_of_sample_dtset = xgb.DMatrix(X_TEST.values) 

## 样本外预测
    out_of_sample_preds = bst.predict(out_of_sample_dtset)
    out_of_sample_data['predict_pb_lf'] = out_of_sample_preds
