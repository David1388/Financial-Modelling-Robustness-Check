from models_task3 import MyTransaction, XgTransaction, logger
from gbmmc import (
    Read_Data,
    Gbm_MC,
    median_95,
    log_return,
    VaR,
    plot_Simulation,
    plot_VaR,
    plot_distribution,
)
from xgmodel import (
    Calculate_portfolio,
    Calculate_features,
    Constructing_training_data,
    Training_XGBoost,
    Prediction_test,
    Evaluate_model,
    Future_predictions,
    Max_Drawdown,
    Plot_training_testing,
    plot_importance_features,
    polt_future,
    Plot_Max_Drawdown,
    xglog_return,
)


def main():
    txn = MyTransaction()
    xgt = XgTransaction()
    pf = ["data/Min Variance.csv", "data/Max Sharpe.csv"]
    for portfolio_file in pf:
        # GBM
        logger.info(f"\n# ========== {portfolio_file} ==========")
        Read_Data(txn, portfolio_file)
        print(f"\n ========== {portfolio_file} ==========")
        logger.info("\n## ------ 'GBM' ------")
        print("\n ------ 'GBM' ------")
        Gbm_MC(txn,sigma_multiplier = 1)
        median_95(txn)
        logger.info("\n### ---- Return ----")
        print("\n ---- Return ----")
        log_return(txn)
        logger.info("\n### ---- Risk ----")
        print("\n ---- Risk ----")
        var_95_future, cvar_95_future = VaR(txn)
        txn.store_results(portfolio_file)  
        plot_Simulation(txn, portfolio_file)
        plot_VaR(txn, portfolio_file)
       
        #economic downturns or volatility shocks
        print("\n ------ 'volatility shocks' ------")
        Gbm_MC(txn,sigma_multiplier = 2.42)
        median_95(txn)
        logger.info("\n### ---- Return ----")
        print("\n ---- Return ----")
        log_return(txn)
        logger.info("\n### ---- Risk ----")
        print("\n ---- Risk ----")
        var_95_future, cvar_95_future = VaR(txn)
        txn.volatility_results(portfolio_file) 
        plot_Simulation(txn, f"{portfolio_file}_Volatility_Shocks")
        plot_VaR(txn, f"{portfolio_file}_Volatility_Shocks")

        
        # XGboost
        logger.info("\n## ------ 'XGboost' ------")
        print("\n ------ 'XGboost' ------")
        Calculate_portfolio(xgt, txn)
        Calculate_features(xgt)
        Constructing_training_data(xgt)
        model = Training_XGBoost(xgt)
        Prediction_test(xgt, model)
        logger.info("\n### ---- Evaluate model ----")
        print("\n ---- Evaluate model ----")
        Evaluate_model(xgt)
        Future_predictions(model, xgt)
        logger.info("\n### ---- Return ----")
        print("\n ---- Return ----")
        xglog_return(xgt)
        logger.info("\n### ---- Risk ----")
        print("\n ---- Risk ----")
        Max_Drawdown(xgt)
        xgt.xg_results(portfolio_file)
        Plot_training_testing(xgt, portfolio_file)
        plot_importance_features(model)
        polt_future(xgt, portfolio_file)
        Plot_Max_Drawdown(xgt, portfolio_file)
    plot_distribution(txn)
    print("\n ========== END ==========")
    return txn.results, txn, xgt, xgt.xgresults


if __name__ == "__main__":
    txn_xgt = main()
