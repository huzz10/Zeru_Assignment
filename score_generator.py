import pandas as pd
import json
import numpy as np
import lightgbm as lgb
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_transactions(file_path):
    """Loads transactions from a JSON file, handling both .json and .zip formats."""
    try:
        if file_path.endswith('.zip'):
            import zipfile
            with zipfile.ZipFile(file_path, 'r') as zf:
                # Assuming the JSON file inside is the first .json file
                json_file_name = next((f for f in zf.namelist() if f.endswith('.json')), None)
                if json_file_name:
                    with zf.open(json_file_name) as f:
                        data = json.load(f)
                else:
                    raise ValueError("No JSON file found inside the ZIP archive.")
        else:
            with open(file_path, 'r') as f:
                data = json.load(f)
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error loading transactions from {file_path}: {e}")
        return pd.DataFrame()

def feature_engineer(df):
    """
    Engineers features for each wallet from the transaction DataFrame.
    Features include basic aggregations, transaction type specific metrics,
    liquidation-related features, and various ratios.
    """
    if df.empty:
        return pd.DataFrame()

    # Correctly use 'timestamp' column as per user's file snippet
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    # Standardize walletAddress column name
    # The snippet shows 'userWallet', so we'll use that and rename it for consistency
    if 'userWallet' in df.columns:
        df.rename(columns={'userWallet': 'walletAddress'}, inplace=True)
    elif 'walletAddress' not in df.columns:
        print("Warning: 'userWallet' or 'walletAddress' column not found. Please check data schema.")
        return pd.DataFrame()

    # --- MODIFICATION STARTS HERE ---
    # Extract 'amount' and 'assetPriceUSD' from 'actionData' and calculate 'value_usd'
    # Use .get() with a default value to handle cases where 'actionData' or keys might be missing
    df['amount'] = df['actionData'].apply(lambda x: float(x.get('amount', 0)) if isinstance(x, dict) else 0)
    df['assetPriceUSD'] = df['actionData'].apply(lambda x: float(x.get('assetPriceUSD', 0)) if isinstance(x, dict) else 0)

    # Calculate transaction value in USD
    # Normalize amount by a common decimal (e.g., 1e18 for Ether, 1e6 for USDC)
    # The snippet shows USDC amount as '2000000000', which implies 6 decimals (2 * 10^9 / 10^6 = 2000)
    # You might need to adjust the divisor based on the actual asset's decimals.
    # For now, let's assume a common large divisor or directly use it as a large integer.
    # A safer approach is to check assetSymbol and use appropriate decimals.
    # For simplicity, if 'amount' is already in smallest unit, divide by 1e18 or 1e6 based on asset.
    # Given the snippet 'amount': '2000000000' and 'USDC', it is likely 6 decimals.
    df['value_usd'] = (df['amount'] / (10**6)) * df['assetPriceUSD'] # Assuming USDC has 6 decimals

    # Fill any NaN in value_usd that might arise from missing actionData or prices
    df['value_usd'] = df['value_usd'].fillna(0)
    # --- MODIFICATION ENDS HERE ---

    # Group by wallet address to create features
    wallet_features = df.groupby('walletAddress').apply(lambda x: pd.Series({
        'total_transactions': len(x),
        'unique_days_active': x['timestamp'].dt.date.nunique(),
        'time_span_days': (x['timestamp'].max() - x['timestamp'].min()).days if len(x) > 1 else 0,
        # Use 'value_usd' for volume calculations
        'total_deposit_volume_usd': x[x['action'] == 'deposit']['value_usd'].sum(),
        'total_borrow_volume_usd': x[x['action'] == 'borrow']['value_usd'].sum(),
        'total_repay_volume_usd': x[x['action'] == 'repay']['value_usd'].sum(),
        'total_redeem_volume_usd': x[x['action'] == 'redeemunderlying']['value_usd'].sum(),
        'liquidation_count': x[x['action'] == 'liquidationCall'].shape[0],
        'liquidation_volume_usd': x[x['action'] == 'liquidationCall']['value_usd'].sum(),
    })).reset_index()

    # Calculate ratios, handling potential division by zero
    wallet_features['repay_ratio'] = wallet_features.apply(
        lambda row: row['total_repay_volume_usd'] / row['total_borrow_volume_usd'] if row['total_borrow_volume_usd'] > 0 else (1 if row['total_repay_volume_usd'] > 0 else 0), axis=1
    )
    wallet_features['redeem_ratio'] = wallet_features.apply(
        lambda row: row['total_redeem_volume_usd'] / row['total_deposit_volume_usd'] if row['total_deposit_volume_usd'] > 0 else (1 if row['total_redeem_volume_usd'] > 0 else 0), axis=1
    )

    # Cap ratios to a reasonable maximum (e.g., 2.0 to prevent extreme outliers)
    wallet_features['repay_ratio'] = wallet_features['repay_ratio'].clip(upper=2.0)
    wallet_features['redeem_ratio'] = wallet_features['redeem_ratio'].clip(upper=2.0)

    # Handle cases where `time_span_days` might be 0 for single-transaction wallets
    # or to avoid division by zero in frequency calculations
    wallet_features['transactions_per_day'] = wallet_features.apply(
        lambda row: row['total_transactions'] / row['time_span_days'] if row['time_span_days'] > 0 else row['total_transactions'], axis=1
    )

    return wallet_features

def generate_heuristic_scores(features_df):
    """
    Generates a heuristic score (0-1000) for training purposes.
    This function defines what constitutes a "good" or "bad" wallet based on engineered features.
    Adjust weights and logic based on your understanding of Aave V2 behavior.
    """
    if features_df.empty:
        return pd.Series(dtype=float)

    scores = pd.Series(0.0, index=features_df.index)

    # Base score, assuming average behavior is around the midpoint
    scores += 500

    # Positive indicators:
    # More transactions generally implies more engagement and liquidity provision
    scores += (features_df['total_transactions'] / features_df['total_transactions'].max()) * 100

    # More unique active days means consistent usage, less bot-like
    scores += (features_df['unique_days_active'] / features_df['unique_days_active'].max()) * 150

    # Higher repay ratio is a strong positive indicator of responsible borrowing
    scores += features_df['repay_ratio'] * 150 # Max 2.0 * 150 = 300

    # Higher redeem ratio implies users responsibly withdrawing
    scores += features_df['redeem_ratio'] * 75 # Max 2.0 * 75 = 150

    # Consistent activity (transactions per day)
    scores += (features_df['transactions_per_day'] / features_df['transactions_per_day'].max()) * 50

    # Negative indicators:
    # Liquidations are very strong negative indicators
    # Apply a non-linear penalty to make initial liquidations hurt more
    scores -= features_df['liquidation_count'].apply(lambda x: min(x * 100, 400)) # Up to -400 for liquidations
    scores -= np.log1p(features_df['liquidation_volume_usd']) * 20 # Penalize based on log of volume

    # Normalize/clip scores to 0-1000
    scores = scores.clip(lower=0, upper=1000)
    # Re-scale to ensure the range is fully utilized if needed, but clip handles boundary
    # If the max generated score is less than 1000, this will stretch it.
    # For a fixed 0-1000 range, clipping is usually sufficient.
    if scores.max() > 0:
        scores = scores / scores.max() * 1000

    return scores.round().astype(int)

def train_model(features_df, heuristic_scores):
    """
    Trains a LightGBM Regressor model using the engineered features and heuristic scores.
    """
    if features_df.empty or heuristic_scores.empty or len(features_df) == 0:
        print("No sufficient data to train the model.")
        return None

    # Features to use for training (excluding 'walletAddress' and raw volumes if ratios are preferred)
    # Ensure these features align with what's engineered and what your model expects
    feature_columns = [
        'total_transactions', 'unique_days_active', 'time_span_days',
        'repay_ratio', 'redeem_ratio', 'liquidation_count', 'transactions_per_day',
        'total_deposit_volume_usd', 'total_borrow_volume_usd', 'total_repay_volume_usd',
        'total_redeem_volume_usd', 'liquidation_volume_usd'
    ]
    # Filter for columns that actually exist in features_df
    X = features_df[[col for col in feature_columns if col in features_df.columns]]
    y = heuristic_scores

    # Handle potential inf or large values in features if any from divisions
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0) # Replace inf with 0 or a reasonable value

    if X.shape[0] == 0 or X.shape[1] == 0:
        print("No valid features for training.")
        return None

    model = lgb.LGBMRegressor(objective='regression_l1', metric='mae', n_estimators=100, learning_rate=0.1, num_leaves=31, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model

def generate_wallet_scores(json_file_path, model=None):
    """
    Main function to generate wallet scores.
    If a model is not provided (e.g., in a standalone script), it will train one
    based on the heuristic scores from the input data itself.
    For production, you would load a pre-trained model.
    """
    print(f"Loading transactions from: {json_file_path}")
    transactions_df = load_transactions(json_file_path)
    if transactions_df.empty:
        print("No transactions loaded. Exiting.")
        return pd.DataFrame()

    print("Engineering features...")
    features_df = feature_engineer(transactions_df)
    if features_df.empty:
        print("No features engineered. Exiting.")
        return pd.DataFrame()

    if model is None:
        print("No pre-trained model provided. Generating heuristic scores and training a model...")
        heuristic_scores = generate_heuristic_scores(features_df)
        if heuristic_scores.empty:
            print("Could not generate heuristic scores. Exiting.")
            return pd.DataFrame()
        model = train_model(features_df, heuristic_scores)
        if model is None:
            print("Model training failed. Exiting.")
            return pd.DataFrame()
    else:
        print("Using provided pre-trained model.")

    print("Predicting scores...")
    # Prepare features for prediction, ensuring column order matches training
    prediction_feature_columns = [
        'total_transactions', 'unique_days_active', 'time_span_days',
        'repay_ratio', 'redeem_ratio', 'liquidation_count', 'transactions_per_day',
        'total_deposit_volume_usd', 'total_borrow_volume_usd', 'total_repay_volume_usd',
        'total_redeem_volume_usd', 'liquidation_volume_usd'
    ]
    X_predict = features_df[[col for col in prediction_feature_columns if col in features_df.columns]]
    X_predict = X_predict.replace([np.inf, -np.inf], np.nan).fillna(0)

    if X_predict.shape[0] == 0 or X_predict.shape[1] == 0:
        print("No valid features for prediction.")
        return pd.DataFrame()

    predicted_scores = model.predict(X_predict)
    # Scale and clip to 0-1000
    predicted_scores = np.clip(predicted_scores, 0, 1000).round().astype(int)

    result_df = pd.DataFrame({
        'walletAddress': features_df['walletAddress'],
        'credit_score': predicted_scores
    })

    return result_df

if __name__ == '__main__':
    # You need to download one of the provided files and place it in the same directory
    # or provide the full path to the downloaded file.
    # The snippet indicates the file is 'user-wallet-transactions.json'
    transaction_file = 'user-wallet-transactions.json' # Make sure this path is correct!

    # Check if the file exists
    if not os.path.exists(transaction_file):
        print(f"Error: Transaction file '{transaction_file}' not found.")
        print("Please download it from the provided Google Drive link and place it in the current directory.")
        print("Or update 'transaction_file' variable with the correct path.")
    else:
        wallet_scores_df = generate_wallet_scores(transaction_file)

        if not wallet_scores_df.empty:
            print("\nSample Wallet Scores:")
            print(wallet_scores_df.head())

            # Save scores to CSV
            output_csv_path = 'wallet_credit_scores.csv'
            wallet_scores_df.to_csv(output_csv_path, index=False)
            print(f"\nWallet scores saved to {output_csv_path}")

            # Basic analysis (for analysis.md)
            print("\nScore Distribution Statistics:")
            print(wallet_scores_df['credit_score'].describe())

            # Plot score distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(wallet_scores_df['credit_score'], bins=20, kde=True, color='skyblue')
            plt.title('Distribution of Wallet Credit Scores')
            plt.xlabel('Credit Score')
            plt.ylabel('Number of Wallets')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()

            # Optional: Detailed analysis of low/high score wallets (for analysis.md)
            # You would join the scores back to the full features_df to analyze their characteristics.
            # E.g.,
            # all_data_with_scores = features_df.merge(wallet_scores_df, on='walletAddress')
            # low_score_wallets_data = all_data_with_scores[all_data_with_scores['credit_score'] < 200]
            # high_score_wallets_data = all_data_with_scores[all_data_with_scores['credit_score'] > 800]
            # print("\nCharacteristics of Low Score Wallets (Example):")
            # print(low_score_wallets_data.describe())
            # print("\nCharacteristics of High Score Wallets (Example):")
            # print(high_score_wallets_data.describe())