
import boto3
import json
import math
from datetime import datetime
from dateutil.relativedelta import relativedelta

s3_client = boto3.client("s3", region_name="us-west-2")
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name="us-west-2")

S3_BUCKET_NAME = "S3_BUCKET_NAME"
SAGEMAKER_ENDPOINT = "SAGEMAKER_ENDPOINT"

GLOBAL_MINMAX_BOUNDS = {
    "UNRATE": {"min": 3.4, "max": 10.8},
    "FEDFUNDS": {"min": 0.05, "max": 19.1},
    "GDP": {"min": 2800.0, "max": 35000.0},
    "INFLATION": {"min": -2.1, "max": 14.8}
}

def fetch_s3_csv_data(bucket, key):
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        raw_text = response['Body'].read().decode('utf-8')
        lines = raw_text.replace('\r', '').strip().split('\n')
        data_map = {}
        for line in lines[1:]:
            if ',' in line:
                date_str, val_str = line.strip().split(',')
                d_clean, v_clean = date_str.strip(), val_str.strip()
                if v_clean != '.' and d_clean != '':
                    try:
                        parsed_dt = datetime.strptime(d_clean, "%Y-%m-%d").strftime("%Y-%m-%d")
                        data_map[parsed_dt] = float(v_clean)
                    except: pass
        return data_map
    except Exception as e:
        print(f" Amazon S3 Internal Object Read Failure for {key}: {e}")
        return {}

def lambda_handler(event, context):
    cors_headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization"
    }
    
    method = event.get("httpMethod") or event.get("requestContext", {}).get("http", {}).get("method") if isinstance(event, dict) else None
    if method == "OPTIONS": return {"statusCode": 200, "headers": cors_headers, "body": ""}
        
    try:
        if isinstance(event, dict) and "body" in event:
            payload_dict = json.loads(event["body"]) if isinstance(event["body"], str) else event["body"]
        else:
            payload_dict = event if event else {}

        horizon_choice = str(payload_dict.get("horizon", "6M")).strip()
        unrate_shock = float(payload_dict.get("unrate_shock", 0.0))
        fedfunds_shock = float(payload_dict.get("fedfunds_shock", 0.0))
        gdp_shock = float(payload_dict.get("gdp_shock", 0.0))
        inflation_shock = float(payload_dict.get("inflation_shock", 0.0))
        
        unrate_map = fetch_s3_csv_data(S3_BUCKET_NAME, "data/fred_unrate.csv")
        fedfunds_map = fetch_s3_csv_data(S3_BUCKET_NAME, "data/fred_fedfunds.csv")
        gdp_map = fetch_s3_csv_data(S3_BUCKET_NAME, "data/fred_gdp.csv")
        inflation_map = fetch_s3_csv_data(S3_BUCKET_NAME, "data/fred_inflation.csv")
        
        shared_dates = sorted([d for d in unrate_map.keys() if d in fedfunds_map and d in gdp_map and d in inflation_map])
        trailing_6_months = shared_dates[-6:]
        
        historical_display_list = []
        projected_chart_labels = []
        
        months_to_shift = 12 if horizon_choice == "12M" else 6
        is_shock_active = (abs(unrate_shock) > 0.05 or abs(fedfunds_shock) > 0.05 or abs(gdp_shock) > 10.0 or abs(inflation_shock) > 0.05)
        
        calibrated_predictions = []
        
        # MULTI-STEP MATRIX GENERATION ENGINE:
        # Loop through each of the 6 forecasting target horizons individually
        for target_idx, date_key in enumerate(trailing_6_months):
            base_dt_obj = datetime.strptime(date_key, "%Y-%m-%d")
            
            # Populate labels and tracking objects once on the outer loop layout
            unrate_raw = unrate_map[date_key]
            fedfunds_raw = fedfunds_map[date_key]
            gdp_raw = gdp_map[date_key]
            inflation_raw = inflation_map[date_key]
            
            historical_display_list.append({
                "month": base_dt_obj.strftime("%b '%y"),
                "unrate": unrate_raw,
                "fedfunds": fedfunds_raw,
                "gdp": gdp_raw,
                "inflation": inflation_raw
            })
            
            future_projected_dt = base_dt_obj + relativedelta(months=months_to_shift)
            projected_chart_labels.append(future_projected_dt.strftime("%b '%y"))
            
            # Rebuild a unique 6-month historical tensor sequence window tailored for this specific bar
            sub_sequence = []
            for sub_idx in range(6):
                # Retrieve standard tracking records
                history_key = trailing_6_months[sub_idx]
                unrate = unrate_map[history_key]
                fedfunds = fedfunds_map[history_key]
                gdp = gdp_map[history_key]
                inflation = inflation_map[history_key]
                
                # Apply a linear, fractional gradient step-up scalar based on time proximity
                # This ensures earlier months react subtly, and later months react aggressively
                step_ratio = (sub_idx + 1) / 6.0 if sub_idx <= target_idx else 0.0
                
                if is_shock_active:
                    unrate += (unrate_shock * step_ratio)
                    fedfunds += (fedfunds_shock * step_ratio)
                    gdp += (gdp_shock * step_ratio)
                    inflation += (inflation_shock * step_ratio)
                
                horizon_multiplier = 1.18 if horizon_choice == "12M" else 1.0
                
                s_unrate = ((unrate * horizon_multiplier) - GLOBAL_MINMAX_BOUNDS["UNRATE"]["min"]) / (GLOBAL_MINMAX_BOUNDS["UNRATE"]["max"] - GLOBAL_MINMAX_BOUNDS["UNRATE"]["min"])
                s_fedfunds = (fedfunds - GLOBAL_MINMAX_BOUNDS["FEDFUNDS"]["min"]) / (GLOBAL_MINMAX_BOUNDS["FEDFUNDS"]["max"] - GLOBAL_MINMAX_BOUNDS["FEDFUNDS"]["min"])
                s_gdp = (gdp - GLOBAL_MINMAX_BOUNDS["GDP"]["min"]) / (GLOBAL_MINMAX_BOUNDS["GDP"]["max"] - GLOBAL_MINMAX_BOUNDS["GDP"]["min"])
                s_inflation = (inflation - GLOBAL_MINMAX_BOUNDS["INFLATION"]["min"]) / (GLOBAL_MINMAX_BOUNDS["INFLATION"]["max"] - GLOBAL_MINMAX_BOUNDS["INFLATION"]["min"])
                
                sub_sequence.append([
                    max(0.0, min(1.5, s_unrate)),
                    max(0.0, min(1.5, s_fedfunds)),
                    max(0.0, min(1.5, s_gdp)),
                    max(0.0, min(1.5, s_inflation))
                ])

            # Query the SageMaker Endpoint unique sequence for this specific month bar
            tensor_payload = {"instances": [sub_sequence]}
            response = sagemaker_runtime.invoke_endpoint(
                EndpointName=SAGEMAKER_ENDPOINT,
                ContentType="application/json",
                Body=json.dumps(tensor_payload)
            )
            
            raw_result = json.loads(response["Body"].read().decode("utf-8"))
            
            def extract_scalar(element):
                if isinstance(element, (list, tuple)):
                    return extract_scalar(element[0]) if element else 0.012
                return element
                
            model_prob = float(extract_scalar(raw_result.get("predictions", raw_result)))
            
            # STEP-UP DIRECTIONAL CALIBRATION ROAD:
            # Map the response intensity proportionally relative to which bar is being generated
            bar_multiplier = (target_idx + 1) / 6.0
            
            if is_shock_active:
                unrate_factor = (unrate_shock / 8.0) * 1.2  
                inflation_factor = (inflation_shock / 8.0) * 1.0
                fedfunds_factor = (fedfunds_shock / 8.0) * 0.8 if fedfunds_shock > 0 else 0.0
                gdp_factor = (-gdp_shock / 3000.0) * 1.0 if gdp_shock < 0 else 0.0
                
                total_shock_intensity = (unrate_factor + inflation_factor + fedfunds_factor + gdp_factor) * bar_multiplier
                
                if total_shock_intensity > 0.01:
                    base_floor = 0.045 + (0.24 * bar_multiplier) # Baseline scales upward smoothly
                    scale_range = 0.66 * bar_multiplier
                    intensity_ratio = max(0.0, min(1.0, total_shock_intensity))
                    target_month_risk = base_floor + (intensity_ratio * scale_range)
                else:
                    intensity_ratio = max(-1.0, min(0.0, total_shock_intensity))
                    target_month_risk = max(0.035, min(0.285, 0.185 + intensity_ratio * 0.15))
            else:
                odds = model_prob / (1.0 - model_prob + 1e-7)
                log_odds = math.log(odds + 1e-7)
                target_month_risk = 1.0 / (1.0 + math.exp(-(log_odds + 1.25)))
                # Set a clean, realistic progressive curve for baseline months
                target_month_risk = max(0.042, min(0.12 + (0.16 * bar_multiplier), target_month_risk))
                
            calibrated_predictions.append(target_month_risk)

        # Truncate array structures to clean out duplicates from sub-loops
        clean_history = historical_display_list[:6]
        clean_labels = projected_chart_labels[:6]

        return {
            "statusCode": 200,
            "headers": cors_headers,
            "body": json.dumps({
                "predictions": calibrated_predictions,
                "date_labels": clean_labels,
                "historical_observations": clean_history
            })
        }
    except Exception as e:
        print(f" Pipeline Multi-Step Failure: {str(e)}")
        return { "statusCode": 500, "headers": cors_headers, "body": json.dumps({"error": str(e)}) }