
import json
import boto3
import numpy as np
from decimal import Decimal
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def decimal_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError

def bubble_ml_processor(event, context):
    """Main ML processing function for Bubble integration"""
    try:
        body = json.loads(event.get('body', '{}')) if isinstance(event.get('body'), str) else event.get('body', {})
        
        operation = body.get('operation')
        data = body.get('data')
        parameters = body.get('parameters', {})
        
        logger.info(f"Processing operation: {operation}")
        
        result = {}
        
        if operation == 'predict':
            result = perform_prediction(data, parameters)
        elif operation == 'cluster':
            result = perform_clustering(data, parameters)
        elif operation == 'optimize':
            result = perform_optimization(data, parameters)
        else:
            result = {'error': f'Unsupported operation: {operation}'}
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps(result, default=decimal_default)
        }
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'error': str(e)})
        }

def perform_prediction(data, parameters):
    """Advanced prediction using ensemble methods"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    
    df = pd.DataFrame(data)
    target_column = parameters.get('target_column', df.columns[-1])
    features = df.drop(columns=[target_column])
    target = df[target_column]
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(features_scaled, target)
    predictions = model.predict(features_scaled)
    
    return {
        'model_type': "Random Forest",
        'predictions': predictions.tolist(),
        'feature_importance': model.feature_importances_.tolist(),
        'feature_names': features.columns.tolist()
    }
        