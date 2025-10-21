import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

def train_model(df):
    # Features numéricas e categóricas
    numerical_features = [
        'hora', 'valor', 'tentativas_senha', 'score_credito',
        'desvio_valor', 'media_valor_user', 'fraudes_passadas',
        'limite_cartao', 'saldo_conta', 'distancia_ultima_transacao',
        'freq_transacao_destinatario', 'valor_total_enviado_destinatario',
        'velocidade_transacoes', 'intervalo_medio_transacoes', 'score_valor_z',
        'dia_semana' # Dia da semana é numérico (0-6)
    ]
    categorical_features = [
        'tipo_transacao', 'periodo_dia', 'tipo_dispositivo', 'os', 'tipo_ip'
    ]
    binary_features = [
        'dispositivo_novo', 'ip_novo', 'mudanca_localizacao', 'destinatario_novo',
        'fim_semana', 'e_vpn', 'senha_alterada_recentemente'
    ]

    # Preenche NaNs com 0 para numéricas e 'unknown' para categóricas
    for col in numerical_features:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')
    for col in binary_features:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Combina todas as features que o modelo usará
    all_features = numerical_features + categorical_features + binary_features
    
    # Garante que todas as features existem no DataFrame
    for feature in all_features:
        if feature not in df.columns:
            if feature in numerical_features:
                df[feature] = 0
            elif feature in categorical_features:
                df[feature] = 'unknown'
            elif feature in binary_features:
                df[feature] = 0

    X = df[all_features]
    y = df['is_fraud']
    
    # Divisão train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Pré-processamento: Escala para numéricas, One-Hot Encoding para categóricas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('bin', 'passthrough', binary_features) # Binary features don't need scaling or encoding
        ],
        remainder='drop' # Drop any other columns not specified
    )

    # Cria um pipeline com pré-processador e SMOTE
    # SMOTE deve ser aplicado APÓS o pré-processamento para evitar problemas
    # e ANTES do treinamento do modelo.
    # No entanto, para SMOTE com ColumnTransformer, é mais fácil aplicar SMOTE
    # após o fit_transform do preprocessor.
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # SMOTE para balancear
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_processed, y_train)
    
    # Treina XGBoost com scale_pos_weight para fraudes raras
    scale_pos_weight = len(y_train_bal[y_train_bal == 0]) / len(y_train_bal[y_train_bal == 1]) if len(y_train_bal[y_train_bal == 1]) > 0 else 1
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss'
    )
    model.fit(X_train_bal, y_train_bal)
    
    # Predições
    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]
    
    # Métricas
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Cross-validation (aplicar preprocessor a X completo antes do CV)
    X_processed_full = preprocessor.transform(X)
    cv_scores = cross_val_score(model, X_processed_full, y, cv=5, scoring='f1')
    
    # Adiciona predições ao DF original
    df['predicted_fraud'] = model.predict(X_processed_full)
    df['proba_fraud'] = model.predict_proba(X_processed_full)[:, 1]
    
    # Risk score aprimorado (threshold para alertas) - Combina probabilidade com bônus por features de risco
    proba = df['proba_fraud'].values
    risk_score_base = proba * 100
    
    # Bônus por features de risco (ajustados para reduzir falsos positivos: pesos moderados)
    # Aumentar a influência de novas features de risco
    risk_score_bonus = (
        df['desvio_valor'] * 15 +  # Grande desvio de valor é forte indicador
        df['score_valor_z'].apply(lambda x: abs(x) * 5 if abs(x) > 2 else 0) + # Z-score alto indica anomalia
        np.where(df['hora'] > 22, 8, 0) +  # Horário de risco (noite)
        np.where(df['hora'] < 6, 8, 0) +   # Horário de risco (madrugada)
        df['dispositivo_novo'] * 12 +
        df['ip_novo'] * 10 +
        df['mudanca_localizacao'] * 18 +
        df['tentativas_senha'] * 7 +
        df['destinatario_novo'] * 5 +
        df['fraudes_passadas'] * 10 +  # Histórico aumenta risco
        df['distancia_ultima_transacao'].apply(lambda x: 10 if x > 500 else 0) + # Grande distância
        df['e_vpn'] * 15 + # Uso de VPN
        df['senha_alterada_recentemente'] * 5 + # Senha alterada pode ser sinal de comprometimento
        df['velocidade_transacoes'].apply(lambda x: 10 if x > 3 else 0) # Velocidade alta
    )
    
    df['risk_score'] = np.clip(risk_score_base + risk_score_bonus, 0, 100)
    
    # Salva o pipeline completo (preprocessor + model)
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', model)])
    joblib.dump(full_pipeline, 'fraud_model_pipeline.pkl')
    
    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred))
    
    return df, {'precision': precision, 'recall': recall, 'f1': f1, 'cv_f1': cv_scores.mean()}

def predict_transaction(new_data, threshold=75):
    try:
        full_pipeline = joblib.load('fraud_model_pipeline.pkl')
        preprocessor = full_pipeline.named_steps['preprocessor']
        model = full_pipeline.named_steps['classifier']
    except FileNotFoundError:
        return {'error': 'Modelo não treinado. Treine primeiro no dashboard.'}
    
    # Features para o modelo (devem ser as mesmas usadas no treinamento)
    numerical_features = [
        'hora', 'valor', 'tentativas_senha', 'score_credito',
        'desvio_valor', 'media_valor_user', 'fraudes_passadas',
        'limite_cartao', 'saldo_conta', 'distancia_ultima_transacao',
        'freq_transacao_destinatario', 'valor_total_enviado_destinatario',
        'velocidade_transacoes', 'intervalo_medio_transacoes', 'score_valor_z',
        'dia_semana'
    ]
    categorical_features = [
        'tipo_transacao', 'periodo_dia', 'tipo_dispositivo', 'os', 'tipo_ip'
    ]
    binary_features = [
        'dispositivo_novo', 'ip_novo', 'mudanca_localizacao', 'destinatario_novo',
        'fim_semana', 'e_vpn', 'senha_alterada_recentemente'
    ]
    all_features = numerical_features + categorical_features + binary_features

    # Cria um DataFrame a partir de new_data, preenchendo defaults se faltar
    new_data_df = pd.DataFrame([new_data])
    for feature in all_features:
        if feature not in new_data_df.columns:
            if feature in numerical_features:
                new_data_df[feature] = 0
            elif feature in categorical_features:
                new_data_df[feature] = 'unknown'
            elif feature in binary_features:
                new_data_df[feature] = 0
    
    # Pré-processa os novos dados
    X_new_processed = preprocessor.transform(new_data_df[all_features])
    
    pred = model.predict(X_new_processed)[0]
    proba = model.predict_proba(X_new_processed)[0, 1]
    
    # Recalcula o risk_score para a nova transação usando a mesma lógica aprimorada
    risk_score_base = proba * 100
    risk_score_bonus = (
        new_data['desvio_valor'] * 15 +
        (abs(new_data['score_valor_z']) * 5 if abs(new_data['score_valor_z']) > 2 else 0) +
        (8 if new_data['hora'] > 22 else 0) +
        (8 if new_data['hora'] < 6 else 0) +
        new_data['dispositivo_novo'] * 12 +
        new_data['ip_novo'] * 10 +
        new_data['mudanca_localizacao'] * 18 +
        new_data['tentativas_senha'] * 7 +
        new_data['destinatario_novo'] * 5 +
        new_data['fraudes_passadas'] * 10 +
        (10 if new_data['distancia_ultima_transacao'] > 500 else 0) +
        new_data['e_vpn'] * 15 +
        new_data['senha_alterada_recentemente'] * 5 +
        (10 if new_data['velocidade_transacoes'] > 3 else 0)
    )
    
    risk = np.clip(risk_score_base + risk_score_bonus, 0, 100)
    
    alert = 'BLOQUEIO AUTOMÁTICO' if risk > threshold else 'APROVADO'
    
    return {
        'predicted_fraud': bool(pred),
        'risk_score': round(risk, 2),
        'alert': alert,
        'proba': round(proba, 3)
    }

if __name__ == "__main__":
    df = pd.read_csv('transactions.csv')
    df_trained, metrics = train_model(df)
    print(f"Métricas: {metrics}")

