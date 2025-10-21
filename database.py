import sqlite3
import pandas as pd

DB_FILE = 'arion.db'

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Tabela users (inalterada)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            media_valor REAL DEFAULT 0,
            fraudes_passadas INTEGER DEFAULT 0
        )
    ''')
    
    # Tabela transactions (EXPANDIDA com novas colunas)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            timestamp TEXT,
            hora INTEGER,
            valor REAL,
            tipo_transacao TEXT,
            dispositivo_novo INTEGER,
            ip_novo INTEGER,
            tentativas_senha INTEGER,
            mudanca_localizacao INTEGER,
            destinatario_novo INTEGER,
            fim_semana INTEGER,
            score_credito INTEGER,
            is_fraud INTEGER,
            risk_score REAL,
            localizacao TEXT,
            descricao_transacao TEXT,
            media_valor_user REAL,
            desvio_valor REAL,
            predicted_fraud INTEGER,
            proba_fraud REAL,
            
            -- NOVAS COLUNAS
            dia_semana INTEGER,
            periodo_dia TEXT,
            limite_cartao INTEGER,
            saldo_conta INTEGER,
            distancia_ultima_transacao INTEGER,
            tipo_dispositivo TEXT,
            os TEXT,
            tipo_ip TEXT,
            e_vpn INTEGER,
            senha_alterada_recentemente INTEGER,
            freq_transacao_destinatario INTEGER,
            valor_total_enviado_destinatario REAL,
            velocidade_transacoes REAL,
            intervalo_medio_transacoes REAL,
            score_valor_z REAL,

            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    conn.commit()
    conn.close()

def insert_data(df):
    conn = sqlite3.connect(DB_FILE)
    
    # Inicializa fraudes_passadas se não existir (garantia)
    if 'fraudes_passadas' not in df.columns:
        df['fraudes_passadas'] = 0
    
    # Upsert para users
    cursor = conn.cursor()
    users_df = df[['user_id', 'media_valor_user', 'fraudes_passadas']].drop_duplicates().rename(columns={'media_valor_user': 'media_valor'})
    for index, row in users_df.iterrows():
        cursor.execute('''
            INSERT INTO users (user_id, media_valor, fraudes_passadas)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                media_valor = EXCLUDED.media_valor,
                fraudes_passadas = EXCLUDED.fraudes_passadas
        ''', (int(row['user_id']), row['media_valor'], int(row['fraudes_passadas'])))
    conn.commit()
    
    # Insere/atualiza transactions (usa replace para simplicidade; em produção, use upsert por ID)
    # Certifique-se de que o DataFrame 'df' contém todas as colunas da tabela 'transactions'
    # Se o df não tiver 'predicted_fraud', 'proba_fraud', 'risk_score', adicione-os com valores padrão
    for col in ['predicted_fraud', 'proba_fraud', 'risk_score']:
        if col not in df.columns:
            df[col] = None # Ou 0, dependendo do tipo e do default desejado

    df.to_sql('transactions', conn, if_exists='replace', index=False)
    
    # Atualiza fraudes_passadas em users baseado nas transações
    fraud_counts = df[df['is_fraud'] == 1]['user_id'].value_counts().to_dict()
    for user_id, count in fraud_counts.items():
        cursor.execute("UPDATE users SET fraudes_passadas = ? WHERE user_id = ?", (int(count), int(user_id)))
    conn.commit()
    
    conn.close()

def get_data():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("""
        SELECT t.*, u.fraudes_passadas 
        FROM transactions t 
        LEFT JOIN users u ON t.user_id = u.user_id
    """, conn)
    conn.close()
    return df

def update_transaction_status(transaction_id, user_id, is_fraud_status):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Atualiza is_fraud da transação (insere se não existir, para demo)
    # Nota: Para transações simuladas, o ID pode não existir no DB.
    # A lógica de INSERT OR REPLACE é para garantir que a transação seja registrada
    # mesmo que seja uma simulação que não veio do dataset original.
    # Em um sistema real, você buscaria a transação pelo ID e a atualizaria.
    cursor.execute("""
        INSERT OR REPLACE INTO transactions 
        (id, user_id, is_fraud, timestamp, hora, valor, tipo_transacao, dispositivo_novo, ip_novo, tentativas_senha, mudanca_localizacao, destinatario_novo, fim_semana, score_credito, localizacao, descricao_transacao, media_valor_user, desvio_valor, fraudes_passadas, predicted_fraud, proba_fraud, risk_score, dia_semana, periodo_dia, limite_cartao, saldo_conta, distancia_ultima_transacao, tipo_dispositivo, os, tipo_ip, e_vpn, senha_alterada_recentemente, freq_transacao_destinatario, valor_total_enviado_destinatario, velocidade_transacoes, intervalo_medio_transacoes, score_valor_z)
        VALUES (?, ?, ?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (int(transaction_id), int(user_id), int(is_fraud_status), 0, 0.0, 'N/A', 0, 0, 0, 0, 0, 0, 0, 'N/A', 'N/A', 0.0, 0.0, 0, 0, 0.0, 0.0, 0, 'N/A', 0, 0, 0, 'N/A', 'N/A', 'N/A', 0, 0, 0, 0.0, 0.0, 0.0, 0.0))
    # O timestamp, hora, valor, etc. são preenchidos com valores padrão para INSERT OR REPLACE.
    # Em um sistema real, você passaria todos os dados da transação simulada para esta função.
    
    # Recalcula fraudes_passadas para o usuário
    cursor.execute("SELECT COUNT(*) FROM transactions WHERE user_id = ? AND is_fraud = 1", (int(user_id),))
    new_fraudes_count = cursor.fetchone()[0]
    
    cursor.execute("UPDATE users SET fraudes_passadas = ? WHERE user_id = ?", (int(new_fraudes_count), int(user_id)))
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("DB configurado!")
