import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_transactions(n_transactions=2000):
    np.random.seed(42)
    random.seed(42)
    
    # Gera user_ids
    user_ids = np.random.randint(1, 1001, n_transactions)
    
    # Timestamps recentes
    base_date = datetime.now() - timedelta(days=30)
    timestamps = [base_date + timedelta(minutes=random.randint(0, 30*24*60)) for _ in range(n_transactions)]
    
    # Horas
    horas = [ts.hour for ts in timestamps]
    
    # Valores normais vs fraudes
    valores_normais = np.random.normal(150, 50, n_transactions)
    valores_fraude = np.random.exponential(500, n_transactions)
    valores = []
    is_fraud = []
    
    for i in range(n_transactions):
        if random.random() < 0.05:  # 5% fraudes
            valores.append(valores_fraude[i])
            is_fraud.append(1)
        else:
            valores.append(max(10, valores_normais[i]))
            is_fraud.append(0)
    
    # Features binárias/categóricas existentes
    dispositivo_novo = np.random.binomial(1, 0.1, n_transactions)  # 10% novo dispositivo
    ip_novo = np.random.binomial(1, 0.15, n_transactions)
    tentativas_senha = np.random.poisson(0.5, n_transactions)
    mudanca_localizacao = np.random.binomial(1, 0.08, n_transactions)
    destinatario_novo = np.random.binomial(1, 0.2, n_transactions)
    fim_semana = np.array([1 if h >= 18 or h < 8 or (h >= 0 and ts.weekday() >= 5) else 0 for h, ts in zip(horas, timestamps)])
    score_credito = np.random.normal(700, 100, n_transactions).astype(int)
    tipos = np.random.choice(['PIX', 'TED', 'DOC', 'Cartão'], n_transactions)
    localizacoes = np.random.choice(['São Paulo', 'Rio de Janeiro', 'Brasília', 'Curitiba', 'Desconhecida'], n_transactions)
    descricoes = [f"Transferência para {random.choice(['João', 'Maria', 'Empresa XYZ', 'Conta Desconhecida'])}" for _ in range(n_transactions)]

    # --- NOVAS FEATURES SINTÉTICAS ---

    # HORÁRIO
    dia_semana = [ts.weekday() for ts in timestamps] # 0=Segunda, 6=Domingo
    periodo_dia = []
    for h in horas:
        if 0 <= h < 6: periodo_dia.append('madrugada')
        elif 6 <= h < 12: periodo_dia.append('manha')
        elif 12 <= h < 18: periodo_dia.append('tarde')
        else: periodo_dia.append('noite')

    # PERFIL FINANCEIRO (Simulado)
    limite_cartao = np.random.normal(2000, 1000, n_transactions).clip(100, 10000).astype(int)
    saldo_conta = np.random.normal(1500, 700, n_transactions).clip(0, 50000).astype(int)

    # LOCALIZAÇÃO
    # Simular distância da última transação (apenas para fins de demonstração)
    distancia_ultima_transacao = np.random.exponential(50, n_transactions).clip(0, 1000).astype(int)
    # Aumentar chance de distância alta para fraudes
    for i in range(n_transactions):
        if is_fraud[i] == 1 and random.random() < 0.3: # 30% das fraudes tem distancia alta
            distancia_ultima_transacao[i] = np.random.randint(500, 5000)

    # DISPOSITIVO
    tipo_dispositivo = np.random.choice(['mobile', 'desktop'], n_transactions, p=[0.7, 0.3])
    os_choices = ['Android', 'iOS', 'Windows', 'MacOS']
    os = np.random.choice(os_choices, n_transactions, p=[0.4, 0.3, 0.2, 0.1])
    tipo_ip = np.random.choice(['residencial', 'corporativo', 'vpn'], n_transactions, p=[0.7, 0.2, 0.1])
    e_vpn = np.random.binomial(1, 0.05, n_transactions) # 5% usam VPN

    # AUTENTIFICAÇÃO
    senha_alterada_recentemente = np.random.binomial(1, 0.2, n_transactions) # 20% alteraram senha recentemente

    # Cálculos derivados por user (existente e expandido)
    media_valor_user = []
    desvio_valor = []
    fraudes_passadas = []
    score_valor_z = [] # Novo: Z-score do valor da transação vs histórico do user
    
    # Para simular frequência e valor total por destinatário
    user_dest_history = {} # {user_id: {destinatario: {'count': N, 'total_value': V, 'last_ts': TS}}}

    for uid in set(user_ids):
        user_mask = user_ids == uid
        user_vals = np.array(valores)[user_mask]
        user_fraudes = sum(np.array(is_fraud)[user_mask])
        
        # Para Z-score
        user_mean_val = np.mean(user_vals) if len(user_vals) > 0 else 0
        user_std_val = np.std(user_vals) if len(user_vals) > 0 else 1 # Evitar divisão por zero

        # Para destinatários
        user_dest_history[uid] = {}

        for i in np.where(user_mask)[0]: # Iterar sobre os índices das transações do usuário
            # Existing derived features
            media_valor_user.append(user_mean_val)
            desvio_valor.append(abs(valores[i] - user_mean_val) / user_mean_val if user_mean_val > 0 else 0)
            fraudes_passadas.append(user_fraudes)

            # New derived feature: Z-score
            score_valor_z.append((valores[i] - user_mean_val) / user_std_val if user_std_val > 0 else 0)

            # Simular histórico de destinatário
            dest = descricoes[i].split(' para ')[-1] # Extrai o destinatário da descrição
            if dest not in user_dest_history[uid]:
                user_dest_history[uid][dest] = {'count': 0, 'total_value': 0, 'last_ts': timestamps[i]}
            
            user_dest_history[uid][dest]['count'] += 1
            user_dest_history[uid][dest]['total_value'] += valores[i]
            user_dest_history[uid][dest]['last_ts'] = timestamps[i]

    # Adicionar features de destinatário ao DF
    freq_transacao_destinatario = []
    valor_total_enviado_destinatario = []
    
    for i in range(n_transactions):
        uid = user_ids[i]
        dest = descricoes[i].split(' para ')[-1]
        
        if dest in user_dest_history[uid]:
            freq_transacao_destinatario.append(user_dest_history[uid][dest]['count'])
            valor_total_enviado_destinatario.append(user_dest_history[uid][dest]['total_value'])
        else: # Caso não encontre (deve ser raro se a lógica acima estiver correta)
            freq_transacao_destinatario.append(0)
            valor_total_enviado_destinatario.append(0)

    # Simular velocidade e intervalo médio (muito simplificado)
    velocidade_transacoes = np.random.exponential(0.5, n_transactions).clip(0.1, 5) # Transações por minuto
    intervalo_medio_transacoes = np.random.exponential(60, n_transactions).clip(1, 1440) # Minutos

    df = pd.DataFrame({
        'id': range(1, n_transactions + 1),
        'user_id': user_ids,
        'timestamp': [ts.isoformat() for ts in timestamps],
        'hora': horas,
        'valor': valores,
        'tipo_transacao': tipos,
        'dispositivo_novo': dispositivo_novo,
        'ip_novo': ip_novo,
        'tentativas_senha': tentativas_senha,
        'mudanca_localizacao': mudanca_localizacao,
        'destinatario_novo': destinatario_novo,
        'fim_semana': fim_semana,
        'score_credito': score_credito,
        'is_fraud': is_fraud,
        'localizacao': localizacoes,
        'descricao_transacao': descricoes,
        'media_valor_user': media_valor_user,
        'desvio_valor': desvio_valor,
        'fraudes_passadas': fraudes_passadas,
        
        # NOVAS FEATURES
        'dia_semana': dia_semana,
        'periodo_dia': periodo_dia,
        'limite_cartao': limite_cartao,
        'saldo_conta': saldo_conta,
        'distancia_ultima_transacao': distancia_ultima_transacao,
        'tipo_dispositivo': tipo_dispositivo,
        'os': os,
        'tipo_ip': tipo_ip,
        'e_vpn': e_vpn,
        'senha_alterada_recentemente': senha_alterada_recentemente,
        'freq_transacao_destinatario': freq_transacao_destinatario,
        'valor_total_enviado_destinatario': valor_total_enviado_destinatario,
        'velocidade_transacoes': velocidade_transacoes,
        'intervalo_medio_transacoes': intervalo_medio_transacoes,
        'score_valor_z': score_valor_z
    })
    
    # Ajusta fraudes_passadas para ser cumulativo (simples)
    for uid in set(user_ids):
        user_mask = df['user_id'] == uid
        df.loc[user_mask, 'fraudes_passadas'] = sum(df.loc[user_mask, 'is_fraud'])
    
    df = df.round(2)
    return df

if __name__ == "__main__":
    df = generate_transactions(2000)
    df.to_csv('transactions.csv', index=False)
    print(f"Dataset gerado: {len(df)} transações, {df['is_fraud'].sum()} fraudes (5%).")
    print("Novas colunas geradas:", [col for col in df.columns if col not in ['id', 'user_id', 'timestamp', 'hora', 'valor', 'tipo_transacao', 'dispositivo_novo', 'ip_novo', 'tentativas_senha', 'mudanca_localizacao', 'destinatario_novo', 'fim_semana', 'score_credito', 'is_fraud', 'localizacao', 'descricao_transacao', 'media_valor_user', 'desvio_valor', 'fraudes_passadas']])

