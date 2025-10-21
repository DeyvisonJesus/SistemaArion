import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
import traceback
from data_generator import generate_transactions
from database import init_db, insert_data, get_data, update_transaction_status
from model import train_model, predict_transaction

st.set_page_config(page_title="Arion - Detecção de Fraudes", layout="wide")
st.title("🛡️ Arion - Detecção de Fraudes Financeiras")
st.markdown("**XGBoost para Precisão Alta, Menos Falsos Positivos e Controle Total via Dashboard**")

# Sidebar: Configurações Globais
st.sidebar.header("⚙️ Configurações")
n_transactions = st.sidebar.slider("Número de Transações para Gerar", 500, 5000, 2000)
threshold_risk = st.sidebar.slider("Threshold para Alerta (reduz falsos positivos)", 50, 95, 75)
export_format = st.sidebar.selectbox("Exportar Relatório", ["CSV", "JSON"])

# Estado da sessão para cache
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'last_prediction_data' not in st.session_state:
    st.session_state.last_prediction_data = None
if 'simulation_preset' not in st.session_state:
    st.session_state.simulation_preset = "Personalizado"

@st.cache_data(ttl=60)
def load_data_cached():
    try:
        if os.path.exists('transactions_predicted.csv'):
            df = pd.read_csv('transactions_predicted.csv')
            return df
        elif os.path.exists('transactions.csv'):
            df = pd.read_csv('transactions.csv')
            return df
        elif os.path.exists('arion.db'):
            df = get_data()
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"[ERRO INTERNO] load_data_cached: {str(e)}")
        return pd.DataFrame()

# Seção 1: Setup e Controle
st.header("🔧 Setup e Treinamento")
col1, col2, col3, col4 = st.columns(4)

status_placeholder = st.empty()

with col1:
    if st.button("📊 Gerar Dados", use_container_width=True):
        try:
            with st.spinner("Gerando transações..."):
                df = generate_transactions(n_transactions)
                df.to_csv('transactions.csv', index=False)
                st.session_state.data_loaded = True
                status_placeholder.success(f"✅ Dataset gerado: {n_transactions} transações (5% fraudes simuladas). Arquivo: transactions.csv")
                st.cache_data.clear()
                time.sleep(2)
        except Exception as e:
            status_placeholder.error(f"❌ Erro ao gerar dados: {str(e)}")
            st.error(traceback.format_exc())

with col2:
    if st.button("🗄️ Configurar Banco", use_container_width=True):
        try:
            if os.path.exists('transactions.csv'):
                with st.spinner("Configurando DB..."):
                    init_db()
                    df = pd.read_csv('transactions.csv')
                    insert_data(df)
                    st.session_state.data_loaded = True
                    status_placeholder.success("✅ Banco configurado com histórico de users! Arquivo: arion.db")
                    st.cache_data.clear()
                    time.sleep(2)
            else:
                status_placeholder.warning("⚠️ Gere dados primeiro!")
        except Exception as e:
            status_placeholder.error(f"❌ Erro ao configurar banco: {str(e)}")
            st.error(traceback.format_exc())

with col3:
    if st.button("🤖 Treinar Modelo", use_container_width=True):
        try:
            if os.path.exists('transactions.csv'):
                with st.spinner("Treinando XGBoost com SMOTE..."):
                    progress_bar = st.progress(0)
                    df = pd.read_csv('transactions.csv')
                    df_trained, metrics = train_model(df)
                    df_trained.to_csv('transactions_predicted.csv', index=False)
                    
                    if os.path.exists('arion.db'):
                        insert_data(df_trained)
                    
                    st.session_state.metrics = metrics
                    st.session_state.model_trained = True
                    progress_bar.progress(100)
                    status_placeholder.success(f"✅ Modelo treinado! Precisão: {metrics['precision']:.2f}, F1: {metrics['f1']:.2f}. Arquivo: transactions_predicted.csv")
                    st.cache_data.clear()
                    time.sleep(2)
            else:
                status_placeholder.warning("⚠️ Gere dados primeiro!")
        except Exception as e:
            status_placeholder.error(f"❌ Erro ao treinar modelo: {str(e)}")
            st.error(traceback.format_exc())

with col4:
    if st.button("🧹 Limpar Tudo", use_container_width=True):
        try:
            for file in ['transactions.csv', 'transactions_predicted.csv', 'arion.db', 'fraud_model_pipeline.pkl']:
                if os.path.exists(file):
                    os.remove(file)
            st.session_state.data_loaded = False
            st.session_state.model_trained = False
            st.session_state.metrics = {}
            st.session_state.last_prediction_data = None
            st.cache_data.clear()
            status_placeholder.success("✅ Tudo limpo! Reinicie o setup.")
            time.sleep(2)
        except Exception as e:
            status_placeholder.error(f"❌ Erro ao limpar: {str(e)}")
            st.error(traceback.format_exc())

# Status do sistema
st.markdown("**Status do Sistema:**")
col_status1, col_status2, col_status3 = st.columns(3)
with col_status1:
    status_data = "✅ Dados Gerados" if st.session_state.data_loaded or os.path.exists('transactions.csv') else "❌ Sem Dados"
    st.metric("Dados", status_data)
with col_status2:
    status_db = "✅ Banco Configurado" if os.path.exists('arion.db') else "❌ Banco Não Configurado"
    st.metric("Banco", status_db)
with col_status3:
    status_model = "✅ Modelo Treinado" if st.session_state.model_trained or os.path.exists('fraud_model_pipeline.pkl') else "❌ Modelo Não Treinado"
    st.metric("Modelo", status_model)

# Seção 2: Monitoramento
st.header("📊 Dashboard de Monitoramento")

df = load_data_cached()

if not df.empty or st.session_state.data_loaded:
    st.success("✅ Dashboard carregado! Dados disponíveis.")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_transactions = len(df)
        st.metric("📈 Total de Transações", f"{total_transactions:,}")
    
    with col2:
        if 'risk_score' in df.columns:
            high_risk = len(df[df['risk_score'] > threshold_risk])
            st.metric("🚨 Alto Risco", f"{high_risk:,}")
        else:
            st.metric("🚨 Alto Risco", "N/A (Treine o modelo)")
    
    with col3:
        fraud_detected = len(df[df['is_fraud'] == 1]) if 'is_fraud' in df.columns else 0
        st.metric("⚠️ Fraudes Detectadas", f"{fraud_detected:,}")
    
    with col4:
        volume_total = df['valor'].sum() if 'valor' in df.columns else 0
        st.metric("💰 Volume Total", f"R$ {volume_total:,.2f}")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Tempo Real", "📊 Analytics", "🔮 Simulador"])
    
    with tab1:
        st.subheader("Monitoramento em Tempo Real")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if 'hora' in df.columns:
                hourly_data = df.groupby('hora').agg({
                    'id': 'count',
                    'is_fraud': 'sum'
                }).reset_index()
                hourly_data.columns = ['hora', 'total_transacoes', 'fraudes']
                
                fig = make_subplots(rows=1, cols=1, subplot_titles=('Transações e Fraudes por Hora',))
                fig.add_trace(go.Bar(x=hourly_data['hora'], y=hourly_data['total_transacoes'], name='Total Transações', marker_color='lightblue'))
                fig.add_trace(go.Bar(x=hourly_data['hora'], y=hourly_data['fraudes'], name='Fraudes', marker_color='red'))
                fig.update_layout(height=400, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Coluna 'hora' não disponível. Gere dados novamente.")
        
        with col2:
            st.subheader("Últimas Transações")
            cols_to_show = ['id', 'user_id', 'valor', 'hora', 'tipo_transacao'] if all(col in df.columns for col in ['id', 'user_id', 'valor', 'hora', 'tipo_transacao']) else df.columns[:5].tolist()
            if 'risk_score' in df.columns:
                cols_to_show.append('risk_score')
            recent_transactions = df.tail(10)[cols_to_show]
            st.dataframe(recent_transactions, use_container_width=True)
    
    with tab2:
        st.subheader("Analytics Avançadas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'valor' in df.columns:
                fig = px.histogram(df, x='valor', nbins=30, title='Distribuição de Valores das Transações')
                st.plotly_chart(fig, use_container_width=True)
            
            if 'tipo_transacao' in df.columns:
                tipo_counts = df['tipo_transacao'].value_counts()
                fig = px.pie(values=tipo_counts.values, names=tipo_counts.index, title='Transações por Tipo')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'risk_score' in df.columns:
                fig = px.histogram(df, x='risk_score', nbins=20, title='Distribuição de Risk Scores')
                fig.add_vline(x=threshold_risk, line_dash="dash", line_color="red", annotation_text="Limite de Alerta")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Treine o modelo para ver risk scores")
            
            if 'is_fraud' in df.columns and 'hora' in df.columns:
                fraud_hourly = df[df['is_fraud'] == 1]['hora'].value_counts().sort_index()
                if not fraud_hourly.empty:
                    fig_fraudes_hora = px.bar(
                        x=fraud_hourly.index,
                        y=fraud_hourly.values,
                        labels={'x': 'Hora do Dia', 'y': 'Número de Fraudes'},
                        title='Fraudes Detectadas por Hora do Dia',
                        text=fraud_hourly.values
                    )
                    fig_fraudes_hora.update_traces(textposition='outside')
                    fig_fraudes_hora.update_layout(xaxis=dict(tickmode='linear'))
                    st.plotly_chart(fig_fraudes_hora, use_container_width=True)

                total_hourly = df['hora'].value_counts().sort_index()
                fraud_ratio = (fraud_hourly / total_hourly).fillna(0)
                fig_fraude_ratio = px.bar(
                    x=fraud_ratio.index,
                    y=fraud_ratio.values,
                    labels={'x': 'Hora do Dia', 'y': 'Proporção de Fraudes'},
                    title='Proporção de Fraudes por Hora do Dia',
                    text=[f"{v:.1%}" for v in fraud_ratio.values]
                )
                fig_fraude_ratio.update_traces(textposition='outside')
                fig_fraude_ratio.update_layout(xaxis=dict(tickmode='linear'), yaxis=dict(range=[0, 1]))
                st.plotly_chart(fig_fraude_ratio, use_container_width=True)
        
        if st.session_state.model_trained and st.session_state.metrics:
            st.subheader("📈 Performance do Modelo")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Precisão", f"{st.session_state.metrics['precision']:.2%}")
            with col_m2:
                st.metric("Recall", f"{st.session_state.metrics['recall']:.2%}")
            with col_m3:
                st.metric("F1-Score", f"{st.session_state.metrics['f1']:.2%}")
            with col_m4:
                st.metric("CV F1", f"{st.session_state.metrics['cv_f1']:.2%}")
    
    with tab3:
        st.subheader("🔮 Simulador de Transação")
        
        if st.session_state.model_trained:
            preset_options = ["Personalizado", "Transação Normal", "Médio Risco", "Fraude Óbvia", "Fraude de Dispositivo"]
            selected_preset = st.selectbox("Escolha um Preset de Transação", preset_options, key="preset_selector")

            presets = {
                "Personalizado": {
                    'user_id': 123, 'valor': 250.0, 'tipo_transacao': 'PIX', 'hora': 14,
                    'dia_semana': 2, 'periodo_dia': 'tarde', 'score_credito': 650,
                    'limite_cartao': 2000, 'saldo_conta': 1500, 'dispositivo_novo': False,
                    'ip_novo': False, 'mudanca_localizacao': False, 'e_vpn': False,
                    'destinatario_novo': False, 'fim_semana': False, 'tentativas_senha': 0,
                    'senha_alterada_recentemente': False, 'tipo_dispositivo': 'mobile',
                    'os_sim': 'Android', 'tipo_ip': 'residencial', 'distancia_ultima_transacao': 50,
                    'media_valor_user': 200.0, 'fraudes_passadas': 0,
                    'freq_transacao_destinatario': 1, 'valor_total_enviado_destinatario': 100.0,
                    'velocidade_transacoes': 0.5, 'intervalo_medio_transacoes': 60.0, 'score_valor_z': 0.0
                },
                "Transação Normal": {
                    'user_id': 101, 'valor': 120.0, 'tipo_transacao': 'PIX', 'hora': 10,
                    'dia_semana': 1, 'periodo_dia': 'manha', 'score_credito': 750,
                    'limite_cartao': 3000, 'saldo_conta': 5000, 'dispositivo_novo': False,
                    'ip_novo': False, 'mudanca_localizacao': False, 'e_vpn': False,
                    'destinatario_novo': False, 'fim_semana': False, 'tentativas_senha': 0,
                    'senha_alterada_recentemente': False, 'tipo_dispositivo': 'mobile',
                    'os_sim': 'Android', 'tipo_ip': 'residencial', 'distancia_ultima_transacao': 10,
                    'media_valor_user': 100.0, 'fraudes_passadas': 0,
                    'freq_transacao_destinatario': 5, 'valor_total_enviado_destinatario': 500.0,
                    'velocidade_transacoes': 0.2, 'intervalo_medio_transacoes': 120.0, 'score_valor_z': 0.1
                },
                "Médio Risco": {
                    'user_id': 205, 'valor': 800.0, 'tipo_transacao': 'TED', 'hora': 23,
                    'dia_semana': 5, 'periodo_dia': 'noite', 'score_credito': 600,
                    'limite_cartao': 1500, 'saldo_conta': 800, 'dispositivo_novo': True,
                    'ip_novo': True, 'mudanca_localizacao': False, 'e_vpn': False,
                    'destinatario_novo': True, 'fim_semana': True, 'tentativas_senha': 1,
                    'senha_alterada_recentemente': False, 'tipo_dispositivo': 'desktop',
                    'os_sim': 'Windows', 'tipo_ip': 'corporativo', 'distancia_ultima_transacao': 150,
                    'media_valor_user': 150.0, 'fraudes_passadas': 0,
                    'freq_transacao_destinatario': 0, 'valor_total_enviado_destinatario': 0.0,
                    'velocidade_transacoes': 1.0, 'intervalo_medio_transacoes': 30.0, 'score_valor_z': 1.5
                },
                "Fraude Óbvia": {
                    'user_id': 310, 'valor': 4500.0, 'tipo_transacao': 'PIX', 'hora': 3,
                    'dia_semana': 6, 'periodo_dia': 'madrugada', 'score_credito': 450,
                    'limite_cartao': 1000, 'saldo_conta': 200, 'dispositivo_novo': True,
                    'ip_novo': True, 'mudanca_localizacao': True, 'e_vpn': True,
                    'destinatario_novo': True, 'fim_semana': True, 'tentativas_senha': 3,
                    'senha_alterada_recentemente': True, 'tipo_dispositivo': 'mobile',
                    'os_sim': 'Android', 'tipo_ip': 'vpn', 'distancia_ultima_transacao': 1200,
                    'media_valor_user': 80.0, 'fraudes_passadas': 2,
                    'freq_transacao_destinatario': 0, 'valor_total_enviado_destinatario': 0.0,
                    'velocidade_transacoes': 4.0, 'intervalo_medio_transacoes': 5.0, 'score_valor_z': 4.0
                },
                "Fraude de Dispositivo": {
                    'user_id': 400, 'valor': 300.0, 'tipo_transacao': 'Cartão', 'hora': 19,
                    'dia_semana': 3, 'periodo_dia': 'noite', 'score_credito': 700,
                    'limite_cartao': 5000, 'saldo_conta': 3000, 'dispositivo_novo': True,
                    'ip_novo': True, 'mudanca_localizacao': True, 'e_vpn': False,
                    'destinatario_novo': False, 'fim_semana': False, 'tentativas_senha': 0,
                    'senha_alterada_recentemente': False, 'tipo_dispositivo': 'desktop',
                    'os_sim': 'MacOS', 'tipo_ip': 'residencial', 'distancia_ultima_transacao': 800,
                    'media_valor_user': 280.0, 'fraudes_passadas': 0,
                    'freq_transacao_destinatario': 3, 'valor_total_enviado_destinatario': 900.0,
                    'velocidade_transacoes': 0.8, 'intervalo_medio_transacoes': 90.0, 'score_valor_z': 0.5
                }
            }

            current_preset_values = presets[selected_preset]

            for key, value in current_preset_values.items():
                if key != 'valor':
                    st.session_state[f'sim_{key}'] = value

            preset_valor = current_preset_values['valor']

            with st.form("simulation_form"):
                st.markdown("---")
                st.subheader("Dados Básicos da Transação")
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    user_id = st.number_input("ID do Usuário", 1, 1000, value=st.session_state.get('sim_user_id', current_preset_values['user_id']))
                    valor = st.number_input("Valor (R$)", 0.0, 10000.0, value=preset_valor)
                    tipo_transacao = st.selectbox("Tipo de Transação", ['PIX', 'TED', 'DOC', 'Cartão'], index=['PIX', 'TED', 'DOC', 'Cartão'].index(st.session_state.get('sim_tipo_transacao', current_preset_values['tipo_transacao'])))
                with col_s2:
                    hora = st.slider("Hora da Transação", 0, 23, value=st.session_state.get('sim_hora', current_preset_values['hora']))
                    dia_semana = st.selectbox("Dia da Semana", options=list(range(7)), format_func=lambda x: ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom'][x], index=st.session_state.get('sim_dia_semana', current_preset_values['dia_semana']))
                    periodo_dia = st.selectbox("Período do Dia", ['madrugada', 'manha', 'tarde', 'noite'], index=['madrugada', 'manha', 'tarde', 'noite'].index(st.session_state.get('sim_periodo_dia', current_preset_values['periodo_dia'])))
                with col_s3:
                    score_credito = st.slider("Score de Crédito", 300, 900, value=st.session_state.get('sim_score_credito', current_preset_values['score_credito']))
                    limite_cartao = st.number_input("Limite do Cartão (R$)", 0, 10000, value=st.session_state.get('sim_limite_cartao', current_preset_values['limite_cartao']))
                    saldo_conta = st.number_input("Saldo da Conta (R$)", 0, 50000, value=st.session_state.get('sim_saldo_conta', current_preset_values['saldo_conta']))

                st.markdown("---")
                st.subheader("Indicadores de Comportamento e Dispositivo")
                col_s4, col_s5, col_s6 = st.columns(3)
                with col_s4:
                    dispositivo_novo = st.checkbox("Dispositivo Novo", value=st.session_state.get('sim_dispositivo_novo', current_preset_values['dispositivo_novo']))
                    ip_novo = st.checkbox("IP Novo", value=st.session_state.get('sim_ip_novo', current_preset_values['ip_novo']))
                    mudanca_localizacao = st.checkbox("Mudança de Localização", value=st.session_state.get('sim_mudanca_localizacao', current_preset_values['mudanca_localizacao']))
                    e_vpn = st.checkbox("É VPN", value=st.session_state.get('sim_e_vpn', current_preset_values['e_vpn']))
                with col_s5:
                    destinatario_novo = st.checkbox("Destinatário Novo", value=st.session_state.get('sim_destinatario_novo', current_preset_values['destinatario_novo']))
                    fim_semana = st.checkbox("Fim de Semana/Noite", value=st.session_state.get('sim_fim_semana', current_preset_values['fim_semana']))
                    tentativas_senha = st.slider("Tentativas de Senha", 0, 5, value=st.session_state.get('sim_tentativas_senha', current_preset_values['tentativas_senha']))
                    senha_alterada_recentemente = st.checkbox("Senha Alterada Recentemente", value=st.session_state.get('sim_senha_alterada_recentemente', current_preset_values['senha_alterada_recentemente']))
                with col_s6:
                    tipo_dispositivo = st.selectbox("Tipo de Dispositivo", ['mobile', 'desktop'], index=['mobile', 'desktop'].index(st.session_state.get('sim_tipo_dispositivo', current_preset_values['tipo_dispositivo'])))
                    os_sim = st.selectbox("Sistema Operacional", ['Android', 'iOS', 'Windows', 'MacOS'], index=['Android', 'iOS', 'Windows', 'MacOS'].index(st.session_state.get('sim_os_sim', current_preset_values['os_sim'])))
                    tipo_ip = st.selectbox("Tipo de IP", ['residencial', 'corporativo', 'vpn'], index=['residencial', 'corporativo', 'vpn'].index(st.session_state.get('sim_tipo_ip', current_preset_values['tipo_ip'])))
                    distancia_ultima_transacao = st.number_input("Distância Última Transação (Km)", 0, 5000, value=st.session_state.get('sim_distancia_ultima_transacao', current_preset_values['distancia_ultima_transacao']))

                st.markdown("---")
                st.subheader("Histórico do Usuário e Destinatário")
                col_s7, col_s8 = st.columns(2)
                with col_s7:
                    media_valor_user = st.number_input("Média Histórica do Usuário (R$)", 50.0, 2000.0, value=st.session_state.get('sim_media_valor_user', current_preset_values['media_valor_user']))
                    desvio_valor = abs(valor - media_valor_user) / media_valor_user if media_valor_user > 0 else 0
                    st.write(f"Desvio de Valor Calculado: {desvio_valor:.2f}")
                    
                    df_current = load_data_cached()
                    current_user_data = df_current[df_current['user_id'] == user_id]
                    fraudes_passadas_db = current_user_data['fraudes_passadas'].iloc[0] if not current_user_data.empty and 'fraudes_passadas' in current_user_data.columns else 0
                    fraudes_passadas = st.number_input(f"Fraudes Passadas (do DB para User {user_id}):", 0, 10, value=st.session_state.get('sim_fraudes_passadas', int(fraudes_passadas_db)))
                with col_s8:
                    freq_transacao_destinatario = st.number_input("Frequência Transação Destinatário", 0, 50, value=st.session_state.get('sim_freq_transacao_destinatario', current_preset_values['freq_transacao_destinatario']))
                    valor_total_enviado_destinatario = st.number_input("Valor Total Enviado Destinatário (R$)", 0.0, 10000.0, value=st.session_state.get('sim_valor_total_enviado_destinatario', current_preset_values['valor_total_enviado_destinatario']))
                    velocidade_transacoes = st.number_input("Velocidade Transações (por min)", 0.0, 10.0, value=st.session_state.get('sim_velocidade_transacoes', current_preset_values['velocidade_transacoes']))
                    intervalo_medio_transacoes = st.number_input("Intervalo Médio Transações (min)", 0.0, 1440.0, value=st.session_state.get('sim_intervalo_medio_transacoes', current_preset_values['intervalo_medio_transacoes']))
                    score_valor_z = st.number_input("Score Z do Valor", -5.0, 5.0, value=st.session_state.get('sim_score_valor_z', current_preset_values['score_valor_z']))

                submitted = st.form_submit_button("🎲 Simular Transação")
            
            if submitted:
                try:
                    temp_transaction_id = int(time.time() * 1000)
                    
                    new_data = {
                        'id': temp_transaction_id,
                        'user_id': user_id,
                        'hora': hora,
                        'valor': valor,
                        'tipo_transacao': tipo_transacao,
                        'dispositivo_novo': int(dispositivo_novo),
                        'ip_novo': int(ip_novo),
                        'tentativas_senha': tentativas_senha,
                        'mudanca_localizacao': int(mudanca_localizacao),
                        'destinatario_novo': int(destinatario_novo),
                        'fim_semana': int(fim_semana),
                        'score_credito': score_credito,
                        'desvio_valor': desvio_valor,
                        'media_valor_user': media_valor_user,
                        'fraudes_passadas': fraudes_passadas,
                        
                        'dia_semana': dia_semana,
                        'periodo_dia': periodo_dia,
                        'limite_cartao': limite_cartao,
                        'saldo_conta': saldo_conta,
                        'distancia_ultima_transacao': distancia_ultima_transacao,
                        'tipo_dispositivo': tipo_dispositivo,
                        'os': os_sim,
                        'tipo_ip': tipo_ip,
                        'e_vpn': int(e_vpn),
                        'senha_alterada_recentemente': int(senha_alterada_recentemente),
                        'freq_transacao_destinatario': freq_transacao_destinatario,
                        'valor_total_enviado_destinatario': valor_total_enviado_destinatario,
                        'velocidade_transacoes': velocidade_transacoes,
                        'intervalo_medio_transacoes': intervalo_medio_transacoes,
                        'score_valor_z': score_valor_z
                    }
                    
                    result = predict_transaction(new_data, threshold=threshold_risk)
                    
                    if 'error' in result:
                        st.error(result['error'])
                    else:
                        col_r1, col_r2 = st.columns(2)
                        
                        with col_r1:
                            if result['alert'] == 'BLOQUEIO AUTOMÁTICO':
                                st.markdown(f"""
                                <div style="color: #fff; padding: 15px; border-left: 5px solid #f44336; border-radius: 5px;">
                                    <h3>🚨 TRANSAÇÃO BLOQUEADA</h3>
                                    <p><strong>Risk Score:</strong> {result['risk_score']}/100</p>
                                    <p><strong>Probabilidade:</strong> {result['proba']:.1%}</p>
                                    <p><strong>Valor:</strong> R$ {valor:.2f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div style="color: #fff; padding: 15px; border-left: 5px solid #4caf50; border-radius: 5px;">
                                    <h3>✅ TRANSAÇÃO APROVADA</h3>
                                    <p><strong>Risk Score:</strong> {result['risk_score']}/100</p>
                                    <p><strong>Probabilidade:</strong> {result['proba']:.1%}</p>
                                    <p><strong>Valor:</strong> R$ {valor:.2f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col_r2:
                            st.write("**Fatores de Risco:**")
                            factors = []
                            if desvio_valor > 0.5: factors.append(f"Alto desvio de valor: {desvio_valor:.1f}")
                            if abs(score_valor_z) > 2: factors.append(f"Z-Score de valor anômalo: {score_valor_z:.1f}")
                            if hora < 6 or hora > 22: factors.append("Horário suspeito (madrugada/noite)")
                            if dispositivo_novo: factors.append("Dispositivo novo")
                            if ip_novo: factors.append("IP novo")
                            if mudanca_localizacao: factors.append("Mudança de local")
                            if tentativas_senha > 0: factors.append(f"Tentativas de senha: {tentativas_senha}")
                            if fraudes_passadas > 0: factors.append(f"Histórico de {fraudes_passadas} fraudes")
                            if distancia_ultima_transacao > 500: factors.append(f"Grande distância da última transação: {distancia_ultima_transacao} Km")
                            if e_vpn: factors.append("Uso de VPN detectado")
                            if senha_alterada_recentemente: factors.append("Senha alterada recentemente")
                            if velocidade_transacoes > 3: factors.append(f"Alta velocidade de transações: {velocidade_transacoes}/min")
                            if destinatario_novo: factors.append("Destinatário novo")
                            
                            if factors:
                                for factor in factors:
                                    st.write(f"• {factor}")
                            else:
                                st.write("• Perfil normal de transação")
                        
                        if result['alert'] == 'BLOQUEIO AUTOMÁTICO':
                            st.warning("Esta transação foi BLOQUEADA AUTOMATICAMENTE. Por favor, confirme se é fraude ou não.")
                            confirm_container = st.container()
                            with confirm_container:
                                col_confirm, col_deny = st.columns(2)
                                with col_confirm:
                                    if st.button("✅ Confirmar Fraude (Golpe)", key="confirm_fraud"):
                                        update_transaction_status(temp_transaction_id, user_id, 1)
                                        st.success(f"Transação {temp_transaction_id} marcada como FRAUDE e usuário {user_id} atualizado.")
                                        st.cache_data.clear()
                                        st.rerun()
                                with col_deny:
                                    if st.button("❌ Liberar Transação (Não é Golpe)", key="deny_fraud"):
                                        update_transaction_status(temp_transaction_id, user_id, 0)
                                        st.info(f"Transação {temp_transaction_id} liberada e usuário {user_id} atualizado.")
                                        st.cache_data.clear()
                                        st.rerun()
                
                except Exception as e:
                    st.error(f"Erro na simulação: {str(e)}")
                    st.error(traceback.format_exc())
        else:
            st.warning("⚠️ Treine o modelo primeiro para predições individuais.")
            st.info("Use o botão '🤖 Treinar Modelo' acima.")

else:
    st.warning("⚠️ Dashboard não carregado. Verifique o debug acima. Passos: 1. Gerar Dados → 2. Configurar Banco → 3. Treinar Modelo.")
    st.info("""
    ### Como usar o sistema:
    
    1. **📊 Gerar Dados**: Cria um dataset sintético de transações com fraudes simuladas
    2. **🗄️ Configurar Banco**: Inicializa o banco SQLite com histórico de usuários  
    3. **🤖 Treinar Modelo**: Treina o XGBoost com balanceamento SMOTE para detectar fraudes
    4. **📊 Monitorar**: Visualize transações, alertas e métricas em tempo real
    5. **🔮 Simular**: Teste o modelo com transações customizadas
    
    ### Tecnologias utilizadas:
    - **Machine Learning**: XGBoost + SMOTE para alta precisão
    - **Banco de Dados**: SQLite para histórico de usuários
    - **Visualização**: Plotly para dashboards interativos
    - **Interface**: Streamlit para experiência moderna
    
    **👆 Comece clicando em "📊 Gerar Dados" acima!**
    """)

st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Status dos Arquivos")

files_status = {
    "transactions.csv": os.path.exists('transactions.csv'),
    "arion.db": os.path.exists('arion.db'),
    "fraud_model_pipeline.pkl": os.path.exists('fraud_model_pipeline.pkl'),
    "transactions_predicted.csv": os.path.exists('transactions_predicted.csv')
}

for file, exists in files_status.items():
    status = "✅" if exists else "❌"
    st.sidebar.write(f"{status} {file}")

st.sidebar.markdown("---")
st.sidebar.info(f"**Threshold atual:** {threshold_risk}")

st.markdown("---")
st.markdown("*Arion - Sistema Inteligente de Detecção de Fraudes | Desenvolvido para Hackathon*")