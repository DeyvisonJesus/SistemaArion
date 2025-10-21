# Arion - Detecção de Fraudes Financeiras com IA

Sistema inteligente para detectar fraudes em tempo real, focado em Pix e transações brasileiras. MVP para Hackathon.

## Setup
1. Instale Python 3.11.9`
2. Instale dependências: `poetry install`
3. Rode o script:
   - `poetry run streamlit run app.py`

## Expansão (v2)
- Adicione NLP: `poetry add spacy` e `poetry run python -m spacy download pt_core_news_sm`.
- Análise de rede: `poetry add networkx`.

Desenvolvido com Python 3.11 + Poetry. Boa sorte no hackathon! 🛡️

# 🚨 CONCEITO – Sistema Inteligente de Detecção e Prevenção de Fraudes Financeiras

### **Nome do Projeto:** *Arion*  
Um sistema inteligente de **detecção e prevenção de fraudes financeiras em tempo real**, voltado para bancos, fintechs, usuários finais e reguladores.

---

## 📌 Identificação do Problema

### **1.1 Problema específico**
**Como detectar fraudes financeiras antes que causem prejuízos aos usuários e instituições?**

- 📈 Perdas com fraudes no Pix cresceram **70% em 2024** (dados do BC)  
- 🎭 Golpes de engenharia social cada vez mais sofisticados  
- ⚠️ Sistemas tradicionais baseados em regras são facilmente burlados  
- 💸 Detecção tardia gera prejuízos milionários  

### **1.2 Importância**
- **Afetados:** 24 milhões de brasileiros foram vítimas de golpes do PIX ou boletos falsos (Datafolha)  
- **Consequências:** Perda financeira + perda de confiança no sistema bancário  
- **Impacto:** R$ 2,9 bilhões de prejuízo anual no Brasil  

### **1.3 Soluções existentes e limitações**
- Sistemas baseados em **regras fixas** → facilmente contornáveis  
- **Análise manual** → lenta e custosa  
- **Soluções internacionais** → não adaptadas ao contexto brasileiro  

---

## 💡 Conceito da Solução

### **2.1 Ideia principal**
Um sistema de **IA** que combina **análise comportamental**, **detecção de padrões** e **machine learning** para identificar fraudes **em tempo real**.

### **2.2 Como funciona**
- 🔎 **Análise comportamental** → aprende os padrões normais de cada usuário  
- 🚨 **Detecção de anomalias** → identifica transações fora do padrão  
- 🌐 **Análise de rede** → detecta contas conectadas a fraudes conhecidas  
- 💬 **Processamento de linguagem natural (PLN)** → analisa descrições suspeitas  

### **2.3 Usuários**
- **Bancos e fintechs** → proteção principal  
- **Usuários finais** → alertas e bloqueios automáticos  
- **Reguladores** → relatórios de compliance  

### **2.4 Exemplo prático**
*"João sempre faz Pix de até R$ 500 durante o dia. Às 2h da manhã, alguém tenta transferir R$ 3.000 para uma conta nova. O sistema detecta: horário anômalo + valor atípico + destinatário suspeito = BLOQUEIO AUTOMÁTICO + alerta ao João."*

---

## ⚙️ Implementação Técnica (Hackathon)

### **Stack sugerida**
- **Backend:** Python + Flask 
- **IA/ML:** scikit-learn, pandas, numpy  
- **Frontend:** Streamlit (para demo rápida)  
- **Banco:** SQLite (para simplicidade)  
- **Visualização:** Plotly, matplotlib  

### **Funcionalidades do MVP**
1. **Dashboard de Monitoramento**  
   - Transações em tempo real  
   - Score de risco por transação  
   - Alertas automáticos  
2. **Engine de Detecção**  
   - Análise de padrões comportamentais  
   - Scoring de risco (0–100)  
   - Regras personalizáveis  
3. **Módulo de Relatórios**  
   - Estatísticas de fraudes detectadas  
   - Falsos positivos/negativos  
   - ROI da prevenção  

---

## 👥 Usuários do Sistema

### **1. Bancos e Fintechs (Principal)**
- **Uso:** Integração via API, dashboards executivos e configuração de regras  
- **Benefícios:** Redução de fraudes, menor custo operacional, compliance automático, melhor experiência do cliente  

### **2. Usuários finais**
- **Uso:** Recebem alertas, confirmam/negar transações e visualizam relatórios de segurança  
- **Benefícios:** Proteção automática, menos golpes, transações legítimas mais rápidas  

### **3. Reguladores (BC, CVM)**
- **Uso:** Relatórios automáticos, dashboards agregados, auditoria de algoritmos  
- **Benefícios:** Visibilidade do cenário nacional, dados para políticas públicas, fiscalização mais eficiente  

---

## 🧑 Personas

- 👨‍💼 **Gerente de Risco do Banco** → precisa de dashboards claros e menos perdas  
- 👩‍💻 **Analista de Fraude** → quer priorização de casos críticos  
- 🧓 **Cliente do Banco** → deseja proteção simples e sem falsos alarmes  
- 👩‍⚖️ **Reguladora do BC** → precisa de dados agregados e compliance  

---

## 🔄 Exemplo de Fluxo (Tentativa de fraude Pix)

1. **Maria** tem o celular clonado  
2. **Fraudador** tenta transferir R$ 2.000 às 3h da manhã  
3. **Sistema** detecta: horário anômalo + valor alto + dispositivo diferente  
4. **Banco** recebe alerta e bloqueia a transação  
5. **Maria** recebe SMS: "Transação bloqueada por segurança. Foi você? (S/N)"  
6. **Maria** responde "N"  
7. **Sistema** confirma fraude e aprende com o caso  
8. **Banco Central** recebe relatório automático  

---

## 📊 Datasets para Treinar

- Simulações de transações **normais vs fraudulentas**  
- Padrões brasileiros (Pix, TED, cartão)  
- Variáveis: horários, valores, frequência, localização  

---

## 🚀 Status
📌 Projeto em desenvolvimento para hackathon – MVP inicial.  

