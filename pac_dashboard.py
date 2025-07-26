import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import fredapi
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Configurazione pagina
st.set_page_config(
    page_title="PAC Dynamic Rebalancing Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Classe per gestire i dati
class DataCollector:
    def __init__(self, fred_api_key: str):
        self.fred = fredapi.Fred(api_key=fred_api_key)
        
    def get_vix_data(self, days: int = 30) -> pd.DataFrame:
        """Ottieni dati VIX da Yahoo Finance"""
        try:
            vix = yf.download("^VIX", period=f"{days*2}d", interval="1d")
            return vix['Close'].dropna()
        except Exception as e:
            st.error(f"Errore nel recupero dati VIX: {e}")
            return pd.Series()
    
    def get_fred_data(self, series_id: str, days: int = 365) -> pd.Series:
        """Ottieni dati da FRED"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days * 1.2)  # Un po' di buffer per weekend/festivi
            data = self.fred.get_series(series_id, start_date=start_date, end_date=end_date)
            data = data.dropna()
            
            if len(data) == 0:
                return pd.Series()
            
            # Per i dati Treasury (giornalieri), limita esattamente ai giorni richiesti
            if series_id in ['GS10', 'GS2']:
                if len(data) > days:
                    data = data.tail(days)
            
            # Per dati mensili (CPI, UNRATE), calcola i mesi corrispondenti
            elif series_id in ['CPIAUCSL', 'UNRATE']:
                # Per questi dati, usa l'intero periodo disponibile se è per calcoli
                # ma se è per display, limita proporzionalmente
                pass  # Mantieni tutti i dati disponibili per ora
                    
            return data
        except Exception as e:
            st.error(f"Errore nel recupero dati FRED {series_id}: {e}")
            return pd.Series()
    
    def get_dxy_data(self, days: int = 30) -> pd.Series:
        """Ottieni dati DXY"""
        try:
            # Calcola il periodo corretto
            if days <= 30:
                period = "1mo"
            elif days <= 90:
                period = "3mo"
            elif days <= 180:
                period = "6mo"
            elif days <= 365:
                period = "1y"
            else:
                period = "2y"
            
            dxy = yf.download("DX-Y.NYB", period=period, interval="1d", progress=False)
            if dxy.empty:
                return pd.Series()
            
            # Prendi solo gli ultimi 'days' giorni
            dxy_close = dxy['Close'].dropna()
            if len(dxy_close) > days:
                dxy_close = dxy_close.tail(days)
                
            return dxy_close
        except Exception as e:
            st.error(f"Errore nel recupero dati DXY: {e}")
            return pd.Series()

# Classe per analizzare scenari
class ScenarioAnalyzer:
    def __init__(self):
        self.base_allocation = {
            'S&P 500': 22,
            'Europa': 20,
            'Emergenti': 17,
            'World': 12,
            'Berkshire Hathaway': 12,
            'Small Cap': 10,
            'Oro': 5,
            'Bitcoin': 2
        }
        
        self.scenarios = {
            'RISK-OFF': {
                'S&P 500': 18, 'Europa': 15, 'Emergenti': 12, 'World': 10,
                'Berkshire Hathaway': 15, 'Small Cap': 7, 'Oro': 12, 'Bitcoin': 11
            },
            'STAGFLAZIONE': {
                'S&P 500': 20, 'Europa': 18, 'Emergenti': 20, 'World': 10,
                'Berkshire Hathaway': 10, 'Small Cap': 8, 'Oro': 12, 'Bitcoin': 2
            },
            'CRESCITA FORTE': {
                'S&P 500': 26, 'Europa': 22, 'Emergenti': 20, 'World': 14,
                'Berkshire Hathaway': 8, 'Small Cap': 13, 'Oro': 2, 'Bitcoin': 5
            },
            'DOLLARO FORTE': {
                'S&P 500': 25, 'Europa': 15, 'Emergenti': 12, 'World': 10,
                'Berkshire Hathaway': 15, 'Small Cap': 8, 'Oro': 8, 'Bitcoin': 7
            },
            'RECESSIONE': {
                'S&P 500': 15, 'Europa': 12, 'Emergenti': 10, 'World': 8,
                'Berkshire Hathaway': 20, 'Small Cap': 5, 'Oro': 15, 'Bitcoin': 15
            }
        }
    
    def analyze_vix(self, vix_data: pd.Series) -> Dict:
        if len(vix_data) == 0:
            return {"level": "N/A", "signal": "neutral", "current": 0, "trend": "flat"}
        
        # Assicurati che sia un valore singolo
        current_vix = float(vix_data.iloc[-1])
        recent_vix = float(vix_data.tail(10).mean())
        
        if current_vix > 35:
            level = "CRITICO"
            signal = "risk-off"
        elif current_vix > 25:
            level = "ALTO"
            signal = "risk-off"
        elif current_vix < 15:
            level = "BASSO"
            signal = "risk-on"
        else:
            level = "MEDIO"
            signal = "neutral"
        
        # Trend
        if len(vix_data) >= 5:
            trend = "up" if float(vix_data.iloc[-1]) > float(vix_data.iloc[-5]) else "down"
        else:
            trend = "flat"
            
        return {
            "level": level,
            "signal": signal,
            "current": current_vix,
            "recent_avg": recent_vix,
            "trend": trend
        }
    
    def analyze_inflation(self, cpi_data: pd.Series) -> Dict:
        if len(cpi_data) == 0:
            return {"level": "N/A", "signal": "neutral", "current": 0}
        
        current_cpi = float(cpi_data.iloc[-1])
        
        if current_cpi > 5:
            level = "CRITICO"
            signal = "high-inflation"
        elif current_cpi > 3:
            level = "ELEVATO"
            signal = "high-inflation"
        elif current_cpi < 2:
            level = "BASSO"
            signal = "low-inflation"
        else:
            level = "TARGET"
            signal = "neutral"
            
        return {
            "level": level,
            "signal": signal,
            "current": current_cpi
        }
    
    def analyze_unemployment(self, unemployment_data: pd.Series) -> Dict:
        if len(unemployment_data) == 0:
            return {"level": "N/A", "signal": "neutral", "current": 0}
        
        current_unemployment = float(unemployment_data.iloc[-1])
        
        if current_unemployment > 8:
            level = "CRITICO"
            signal = "recession"
        elif current_unemployment > 6:
            level = "ELEVATO"
            signal = "weak-economy"
        elif current_unemployment < 4:
            level = "BASSO"
            signal = "strong-economy"
        else:
            level = "NORMALE"
            signal = "neutral"
            
        return {
            "level": level,
            "signal": signal,
            "current": current_unemployment
        }
    
    def analyze_yield_curve(self, ten_year: pd.Series, two_year: pd.Series) -> Dict:
        if len(ten_year) == 0 or len(two_year) == 0:
            return {"level": "N/A", "signal": "neutral", "current": 0}
        
        # Allinea le date
        common_dates = ten_year.index.intersection(two_year.index)
        if len(common_dates) == 0:
            return {"level": "N/A", "signal": "neutral", "current": 0}
        
        ten_year_aligned = ten_year[common_dates]
        two_year_aligned = two_year[common_dates]
        
        spread = ten_year_aligned - two_year_aligned
        current_spread = float(spread.iloc[-1])
        
        if current_spread < 0:
            level = "INVERTITO"
            signal = "recession-risk"
        elif current_spread < 0.5:
            level = "PIATTO"
            signal = "caution"
        else:
            level = "NORMALE"
            signal = "healthy"
            
        return {
            "level": level,
            "signal": signal,
            "current": current_spread
        }
    
    def analyze_dxy(self, dxy_data: pd.Series) -> Dict:
        if len(dxy_data) == 0:
            return {"level": "N/A", "signal": "neutral", "current": 0}
        
        current_dxy = float(dxy_data.iloc[-1])
        
        if current_dxy > 105:
            level = "FORTE"
            signal = "strong-dollar"
        elif current_dxy < 95:
            level = "DEBOLE"
            signal = "weak-dollar"
        else:
            level = "NEUTRALE"
            signal = "neutral"
            
        return {
            "level": level,
            "signal": signal,
            "current": current_dxy
        }
    
    def determine_scenario(self, indicators: Dict) -> str:
        vix_signal = indicators.get('vix', {}).get('signal', 'neutral')
        inflation_signal = indicators.get('inflation', {}).get('signal', 'neutral')
        unemployment_signal = indicators.get('unemployment', {}).get('signal', 'neutral')
        yield_signal = indicators.get('yield_curve', {}).get('signal', 'neutral')
        dxy_signal = indicators.get('dxy', {}).get('signal', 'neutral')
        
        # Logica per determinare lo scenario
        if (vix_signal == 'risk-off' and 
            (unemployment_signal in ['weak-economy', 'recession'] or yield_signal == 'recession-risk')):
            return 'RISK-OFF'
        elif (inflation_signal == 'high-inflation' and 
              unemployment_signal in ['weak-economy', 'recession']):
            return 'STAGFLAZIONE'
        elif (vix_signal == 'risk-on' and 
              unemployment_signal == 'strong-economy' and 
              inflation_signal in ['low-inflation', 'neutral']):
            return 'CRESCITA FORTE'
        elif (dxy_signal == 'strong-dollar' and 
              inflation_signal == 'high-inflation'):
            return 'DOLLARO FORTE'
        elif (yield_signal == 'recession-risk' and 
              unemployment_signal in ['weak-economy', 'recession']):
            return 'RECESSIONE'
        else:
            return 'NEUTRALE'

# Funzioni per i grafici
def create_gauge_chart(value: float, title: str, min_val: float, max_val: float, 
                      thresholds: List[float], colors: List[str]) -> go.Figure:
    """Crea un grafico a gauge per indicatori"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 16}},
        gauge = {
            'axis': {'range': [min_val, max_val], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [min_val, thresholds[0]], 'color': colors[0]},
                {'range': [thresholds[0], thresholds[1]], 'color': colors[1]},
                {'range': [thresholds[1], thresholds[2]], 'color': colors[2]},
                {'range': [thresholds[2], max_val], 'color': colors[3]}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_trend_chart(data: pd.Series, title: str, thresholds: List[float] = None) -> go.Figure:
    """Crea un grafico di trend con soglie"""
    fig = go.Figure()
    
    # Verifica che ci siano dati
    if len(data) == 0:
        fig.add_annotation(
            text="Nessun dato disponibile",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
    else:
        # Linea principale
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data.values,
            mode='lines',
            name=title,
            line=dict(width=2, color='blue')
        ))
    
        # Aggiungi soglie se fornite
        if thresholds:
            colors = ['green', 'yellow', 'orange', 'red']
            names = ['Basso', 'Medio', 'Alto', 'Critico']
            for i, threshold in enumerate(thresholds):
                fig.add_hline(y=threshold, line_dash="dash", 
                             line_color=colors[i], 
                             annotation_text=names[i],
                             annotation_position="right")
    
    fig.update_layout(
        title=title,
        xaxis_title="Data",
        yaxis_title="Valore",
        height=400,
        showlegend=True,
        xaxis=dict(
            tickangle=45,
            tickformat='%Y-%m-%d'
        )
    )
    
    return fig

def create_allocation_comparison(base: Dict, scenario: Dict, scenario_name: str) -> go.Figure:
    """Crea grafico comparativo allocazioni"""
    assets = list(base.keys())
    base_values = list(base.values())
    scenario_values = list(scenario.values())
    
    fig = go.Figure(data=[
        go.Bar(name='Allocazione Base', x=assets, y=base_values, marker_color='lightblue'),
        go.Bar(name=f'Scenario {scenario_name}', x=assets, y=scenario_values, marker_color='darkblue')
    ])
    
    fig.update_layout(
        title=f'Confronto Allocazioni: Base vs {scenario_name}',
        xaxis_title="Asset",
        yaxis_title="Allocazione %",
        barmode='group',
        height=500
    )
    
    return fig

# UI Principal
def main():
    st.title("🎯 PAC Dynamic Rebalancing Dashboard")
    st.markdown("---")
    
    # Sidebar per configurazione
    with st.sidebar:
        st.header("⚙️ Configurazione")
        fred_api_key = st.text_input("FRED API Key", type="password", 
                                   help="Inserisci la tua chiave API FRED")
        
        if not fred_api_key:
            st.warning("⚠️ Inserisci la chiave API FRED per continuare")
            st.stop()
        
        st.header("📊 Parametri")
        lookback_days = st.slider("Giorni di lookback", 30, 365, 90)
        auto_refresh = st.checkbox("Auto-refresh (60s)", value=False)
    
    # Inizializza collectors
    collector = DataCollector(fred_api_key)
    analyzer = ScenarioAnalyzer()
    
    # Auto-refresh
    if auto_refresh:
        st.empty()
        placeholder = st.empty()
        time.sleep(60)
        st.experimental_rerun()
    
    # Raccolta dati
    with st.spinner("📡 Raccolta dati in corso..."):
        try:
            # VIX
            vix_data = collector.get_vix_data(lookback_days)
            
            # Dati FRED
            cpi_data = collector.get_fred_data('CPIAUCSL', lookback_days*3)  # CPI mensile
            unemployment_data = collector.get_fred_data('UNRATE', lookback_days*3)  # Disoccupazione mensile
            ten_year_data = collector.get_fred_data('GS10', lookback_days)  # 10-Year Treasury
            two_year_data = collector.get_fred_data('GS2', lookback_days)   # 2-Year Treasury
            
            # DXY
            dxy_data = collector.get_dxy_data(lookback_days)
            
            # Calcola CPI YoY se abbiamo abbastanza dati
            if len(cpi_data) >= 12:
                cpi_yoy = cpi_data.pct_change(12) * 100
                cpi_yoy = cpi_yoy.dropna()
            else:
                cpi_yoy = pd.Series()
            
        except Exception as e:
            st.error(f"Errore nella raccolta dati: {e}")
            st.stop()
    
    # Analisi indicatori
    indicators = {
        'vix': analyzer.analyze_vix(vix_data),
        'inflation': analyzer.analyze_inflation(cpi_yoy),
        'unemployment': analyzer.analyze_unemployment(unemployment_data),
        'yield_curve': analyzer.analyze_yield_curve(ten_year_data, two_year_data),
        'dxy': analyzer.analyze_dxy(dxy_data)
    }
    
    # Determina scenario
    current_scenario = analyzer.determine_scenario(indicators)
    
    # Layout principale
    col1, col2 = st.columns([2, 1])
    
    with col2:
        # Scenario attuale
        st.markdown("### 🎯 Scenario Attuale")
        scenario_colors = {
            'NEUTRALE': 'blue',
            'RISK-OFF': 'red',
            'STAGFLAZIONE': 'orange',
            'CRESCITA FORTE': 'green',
            'DOLLARO FORTE': 'purple',
            'RECESSIONE': 'darkred'
        }
        
        st.markdown(f"<h2 style='color: {scenario_colors.get(current_scenario, 'blue')}'>{current_scenario}</h2>", 
                   unsafe_allow_html=True)
        
        # Riepilogo indicatori con spiegazioni - VERSIONE CORRETTA
        st.markdown("### 📊 Riepilogo Indicatori")
        
        for name, data in indicators.items():
            level = data.get('level', 'N/A')
            current = data.get('current', 0)
            
            if name == 'vix':
                st.metric("VIX (Volatilità)", f"{current:.1f}", f"Livello: {level}")
                with st.expander("📖 VIX - Cos'è e come interpretarlo"):
                    st.markdown("""
                    **Cosa indica**: Misura la volatilità implicita attesa dal mercato azionario USA nei prossimi 30 giorni
                    - **Basso (<15)**: 🟢 Mercati calmi, sentiment positivo
                    - **Medio (15-25)**: 🟡 Volatilità normale  
                    - **Alto (25-35)**: 🟠 Nervosismo del mercato, incertezza
                    - **Critico (>35)**: 🔴 Panico, crisi in corso
                    """)
                    
            elif name == 'inflation':
                st.metric("Inflazione CPI YoY", f"{current:.1f}%", f"Livello: {level}")
                with st.expander("📖 Inflazione - Cos'è e come interpretarla"):
                    st.markdown("""
                    **Cosa indica**: Variazione percentuale annuale dell'indice dei prezzi al consumo USA
                    - **Basso (<2%)**: 🔵 Sotto target Fed, possibile deflazione
                    - **Target (2-3%)**: 🟢 Obiettivo Fed, economia sana
                    - **Elevato (3-5%)**: 🟠 Sopra target, pressioni inflazionistiche
                    - **Critico (>5%)**: 🔴 Inflazione alta, rischio stagflazione
                    """)
                    
            elif name == 'unemployment':
                st.metric("Disoccupazione", f"{current:.1f}%", f"Livello: {level}")
                with st.expander("📖 Disoccupazione - Cos'è e come interpretarla"):
                    st.markdown("""
                    **Cosa indica**: Percentuale della forza lavoro USA senza impiego ma in cerca di lavoro
                    - **Basso (<4%)**: 🟢 Piena occupazione, economia forte
                    - **Normale (4-6%)**: 🟡 Livello fisiologico di disoccupazione
                    - **Elevato (6-8%)**: 🟠 Debolezza economica, rallentamento
                    - **Critico (>8%)**: 🔴 Recessione, crisi occupazionale
                    """)
                    
            elif name == 'yield_curve':
                st.metric("Yield Curve (10Y-2Y)", f"{current:.2f}%", f"Livello: {level}")
                with st.expander("📖 Yield Curve - Cos'è e come interpretarla"):
                    st.markdown("""
                    **Cosa indica**: Differenza tra rendimenti Treasury 10 anni e 2 anni (forma della curva dei rendimenti)
                    - **Normale (>0.5%)**: 🟢 Crescita economica attesa, normalità
                    - **Piatto (0-0.5%)**: 🟡 Rallentamento previsto, cautela
                    - **Invertito (<0%)**: 🔴 **ALLARME RECESSIONE** - storicamente precede le recessioni di 6-18 mesi
                    
                    *Inversione = investitori preferiscono titoli a lungo termine (aspettano tassi più bassi)*
                    """)
                    
            elif name == 'dxy':
                st.metric("DXY (Dollaro)", f"{current:.1f}", f"Livello: {level}")
                with st.expander("📖 DXY - Cos'è e come interpretarlo"):
                    st.markdown("""
                    **Cosa indica**: Forza del dollaro USA contro un paniere di 6 valute principali (EUR, JPY, GBP, CAD, SEK, CHF)
                    - **Debole (<95)**: 🔴 Dollaro svalutato, inflazione importata, commodities care
                    - **Neutrale (95-105)**: 🟡 Equilibrio valutario normale
                    - **Forte (>105)**: 🟠 Dollaro sopravvalutato, deflationary per USA, stress mercati emergenti
                    
                    *Dollaro forte = problema per export USA e debiti in $ dei paesi emergenti*
                    """)
            
            st.markdown("---")
    
    with col1:
        # Grafici a gauge
        st.markdown("### 🌡️ Dashboard Indicatori")
        
        gauge_col1, gauge_col2 = st.columns(2)
        
        with gauge_col1:
            # VIX Gauge
            if len(vix_data) > 0:
                vix_fig = create_gauge_chart(
                    indicators['vix']['current'],
                    "VIX (Indice Paura)",
                    0, 50,
                    [15, 25, 35],
                    ['lightgreen', 'yellow', 'orange', 'red']
                )
                st.plotly_chart(vix_fig, use_container_width=True)
            
            # Inflazione Gauge
            if len(cpi_yoy) > 0:
                inflation_fig = create_gauge_chart(
                    indicators['inflation']['current'],
                    "Inflazione CPI YoY %",
                    -1, 8,
                    [2, 3, 5],
                    ['lightblue', 'lightgreen', 'yellow', 'red']
                )
                st.plotly_chart(inflation_fig, use_container_width=True)
        
        with gauge_col2:
            # Disoccupazione Gauge
            if len(unemployment_data) > 0:
                unemployment_fig = create_gauge_chart(
                    indicators['unemployment']['current'],
                    "Disoccupazione USA %",
                    2, 12,
                    [4, 6, 8],
                    ['green', 'yellow', 'orange', 'red']
                )
                st.plotly_chart(unemployment_fig, use_container_width=True)
            
            # DXY Gauge
            if len(dxy_data) > 0:
                dxy_fig = create_gauge_chart(
                    indicators['dxy']['current'],
                    "DXY (Forza Dollaro)",
                    85, 115,
                    [95, 100, 105],
                    ['red', 'yellow', 'lightgreen', 'orange']
                )
                st.plotly_chart(dxy_fig, use_container_width=True)
        
        # Yield Curve come gauge separato
        if len(ten_year_data) > 0 and len(two_year_data) > 0:
            common_dates = ten_year_data.index.intersection(two_year_data.index)
            if len(common_dates) > 0:
                yield_fig = create_gauge_chart(
                    indicators['yield_curve']['current'],
                    "Yield Curve Spread (10Y-2Y)",
                    -2, 3,
                    [-0.5, 0, 0.5],
                    ['red', 'orange', 'yellow', 'green']
                )
                st.plotly_chart(yield_fig, use_container_width=True)
    

    # Allocazione suggerita
    st.markdown("### 💼 Allocazione Suggerita")
    
    if current_scenario != 'NEUTRALE':
        scenario_allocation = analyzer.scenarios[current_scenario]
        allocation_fig = create_allocation_comparison(
            analyzer.base_allocation, 
            scenario_allocation, 
            current_scenario
        )
        st.plotly_chart(allocation_fig, use_container_width=True)
        
        # Tabella delle variazioni
        st.markdown("#### 📋 Variazioni Raccomandate")
        changes_data = []
        for asset in analyzer.base_allocation.keys():
            base_val = analyzer.base_allocation[asset]
            scenario_val = scenario_allocation[asset]
            change = scenario_val - base_val
            changes_data.append({
                'Asset': asset,
                'Base %': base_val,
                'Nuovo %': scenario_val,
                'Variazione': f"{change:+.0f}%",
                'Variazione pp': change
            })
        
        changes_df = pd.DataFrame(changes_data)
        changes_df = changes_df.sort_values('Variazione pp', ascending=False)
        
        # Colora le righe in base alla variazione
        def color_changes(val):
            if val > 0:
                return 'background-color: lightgreen'
            elif val < 0:
                return 'background-color: lightcoral'
            else:
                return ''
        
        styled_df = changes_df.style.applymap(color_changes, subset=['Variazione pp'])
        st.dataframe(styled_df, use_container_width=True)
        
    else:
        st.info("🟢 Scenario NEUTRALE: Mantieni l'allocazione base")
        base_df = pd.DataFrame(list(analyzer.base_allocation.items()), 
                              columns=['Asset', 'Allocazione %'])
        st.dataframe(base_df, use_container_width=True)
    
    # Footer con timestamp
    st.markdown("---")
    st.markdown(f"*Ultimo aggiornamento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Legend
    with st.expander("📖 Guida agli Scenari di Investimento"):
        st.markdown("""
        ### 🎯 Scenari di Ribilanciamento
        
        **🔵 NEUTRALE** - *Mantieni allocazione base*
        - Tutti gli indicatori in range normale
        - VIX 15-25, Inflazione 2-3%, Disoccupazione 4-6%
        - Nessun segnale di stress particolare
        
        **🔴 RISK-OFF** - *Protezione patrimoniale*
        - **Trigger**: VIX alto + (Disoccupazione >6% OR Yield invertito)
        - **Strategia**: ↗️ Oro, Bitcoin, Berkshire | ↘️ Azionario generale
        - **Razionale**: Mercati in panico, rifugio in beni sicuri
        
        **🟠 STAGFLAZIONE** - *Protezione inflazione*
        - **Trigger**: Inflazione >4% + Disoccupazione >5% + VIX >20
        - **Strategia**: ↗️ Emergenti, Oro | ↘️ Crescita USA/Europa
        - **Razionale**: Economia stagnante con prezzi in crescita
        
        **🟢 CRESCITA FORTE** - *Massima esposizione growth*
        - **Trigger**: VIX <15 + Disoccupazione <4.5% + Inflazione <3%
        - **Strategia**: ↗️ Azionario USA/Europa, Small Cap | ↘️ Oro, Berkshire
        - **Razionale**: Economia in espansione, sentiment positivo
        
        **🟣 DOLLARO FORTE** - *Adattamento valutario*
        - **Trigger**: DXY >105 + Inflazione >3.5%
        - **Strategia**: ↗️ USA, Berkshire, Bitcoin | ↘️ Europa, Emergenti
        - **Razionale**: Dollaro forte penalizza mercati internazionali
        
        **⚫ RECESSIONE** - *Massima protezione*
        - **Trigger**: Yield invertito 30+ giorni + Disoccupazione crescente
        - **Strategia**: ↗️ Berkshire, Oro, Bitcoin | ↘️ Tutto l'azionario
        - **Razionale**: Recessione imminente, massima prudenza
        
        ### 🏛️ Perché Questi Indicatori?
        
        - **VIX**: Termometro della paura del mercato, predice volatilità
        - **Inflazione**: Impatta politica Fed e potere d'acquisto
        - **Disoccupazione**: Indica salute economia reale 
        - **Yield Curve**: Miglior predittore storico di recessioni
        - **DXY**: Influenza competitività USA vs resto del mondo
        """)
        
    with st.expander("⚖️ Logica delle Allocazioni"):
        st.markdown("""
        ### 🛡️ Asset Difensivi (Risk-Off)
        - **Berkshire Hathaway**: Cash + aziende quality, gestione Buffett
        - **Oro**: Bene rifugio storico, hedge inflazione
        - **Bitcoin**: Riserva di valore digitale, decorrelato (ma volatile)
        
        ### 📈 Asset Crescita (Risk-On)  
        - **S&P 500**: Locomotiva dell'economia USA
        - **Small Cap**: Maggiore potenziale ma più rischio
        - **Europa/Emergenti**: Diversificazione geografica
        
        ### 🌍 Considerazioni Geografiche
        - **Dollaro forte** → Favorisce asset USA, penalizza internazionali
        - **Inflazione USA** → Mercati emergenti spesso beneficiano
        - **Recessione USA** → Spesso si diffonde globalmente
        """)
    

if __name__ == "__main__":
    main()