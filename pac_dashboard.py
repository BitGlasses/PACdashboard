import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import fredapi
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import warnings
import time

warnings.filterwarnings('ignore')

st.set_page_config(page_title="PAC Dynamic Rebalancing Dashboard", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

class DataCollector:
    def __init__(self, fred_api_key: str):
        self.fred = fredapi.Fred(api_key=fred_api_key)
        
    def get_vix_data(self, end_date: datetime, days: int = 30) -> pd.Series:
        try:
            start_date = end_date - timedelta(days=days * 2)
            vix = yf.download("^VIX", start=start_date, end=end_date, interval="1d", progress=False)
            return vix['Close'].dropna()
        except:
            return pd.Series()
    
    def get_fred_data(self, series_id: str, end_date: datetime, days: int = 365) -> pd.Series:
        try:
            start_date = end_date - timedelta(days=days * 1.5)
            data = self.fred.get_series(series_id, start_date, end_date)
            return data.dropna()
        except:
            return pd.Series()
    
    def get_dxy_data(self, end_date: datetime, days: int = 30) -> pd.Series:
        try:
            start_date = end_date - timedelta(days=days * 2)
            dxy = yf.download("DX-Y.NYB", start=start_date, end=end_date, interval="1d", progress=False)
            return dxy['Close'].dropna()
        except:
            return pd.Series()

class ScenarioAnalyzer:
    def __init__(self):
        self.base_allocation = {'S&P 500': 22, 'Europa': 20, 'Emergenti': 17, 'World': 12, 'Berkshire Hathaway': 12, 'Small Cap': 10, 'Oro': 5, 'Bitcoin': 2}
        self.scenarios = {
            'RISK-OFF': {'S&P 500': 18, 'Europa': 15, 'Emergenti': 12, 'World': 10, 'Berkshire Hathaway': 15, 'Small Cap': 7, 'Oro': 12, 'Bitcoin': 11},
            'STAGFLAZIONE': {'S&P 500': 20, 'Europa': 18, 'Emergenti': 20, 'World': 10, 'Berkshire Hathaway': 10, 'Small Cap': 8, 'Oro': 12, 'Bitcoin': 2},
            'CRESCITA FORTE': {'S&P 500': 26, 'Europa': 22, 'Emergenti': 20, 'World': 14, 'Berkshire Hathaway': 8, 'Small Cap': 13, 'Oro': 2, 'Bitcoin': 5},
            'DOLLARO FORTE': {'S&P 500': 25, 'Europa': 15, 'Emergenti': 12, 'World': 10, 'Berkshire Hathaway': 15, 'Small Cap': 8, 'Oro': 8, 'Bitcoin': 7},
            'RECESSIONE': {'S&P 500': 15, 'Europa': 12, 'Emergenti': 10, 'World': 8, 'Berkshire Hathaway': 20, 'Small Cap': 5, 'Oro': 15, 'Bitcoin': 15}
        }
    
    def analyze_vix(self, vix_data: pd.Series) -> Dict:
        if vix_data.empty: return {"level": "N/A", "signal": "neutral", "current": np.nan}
        try:
            # Gestisce sia Series che valori scalari
            last_value = vix_data.iloc[-1]
            current_vix = float(last_value.iloc[0] if hasattr(last_value, 'iloc') else last_value)
        except (TypeError, IndexError, AttributeError):
            return {"level": "N/A", "signal": "neutral", "current": np.nan}
        level = "CRITICO" if current_vix > 35 else "ALTO" if current_vix > 25 else "BASSO" if current_vix < 15 else "MEDIO"
        signal = "risk-off" if current_vix > 25 else "risk-on" if current_vix < 15 else "neutral"
        return {"level": level, "signal": signal, "current": current_vix}
    
    def analyze_inflation(self, cpi_data: pd.Series) -> Dict:
        if cpi_data.empty: return {"level": "N/A", "signal": "neutral", "current": np.nan}
        try:
            last_value = cpi_data.iloc[-1]
            current_cpi = float(last_value.iloc[0] if hasattr(last_value, 'iloc') else last_value)
        except (TypeError, IndexError, AttributeError):
            return {"level": "N/A", "signal": "neutral", "current": np.nan}
        level = "CRITICO" if current_cpi > 5 else "ELEVATO" if current_cpi > 3 else "BASSO" if current_cpi < 2 else "TARGET"
        signal = "high-inflation" if current_cpi > 3 else "low-inflation" if current_cpi < 2 else "neutral"
        return {"level": level, "signal": signal, "current": current_cpi}

    def analyze_unemployment(self, unemployment_data: pd.Series) -> Dict:
        if unemployment_data.empty: return {"level": "N/A", "signal": "neutral", "current": np.nan}
        try:
            last_value = unemployment_data.iloc[-1]
            current_unemployment = float(last_value.iloc[0] if hasattr(last_value, 'iloc') else last_value)
        except (TypeError, IndexError, AttributeError):
            return {"level": "N/A", "signal": "neutral", "current": np.nan}
        level = "CRITICO" if current_unemployment > 8 else "ELEVATO" if current_unemployment > 6 else "BASSO" if current_unemployment < 4 else "NORMALE"
        signal = "recession" if current_unemployment > 8 else "weak-economy" if current_unemployment > 6 else "strong-economy" if current_unemployment < 4 else "neutral"
        return {"level": level, "signal": signal, "current": current_unemployment}

    def analyze_yield_curve(self, ten_year: pd.Series, two_year: pd.Series) -> Dict:
        if ten_year.empty or two_year.empty: return {"level": "N/A", "signal": "neutral", "current": np.nan}
        common_index = ten_year.index.intersection(two_year.index)
        if common_index.empty: return {"level": "N/A", "signal": "neutral", "current": np.nan}
        spread = (ten_year.loc[common_index] - two_year.loc[common_index]).dropna()
        if spread.empty: return {"level": "N/A", "signal": "neutral", "current": np.nan}
        try:
            last_value = spread.iloc[-1]
            current_spread = float(last_value.iloc[0] if hasattr(last_value, 'iloc') else last_value)
        except (TypeError, IndexError, AttributeError):
            return {"level": "N/A", "signal": "neutral", "current": np.nan}
        level = "INVERTITO" if current_spread < 0 else "PIATTO" if current_spread < 0.5 else "NORMALE"
        signal = "recession-risk" if current_spread < 0 else "caution" if current_spread < 0.5 else "healthy"
        return {"level": level, "signal": signal, "current": current_spread}

    def analyze_dxy(self, dxy_data: pd.Series) -> Dict:
        if dxy_data.empty: return {"level": "N/A", "signal": "neutral", "current": np.nan}
        try:
            last_value = dxy_data.iloc[-1]
            current_dxy = float(last_value.iloc[0] if hasattr(last_value, 'iloc') else last_value)
        except (TypeError, IndexError, AttributeError):
            return {"level": "N/A", "signal": "neutral", "current": np.nan}
        level = "FORTE" if current_dxy > 105 else "DEBOLE" if current_dxy < 95 else "NEUTRALE"
        signal = "strong-dollar" if current_dxy > 105 else "weak-dollar" if current_dxy < 95 else "neutral"
        return {"level": level, "signal": signal, "current": current_dxy}
    
    def determine_scenario(self, indicators: Dict) -> str:
        vix_signal = indicators.get('vix', {}).get('signal', 'neutral')
        inflation_signal = indicators.get('inflation', {}).get('signal', 'neutral')
        unemployment_signal = indicators.get('unemployment', {}).get('signal', 'neutral')
        yield_signal = indicators.get('yield_curve', {}).get('signal', 'neutral')
        
        if yield_signal == 'recession-risk' and unemployment_signal in ['weak-economy', 'recession']: return 'RECESSIONE'
        if vix_signal == 'risk-off' and (unemployment_signal in ['weak-economy', 'recession'] or yield_signal == 'recession-risk'): return 'RISK-OFF'
        if inflation_signal == 'high-inflation' and unemployment_signal in ['weak-economy', 'recession']: return 'STAGFLAZIONE'
        if vix_signal == 'risk-on' and unemployment_signal == 'strong-economy': return 'CRESCITA FORTE'
        if indicators.get('dxy', {}).get('signal') == 'strong-dollar': return 'DOLLARO FORTE'
        return 'NEUTRALE'

def create_gauge_chart(value: float, title: str, min_val: float, max_val: float, thresholds: List[float], colors: List[str]) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        gauge={'axis': {'range': [min_val, max_val]}, 'bar': {'color': "darkblue"},
               'steps': [{'range': [min_val, thresholds[0]], 'color': colors[0]},
                         {'range': [thresholds[0], thresholds[1]], 'color': colors[1]},
                         {'range': [thresholds[1], thresholds[2]], 'color': colors[2]},
                         {'range': [thresholds[2], max_val], 'color': colors[3]}]}
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_allocation_comparison(base: Dict, scenario: Dict, scenario_name: str) -> go.Figure:
    df = pd.DataFrame({'Asset': list(base.keys()), 'Base': list(base.values()), 'Scenario': list(scenario.values())})
    fig = px.bar(df, x='Asset', y=['Base', 'Scenario'], barmode='group', title=f'Confronto Allocazioni: Base vs {scenario_name}',
                 labels={'value': 'Allocazione %', 'variable': 'Legenda'}, color_discrete_map={'Base': 'lightblue', 'Scenario': 'darkblue'})
    fig.update_layout(height=500)
    return fig

def create_vix_chart_with_thresholds(history_cache: Dict) -> go.Figure:
    if not history_cache:
        return go.Figure().add_annotation(text="Nessun dato storico disponibile", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    dates = list(history_cache.keys())
    vix_values = [history_cache[date].get('vix', np.nan) for date in dates]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=vix_values, name="VIX", line=dict(color="red", width=2)))
    
    # Soglie VIX
    fig.add_hline(y=15, line_dash="dash", line_color="green", annotation_text="VIX 15 - Soglia Bassa")
    fig.add_hline(y=25, line_dash="dash", line_color="orange", annotation_text="VIX 25 - Soglia Alta")
    fig.add_hline(y=35, line_dash="dash", line_color="red", annotation_text="VIX 35 - Soglia Critica")
    
    fig.update_layout(title="VIX - Evoluzione Storica con Soglie", height=350, yaxis_title="VIX", showlegend=False)
    return fig

def create_dxy_chart_with_thresholds(history_cache: Dict) -> go.Figure:
    if not history_cache:
        return go.Figure().add_annotation(text="Nessun dato storico disponibile", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    dates = list(history_cache.keys())
    dxy_values = [history_cache[date].get('dxy', np.nan) for date in dates]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=dxy_values, name="DXY", line=dict(color="blue", width=2)))
    
    # Soglie DXY
    fig.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="DXY 95 - Dollaro Debole")
    fig.add_hline(y=105, line_dash="dash", line_color="orange", annotation_text="DXY 105 - Dollaro Forte")
    
    fig.update_layout(title="DXY - Evoluzione Storica con Soglie", height=350, yaxis_title="DXY", showlegend=False)
    return fig

@st.cache_data(ttl=3600)
def get_cached_historical_data(fred_api_key: str, lookback_days: int) -> Dict:
    collector = DataCollector(fred_api_key)
    analyzer = ScenarioAnalyzer()
    history_cache = {}
    today = datetime.now()
    
    for i in range(lookback_days, -1, -1):
        current_date = today - timedelta(days=i)
        try:
            vix_data = collector.get_vix_data(current_date, 30)
            dxy_data = collector.get_dxy_data(current_date, 30)
            vix_analysis = analyzer.analyze_vix(vix_data)
            dxy_analysis = analyzer.analyze_dxy(dxy_data)
            
            history_cache[current_date.date()] = {
                'vix': vix_analysis.get('current', np.nan),
                'dxy': dxy_analysis.get('current', np.nan)
            }
        except Exception as e:
            # Se c'è un errore per questa data specifica, continua con NaN
            history_cache[current_date.date()] = {
                'vix': np.nan,
                'dxy': np.nan
            }
        time.sleep(0.05)
    
    return history_cache

@st.cache_data(ttl=3600)
def get_monthly_indicators_data(fred_api_key: str, months_back: int = 24) -> Dict:
    """Raccoglie i dati mensili degli indicatori economici per i grafici di trend"""
    collector = DataCollector(fred_api_key)
    end_date = datetime.now()
    
    try:
        # Raccoglie dati per il periodo specificato (+ buffer per calcoli)
        days_back = months_back * 35  # ~35 giorni per mese con buffer
        cpi_monthly = collector.get_fred_data('CPIAUCSL', end_date, days_back)
        unemployment_data = collector.get_fred_data('UNRATE', end_date, days_back)
        ten_year_data = collector.get_fred_data('GS10', end_date, days_back)
        two_year_data = collector.get_fred_data('GS2', end_date, days_back)
        
        # Calcola inflazione YoY
        cpi_yoy = cpi_monthly.pct_change(12).dropna() * 100 if not cpi_monthly.empty else pd.Series()
        
        # Calcola yield curve spread
        yield_spread = pd.Series(dtype=float)
        if not ten_year_data.empty and not two_year_data.empty:
            common_index = ten_year_data.index.intersection(two_year_data.index)
            if not common_index.empty:
                yield_spread = (ten_year_data.loc[common_index] - two_year_data.loc[common_index]).dropna()
        
        return {
            'inflation': cpi_yoy.tail(months_back) if not cpi_yoy.empty else pd.Series(),
            'unemployment': unemployment_data.tail(months_back) if not unemployment_data.empty else pd.Series(),
            'yield_spread': yield_spread.tail(months_back) if not yield_spread.empty else pd.Series(),
            'ten_year': ten_year_data.tail(months_back) if not ten_year_data.empty else pd.Series(),
            'two_year': two_year_data.tail(months_back) if not two_year_data.empty else pd.Series()
        }
    except Exception as e:
        st.error(f"Errore nel caricamento dei dati mensili: {e}")
        return {}

def create_inflation_chart(inflation_data: pd.Series, months_back: int) -> go.Figure:
    """Crea grafico specifico per l'inflazione con soglie"""
    if inflation_data.empty:
        return go.Figure().add_annotation(
            text="Nessun dato inflazione disponibile", 
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=inflation_data.index,
        y=inflation_data.values,
        name="Inflazione YoY (%)",
        line=dict(color="red", width=3),
        fill='tonexty' if len(inflation_data) > 1 else None,
        fillcolor='rgba(255,0,0,0.1)'
    ))
    
    # Soglie inflazione
    fig.add_hline(y=2, line_dash="dash", line_color="green", 
                  annotation_text="Target 2%", annotation_position="bottom right")
    fig.add_hline(y=3, line_dash="dash", line_color="orange", 
                  annotation_text="Soglia Alta 3%", annotation_position="top right")
    fig.add_hline(y=5, line_dash="dash", line_color="red", 
                  annotation_text="Soglia Critica 5%", annotation_position="top right")
    
    fig.update_layout(
        title=f"Inflazione YoY - Ultimi {months_back} Mesi",
        height=300,
        yaxis_title="Inflazione (%)",
        showlegend=False,
        hovermode='x unified'
    )
    return fig

def create_unemployment_chart(unemployment_data: pd.Series, months_back: int) -> go.Figure:
    """Crea grafico specifico per la disoccupazione con soglie"""
    if unemployment_data.empty:
        return go.Figure().add_annotation(
            text="Nessun dato disoccupazione disponibile", 
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=unemployment_data.index,
        y=unemployment_data.values,
        name="Disoccupazione (%)",
        line=dict(color="orange", width=3),
        fill='tonexty' if len(unemployment_data) > 1 else None,
        fillcolor='rgba(255,165,0,0.1)'
    ))
    
    # Soglie disoccupazione
    fig.add_hline(y=4, line_dash="dash", line_color="green", 
                  annotation_text="Soglia Bassa 4%", annotation_position="bottom right")
    fig.add_hline(y=6, line_dash="dash", line_color="orange", 
                  annotation_text="Soglia Alta 6%", annotation_position="top right")
    fig.add_hline(y=8, line_dash="dash", line_color="red", 
                  annotation_text="Soglia Critica 8%", annotation_position="top right")
    
    fig.update_layout(
        title=f"Tasso di Disoccupazione - Ultimi {months_back} Mesi",
        height=300,
        yaxis_title="Disoccupazione (%)",
        showlegend=False,
        hovermode='x unified'
    )
    return fig

def create_yield_spread_chart(yield_spread: pd.Series, months_back: int) -> go.Figure:
    """Crea grafico specifico per lo yield spread con soglie"""
    if yield_spread.empty:
        return go.Figure().add_annotation(
            text="Nessun dato yield spread disponibile", 
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yield_spread.index,
        y=yield_spread.values,
        name="Yield Spread 10Y-2Y (%)",
        line=dict(color="blue", width=3),
        fill='tonexty' if len(yield_spread) > 1 else None,
        fillcolor='rgba(0,0,255,0.1)'
    ))
    
    # Soglie yield spread
    fig.add_hline(y=0, line_dash="solid", line_color="red", line_width=2,
                  annotation_text="Curva Invertita 0%", annotation_position="bottom right")
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange", 
                  annotation_text="Curva Piatta 0.5%", annotation_position="top right")
    fig.add_hline(y=1, line_dash="dash", line_color="green", 
                  annotation_text="Curva Normale 1%", annotation_position="top right")
    
    # Evidenzia zona di inversione
    fig.add_hrect(y0=-2, y1=0, fillcolor="red", opacity=0.1, line_width=0)
    
    fig.update_layout(
        title=f"Yield Spread 10Y-2Y - Ultimi {months_back} Mesi",
        height=300,
        yaxis_title="Spread (%)",
        showlegend=False,
        hovermode='x unified'
    )
    return fig

def create_treasury_rates_chart(ten_year: pd.Series, two_year: pd.Series, months_back: int) -> go.Figure:
    """Crea grafico per i tassi dei treasury"""
    if ten_year.empty and two_year.empty:
        return go.Figure().add_annotation(
            text="Nessun dato treasury disponibile", 
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    fig = go.Figure()
    
    if not ten_year.empty:
        fig.add_trace(go.Scatter(
            x=ten_year.index,
            y=ten_year.values,
            name="Treasury 10Y",
            line=dict(color="darkblue", width=2)
        ))
    
    if not two_year.empty:
        fig.add_trace(go.Scatter(
            x=two_year.index,
            y=two_year.values,
            name="Treasury 2Y",
            line=dict(color="lightblue", width=2)
        ))
    
    fig.update_layout(
        title=f"Tassi Treasury - Ultimi {months_back} Mesi",
        height=300,
        yaxis_title="Tasso (%)",
        legend=dict(x=0.01, y=0.99),
        hovermode='x unified'
    )
    return fig

def get_current_data(collector: DataCollector, analyzer: ScenarioAnalyzer, lookback_days: int):
    end_date = datetime.now()
    vix_data = collector.get_vix_data(end_date, lookback_days)
    cpi_monthly = collector.get_fred_data('CPIAUCSL', end_date, 730)
    unemployment_data = collector.get_fred_data('UNRATE', end_date, 730)
    ten_year_data = collector.get_fred_data('GS10', end_date, lookback_days)
    two_year_data = collector.get_fred_data('GS2', end_date, lookback_days)
    dxy_data = collector.get_dxy_data(end_date, lookback_days)
    
    cpi_yoy = cpi_monthly.pct_change(12).dropna() * 100 if not cpi_monthly.empty else pd.Series()

    indicators = {
        'vix': analyzer.analyze_vix(vix_data),
        'inflation': analyzer.analyze_inflation(cpi_yoy),
        'unemployment': analyzer.analyze_unemployment(unemployment_data),
        'yield_curve': analyzer.analyze_yield_curve(ten_year_data, two_year_data),
        'dxy': analyzer.analyze_dxy(dxy_data)
    }
    current_scenario = analyzer.determine_scenario(indicators)
    return indicators, current_scenario

def main():
    #st.title("🎯 PAC Dynamic Rebalancing Dashboard")
    #st.markdown("---")
    
    with st.sidebar:
        st.header("⚙️ Configurazione")
        fred_api_key = st.text_input("FRED API Key", type="password", help="Inserisci la tua chiave API FRED")
        if not fred_api_key:
            st.warning("⚠️ Inserisci la chiave API FRED per continuare")
            st.stop()
        
        st.markdown("---")
        st.header("🕰️ Parametri Storici")
        history_lookback = st.number_input("Giorni di storico VIX & DXY", min_value=5, max_value=90, value=30, step=5)
        monthly_lookback = st.number_input("Mesi indicatori economici", min_value=6, max_value=60, value=24, step=6)
        
        if st.button("🔄 Aggiorna Cache"):
            st.cache_data.clear()
            st.success("Cache pulita!")
            st.rerun()

    collector = DataCollector(fred_api_key)
    analyzer = ScenarioAnalyzer()
    
    with st.spinner("📡 Analisi dati correnti..."):
        indicators, current_scenario = get_current_data(collector, analyzer, 90)
    
    # Layout compatto con scenario e indicatori integrati
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("#### 🎯 Scenario")
        scenario_colors = {'NEUTRALE': 'blue', 'RISK-OFF': 'red', 'STAGFLAZIONE': 'orange', 'CRESCITA FORTE': 'green', 'DOLLARO FORTE': 'purple', 'RECESSIONE': 'darkred'}
        st.markdown(f"<h3 style='color: {scenario_colors.get(current_scenario, 'blue')};'>{current_scenario}</h3>", unsafe_allow_html=True)
        
        st.markdown("#### 📊 Indicatori")
        
        # VIX con soglie integrate
        vix_data = indicators['vix']
        if pd.notna(vix_data.get('current', 0)):
            level_color = {'CRITICO': '🔴', 'ALTO': '🟠', 'MEDIO': '🟡', 'BASSO': '🟢'}.get(vix_data['level'], '⚪')
            st.metric("VIX", f"{vix_data['current']:.1f}", f"{level_color} {vix_data['level']}", delta_color="off")
        
        # Inflazione con soglie integrate
        inf_data = indicators['inflation']
        if pd.notna(inf_data.get('current', 0)):
            level_color = {'CRITICO': '🔴', 'ELEVATO': '🟠', 'TARGET': '🟢', 'BASSO': '🟡'}.get(inf_data['level'], '⚪')
            st.metric("Inflazione", f"{inf_data['current']:.1f}%", f"{level_color} {inf_data['level']}", delta_color="off")
        
        # Disoccupazione con soglie integrate
        unemp_data = indicators['unemployment']
        if pd.notna(unemp_data.get('current', 0)):
            level_color = {'CRITICO': '🔴', 'ELEVATO': '🟠', 'NORMALE': '🟡', 'BASSO': '🟢'}.get(unemp_data['level'], '⚪')
            st.metric("Disoccupazione", f"{unemp_data['current']:.1f}%", f"{level_color} {unemp_data['level']}", delta_color="off")
        
        # Yield Curve con soglie integrate
        yield_data = indicators['yield_curve']
        if pd.notna(yield_data.get('current', 0)):
            level_color = {'INVERTITO': '🔴', 'PIATTO': '🟠', 'NORMALE': '🟢'}.get(yield_data['level'], '⚪')
            st.metric("10Y-2Y", f"{yield_data['current']:.2f}%", f"{level_color} {yield_data['level']}", delta_color="off")
        
        # DXY con soglie integrate
        dxy_data = indicators['dxy']
        if pd.notna(dxy_data.get('current', 0)):
            level_color = {'FORTE': '🟠', 'DEBOLE': '🔴', 'NEUTRALE': '🟡'}.get(dxy_data['level'], '⚪')
            st.metric("DXY", f"{dxy_data['current']:.1f}", f"{level_color} {dxy_data['level']}", delta_color="off")

        # Tendina con soglie e significati
        with st.expander("📋 Soglie"):
            st.markdown("""
            **🔥 VIX (Volatility Index)**
            - 🟢 **BASSO** (<15): Mercati calmi, bassa volatilità attesa
            - 🟡 **MEDIO** (15-25): Volatilità normale
            - 🟠 **ALTO** (25-35): Incertezza elevata, mercati nervosi
            - 🔴 **CRITICO** (>35): Panico di mercato, volatilità estrema
            
            **📈 Inflazione YoY**
            - 🟡 **BASSO** (<2%): Sotto target Fed, possibile deflazione
            - 🟢 **TARGET** (2-3%): Range ottimale per crescita economica
            - 🟠 **ELEVATO** (3-5%): Pressioni inflazionistiche moderate
            - 🔴 **CRITICO** (>5%): Inflazione galoppante, erosione potere d'acquisto
            
            **👥 Disoccupazione**
            - 🟢 **BASSO** (<4%): Piena occupazione, economia forte
            - 🟡 **NORMALE** (4-6%): Livelli sostenibili
            - 🟠 **ELEVATO** (6-8%): Debolezza economica
            - 🔴 **CRITICO** (>8%): Recessione, crisi occupazionale
            
            **📊 Yield Curve (10Y-2Y)**
            - 🔴 **INVERTITO** (<0%): Segnale recessione imminente
            - 🟠 **PIATTO** (0-0.5%): Rallentamento economico
            - 🟢 **NORMALE** (>0.5%): Crescita economica sana
            
            **💵 DXY (Dollar Index)**
            - 🔴 **DEBOLE** (<95): Dollaro sotto pressione
            - 🟡 **NEUTRALE** (95-105): Range equilibrato
            - 🟠 **FORTE** (>105): Dollaro dominante, pressione su asset esteri
            """)
            

    with col2:
        st.markdown("#### 💼 Allocazione Suggerita")
        if current_scenario != 'NEUTRALE':
            # Grafico a torta per l'allocazione dello scenario
            scenario_allocation = analyzer.scenarios[current_scenario]
            fig_pie = px.pie(
                values=list(scenario_allocation.values()),
                names=list(scenario_allocation.keys()),
                title=f"Allocazione Scenario {current_scenario}",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(
                height=400,
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            # Grafico a torta per l'allocazione base
            fig_pie = px.pie(
                values=list(analyzer.base_allocation.values()),
                names=list(analyzer.base_allocation.keys()),
                title="Allocazione Base (Scenario Neutrale)",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(
                height=400,
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            st.info("🟢 Scenario NEUTRALE: Mantieni l'allocazione base.")

    st.markdown("---")
    
    with st.expander("📈 Trend Mensile Indicatori Economici", expanded=False):
        with st.spinner(f"📡 Caricamento indicatori ultimi {monthly_lookback} mesi..."):
            monthly_data = get_monthly_indicators_data(fred_api_key, monthly_lookback)
        
        if monthly_data:
            # Prima riga: Inflazione e Disoccupazione
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_inflation_chart(monthly_data.get('inflation', pd.Series()), monthly_lookback), use_container_width=True)
            with col2:
                st.plotly_chart(create_unemployment_chart(monthly_data.get('unemployment', pd.Series()), monthly_lookback), use_container_width=True)
            
            # Seconda riga: Yield Spread e Treasury Rates
            col3, col4 = st.columns(2)
            with col3:
                st.plotly_chart(create_yield_spread_chart(monthly_data.get('yield_spread', pd.Series()), monthly_lookback), use_container_width=True)
            with col4:
                st.plotly_chart(create_treasury_rates_chart(monthly_data.get('ten_year', pd.Series()), monthly_data.get('two_year', pd.Series()), monthly_lookback), use_container_width=True)
        else:
            st.warning("⚠️ Nessun dato mensile disponibile")

    st.markdown("---")
    with st.expander("📈 Storico VIX & DXY", expanded=False):
        with st.spinner(f"📡 Caricamento storico di {history_lookback} giorni..."):
            history_cache = get_cached_historical_data(fred_api_key, history_lookback)
        
        st.plotly_chart(create_vix_chart_with_thresholds(history_cache), use_container_width=True)
        st.plotly_chart(create_dxy_chart_with_thresholds(history_cache), use_container_width=True)

    st.markdown(f"*Ultimo aggiornamento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

if __name__ == "__main__":
    main() 
