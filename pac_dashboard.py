import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import fredapi
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple
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
        self.base_allocation = {'S&P 500': 21, 'Europa': 12, 'Emergenti': 13, 'NASDAQ': 9, 'Berkshire Hathaway': 16, 'Small Cap': 11, 'Oro': 13, 'Bitcoin': 5}
        self.scenarios = {
            'RISK-OFF': {'S&P 500': 16, 'Europa': 10, 'Emergenti': 12, 'NASDAQ': 5, 'Berkshire Hathaway': 18, 'Small Cap': 12, 'Oro': 15, 'Bitcoin': 3},
            'STAGFLAZIONE': {'S&P 500': 18, 'Europa': 12, 'Emergenti': 22, 'NASDAQ': 8, 'Berkshire Hathaway': 12, 'Small Cap': 8, 'Oro': 15, 'Bitcoin': 5},
            'CRESCITA FORTE': {'S&P 500': 22, 'Europa': 15, 'Emergenti': 16, 'NASDAQ': 15, 'Berkshire Hathaway': 8, 'Small Cap': 15, 'Oro': 2, 'Bitcoin': 7},
            'DOLLARO FORTE': {'S&P 500': 22, 'Europa': 8, 'Emergenti': 7, 'NASDAQ': 12, 'Berkshire Hathaway': 18, 'Small Cap': 13, 'Oro': 13, 'Bitcoin': 7},
            'RECESSIONE': {'S&P 500': 13, 'Europa': 15, 'Emergenti': 8, 'NASDAQ': 3, 'Berkshire Hathaway': 25, 'Small Cap': 8, 'Oro': 18, 'Bitcoin': 5},
            'NEUTRALE': {'S&P 500': 21, 'Europa': 12, 'Emergenti': 13, 'NASDAQ': 9, 'Berkshire Hathaway': 16, 'Small Cap': 11, 'Oro': 13, 'Bitcoin': 5}
        }
    
    def analyze_vix(self, vix_data: pd.Series) -> Dict:
        if vix_data.empty: return {"level": "N/A", "signal": "neutral", "current": np.nan}
        try:
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
    
    def calculate_scenario_scores(self, indicators: Dict) -> Dict[str, float]:
        """
        Calcola i punteggi per ogni scenario basandosi sugli indicatori
        Ogni scenario può ottenere punti da 0 a 100
        """
        vix_signal = indicators.get('vix', {}).get('signal', 'neutral')
        inflation_signal = indicators.get('inflation', {}).get('signal', 'neutral')
        unemployment_signal = indicators.get('unemployment', {}).get('signal', 'neutral')
        yield_signal = indicators.get('yield_curve', {}).get('signal', 'neutral')
        dxy_signal = indicators.get('dxy', {}).get('signal', 'neutral')
        
        # Ottieni i valori numerici degli indicatori per punteggi graduali
        vix_value = indicators.get('vix', {}).get('current', 20)
        inflation_value = indicators.get('inflation', {}).get('current', 2.5)
        unemployment_value = indicators.get('unemployment', {}).get('current', 4)
        yield_value = indicators.get('yield_curve', {}).get('current', 1)
        dxy_value = indicators.get('dxy', {}).get('current', 100)
        
        scores = {}
        
        # === SCENARIO RECESSIONE ===
        recession_score = 0
        # Yield curve invertita è importante ma non dominante
        if yield_signal == 'recession-risk':
            recession_score += 35
        elif yield_signal == 'caution':
            recession_score += 18
        
        # Disoccupazione alta - indicatore più diretto
        if unemployment_signal == 'recession':
            recession_score += 35
        elif unemployment_signal == 'weak-economy':
            recession_score += 18
        
        # VIX alto (paura)
        if vix_signal == 'risk-off':
            recession_score += 15
        elif vix_value > 20:
            recession_score += 8
        
        # Inflazione bassa (deflazione)
        if inflation_signal == 'low-inflation':
            recession_score += 15
        
        scores['RECESSIONE'] = min(100, recession_score)
        
        # === SCENARIO RISK-OFF ===
        risk_off_score = 0
        # VIX molto alto è il segnale principale
        if vix_signal == 'risk-off':
            risk_off_score += 40
        elif vix_value > 20:
            risk_off_score += 20
        
        # Problemi economici generali
        if unemployment_signal in ['weak-economy', 'recession']:
            risk_off_score += 25
        elif unemployment_value > 5:
            risk_off_score += 12
        
        # Yield curve problematica
        if yield_signal in ['recession-risk', 'caution']:
            risk_off_score += 20
        
        # Inflazione alta o bassa (instabilità)
        if inflation_signal in ['high-inflation', 'low-inflation']:
            risk_off_score += 15
        
        scores['RISK-OFF'] = min(100, risk_off_score)
        
        # === SCENARIO STAGFLAZIONE ===
        stagflation_score = 0
        # Inflazione alta è il segnale principale
        if inflation_signal == 'high-inflation':
            stagflation_score += 40
        elif inflation_value > 3:
            stagflation_score += 20
        
        # Disoccupazione alta o crescente
        if unemployment_signal in ['weak-economy', 'recession']:
            stagflation_score += 30
        elif unemployment_value > 5:
            stagflation_score += 15
        
        # VIX moderatamente alto (incertezza)
        if vix_value > 18:
            stagflation_score += 15
        
        # Yield curve piatta/invertita
        if yield_signal in ['recession-risk', 'caution']:
            stagflation_score += 15
        
        scores['STAGFLAZIONE'] = min(100, stagflation_score)
        
        # === SCENARIO CRESCITA FORTE ===
        growth_score = 0
        # Bassa disoccupazione è il segnale principale
        if unemployment_signal == 'strong-economy':
            growth_score += 35
        elif unemployment_value < 4.5:
            growth_score += 18
        
        # VIX basso (fiducia)
        if vix_signal == 'risk-on':
            growth_score += 30
        elif vix_value < 18:
            growth_score += 15
        
        # Yield curve normale/sana - più importante per crescita
        if yield_signal == 'healthy':
            growth_score += 25
        elif yield_value > 0.5:
            growth_score += 12
        
        # Inflazione sotto controllo ma non troppo bassa
        if inflation_signal == 'neutral':
            growth_score += 10
        elif 2 <= inflation_value <= 3:
            growth_score += 5
        
        scores['CRESCITA FORTE'] = min(100, growth_score)
        
        # === SCENARIO DOLLARO FORTE ===
        strong_dollar_score = 0
        # DXY alto è il segnale principale - ancora più dominante
        if dxy_signal == 'strong-dollar':
            strong_dollar_score += 50
        elif dxy_value > 102:
            strong_dollar_score += 30
        
        # Economia USA relativamente forte
        if unemployment_signal == 'strong-economy':
            strong_dollar_score += 15
        elif unemployment_value < 5:
            strong_dollar_score += 8
        
        # VIX controllato
        if vix_value < 20:
            strong_dollar_score += 15
        
        # Yield alti (attrattività dollaro)
        if yield_value > 1:
            strong_dollar_score += 10
        
        # Inflazione sotto controllo
        if inflation_value < 4:
            strong_dollar_score += 10
        
        scores['DOLLARO FORTE'] = min(100, strong_dollar_score)
        
        # === SCENARIO NEUTRALE ===
        # Il punteggio neutrale è inversamente correlato agli altri scenari
        max_other_score = max([scores[s] for s in scores.keys()])
        neutral_base = 60  # Punteggio base per neutralità
        
        # Penalizza se ci sono segnali forti in altre direzioni
        neutral_penalty = max_other_score * 0.6
        neutral_score = max(10, neutral_base - neutral_penalty)
        
        # Bonus se tutti gli indicatori sono in range normale
        normal_indicators = 0
        if 15 <= vix_value <= 25: normal_indicators += 1
        if 2 <= inflation_value <= 3: normal_indicators += 1
        if 4 <= unemployment_value <= 6: normal_indicators += 1
        if 0.5 <= yield_value <= 2: normal_indicators += 1
        if 95 <= dxy_value <= 105: normal_indicators += 1
        
        neutral_bonus = normal_indicators * 8
        neutral_score += neutral_bonus
        
        scores['NEUTRALE'] = min(100, neutral_score)
        
        return scores
    
    def get_scenario_probabilities(self, indicators: Dict) -> List[Tuple[str, float, float]]:
        """
        Restituisce una lista di tuple (scenario, score, probability) ordinata per probabilità
        """
        scores = self.calculate_scenario_scores(indicators)
        
        # Normalizza i punteggi in probabilità (somma = 100%)
        total_score = sum(scores.values())
        if total_score == 0:
            # Se tutti i punteggi sono 0, assegna probabilità uniforme
            probabilities = {scenario: 100/len(scores) for scenario in scores}
        else:
            probabilities = {scenario: (score/total_score)*100 for scenario, score in scores.items()}
        
        # Ordina per probabilità decrescente
        sorted_scenarios = sorted(
            [(scenario, scores[scenario], probabilities[scenario]) 
             for scenario in scores.keys()], 
            key=lambda x: x[2], 
            reverse=True
        )
        
        return sorted_scenarios
    
    def get_recommended_allocation(self, scenario_probabilities: List[Tuple[str, float, float]]) -> Dict[str, float]:
        """
        Calcola l'allocazione consigliata basata sulla media pesata degli scenari
        """
        weighted_allocation = {}
        
        # Inizializza con tutti gli asset a 0
        all_assets = set()
        for scenario_name in self.scenarios:
            all_assets.update(self.scenarios[scenario_name].keys())
        
        for asset in all_assets:
            weighted_allocation[asset] = 0
        
        # Calcola la media pesata
        for scenario, score, probability in scenario_probabilities:
            weight = probability / 100  # Converti in decimale
            scenario_allocation = self.scenarios[scenario]
            
            for asset, allocation in scenario_allocation.items():
                weighted_allocation[asset] += allocation * weight
        
        # Arrotonda e normalizza per assicurare che la somma sia 100%
        for asset in weighted_allocation:
            weighted_allocation[asset] = round(weighted_allocation[asset], 1)
        
        # Correzione per errori di arrotondamento
        total = sum(weighted_allocation.values())
        if total != 100:
            # Aggiusta l'asset con l'allocazione maggiore
            max_asset = max(weighted_allocation.keys(), key=lambda k: weighted_allocation[k])
            weighted_allocation[max_asset] += round(100 - total, 1)
        
        return weighted_allocation

def create_scenario_probability_chart(scenario_probabilities: List[Tuple[str, float, float]], indicators: Dict) -> go.Figure:
    """Crea un grafico a barre orizzontali impilate per le probabilità degli scenari con contributi degli indicatori"""
    scenarios = [item[0] for item in scenario_probabilities]
    probabilities = [item[2] for item in scenario_probabilities]
    
    # Colori per gli indicatori
    indicator_colors = {
        'VIX': '#FF6B6B',        # Rosso
        'Inflazione': '#4ECDC4',  # Turchese
        'Disoccupazione': '#45B7D1',  # Blu
        'Yield Curve': '#96CEB4',     # Verde
        'DXY': '#FECA57',        # Giallo
        'Base/Neutral': '#95A5A6' # Grigio
    }
    
    # Calcola i contributi dettagliati per ogni scenario
    analyzer = ScenarioAnalyzer()
    detailed_scores = []
    
    for scenario, total_score, probability in scenario_probabilities:
        contributions = calculate_indicator_contributions(scenario, indicators)
        detailed_scores.append(contributions)
    
    fig = go.Figure()
    
    # Crea le barre impilate per ogni indicatore
    for indicator in ['VIX', 'Inflazione', 'Disoccupazione', 'Yield Curve', 'DXY', 'Base/Neutral']:
        indicator_values = []
        
        for i, scenario in enumerate(scenarios):
            contribution = detailed_scores[i].get(indicator, 0)
            # Converti il contributo in percentuale della probabilità totale
            indicator_values.append(contribution * probabilities[i] / 100)
        
        fig.add_trace(go.Bar(
            name=indicator,
            y=scenarios,
            x=indicator_values,
            orientation='h',
            marker_color=indicator_colors[indicator],
            hovertemplate=f'<b>{indicator}</b><br>' +
                         'Scenario: %{y}<br>' +
                         'Contributo: %{x:.1f}%<extra></extra>'
        ))
    
    # Aggiungi etichette con probabilità totali
    for i, (scenario, _, probability) in enumerate(scenario_probabilities):
        fig.add_annotation(
            x=probability + 2,
            y=i,
            text=f'{probability:.1f}%',
            showarrow=False,
            font=dict(color='black', size=12, family='Arial Black'),
            xanchor='left'
        )
    
    fig.update_layout(
        title={
            'text': '🎯 Probabilità Scenari di Mercato - Contributi per Indicatore',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'family': 'Arial Black'}
        },
        xaxis_title='Probabilità (%)',
        yaxis_title='',  # Rimuovi il titolo dell'asse Y per evitare sovrapposizioni
        height=450,
        margin=dict(l=140, r=70, t=80, b=50),  # Aumenta margine sinistro per i nomi scenari
        xaxis=dict(range=[0, max(probabilities) * 1.2]),
        barmode='stack',
        legend=dict(
            orientation='v',  # Cambia da orizzontale a verticale
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.02,  # Sposta la legenda a destra del grafico
            font=dict(size=10)
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            tickfont=dict(size=11),  # Font più piccolo per i nomi degli scenari
            automargin=True
        )
    )
    
    return fig

def calculate_indicator_contributions(scenario: str, indicators: Dict) -> Dict[str, float]:
    """Calcola il contributo percentuale di ogni indicatore al punteggio di uno scenario"""
    vix_value = indicators.get('vix', {}).get('current', 20)
    inflation_value = indicators.get('inflation', {}).get('current', 2.5)
    unemployment_value = indicators.get('unemployment', {}).get('current', 4)
    yield_value = indicators.get('yield_curve', {}).get('current', 1)
    dxy_value = indicators.get('dxy', {}).get('current', 100)
    
    vix_signal = indicators.get('vix', {}).get('signal', 'neutral')
    inflation_signal = indicators.get('inflation', {}).get('signal', 'neutral')
    unemployment_signal = indicators.get('unemployment', {}).get('signal', 'neutral')
    yield_signal = indicators.get('yield_curve', {}).get('signal', 'neutral')
    dxy_signal = indicators.get('dxy', {}).get('signal', 'neutral')
    
    contributions = {'VIX': 0, 'Inflazione': 0, 'Disoccupazione': 0, 'Yield Curve': 0, 'DXY': 0, 'Base/Neutral': 0}
    
    if scenario == 'RECESSIONE':
        if yield_signal == 'recession-risk':
            contributions['Yield Curve'] = 35
        elif yield_signal == 'caution':
            contributions['Yield Curve'] = 18
        
        if unemployment_signal == 'recession':
            contributions['Disoccupazione'] = 35
        elif unemployment_signal == 'weak-economy':
            contributions['Disoccupazione'] = 18
        
        if vix_signal == 'risk-off':
            contributions['VIX'] = 15
        elif vix_value > 20:
            contributions['VIX'] = 8
        
        if inflation_signal == 'low-inflation':
            contributions['Inflazione'] = 15
    
    elif scenario == 'RISK-OFF':
        if vix_signal == 'risk-off':
            contributions['VIX'] = 40
        elif vix_value > 20:
            contributions['VIX'] = 20
        
        if unemployment_signal in ['weak-economy', 'recession']:
            contributions['Disoccupazione'] = 25
        elif unemployment_value > 5:
            contributions['Disoccupazione'] = 12
        
        if yield_signal in ['recession-risk', 'caution']:
            contributions['Yield Curve'] = 20
        
        if inflation_signal in ['high-inflation', 'low-inflation']:
            contributions['Inflazione'] = 15
    
    elif scenario == 'STAGFLAZIONE':
        if inflation_signal == 'high-inflation':
            contributions['Inflazione'] = 40
        elif inflation_value > 3:
            contributions['Inflazione'] = 20
        
        if unemployment_signal in ['weak-economy', 'recession']:
            contributions['Disoccupazione'] = 30
        elif unemployment_value > 5:
            contributions['Disoccupazione'] = 15
        
        if vix_value > 18:
            contributions['VIX'] = 15
        
        if yield_signal in ['recession-risk', 'caution']:
            contributions['Yield Curve'] = 15
    
    elif scenario == 'CRESCITA FORTE':
        if unemployment_signal == 'strong-economy':
            contributions['Disoccupazione'] = 35
        elif unemployment_value < 4.5:
            contributions['Disoccupazione'] = 18
        
        if vix_signal == 'risk-on':
            contributions['VIX'] = 30
        elif vix_value < 18:
            contributions['VIX'] = 15
        
        if yield_signal == 'healthy':
            contributions['Yield Curve'] = 25
        elif yield_value > 0.5:
            contributions['Yield Curve'] = 12
        
        if inflation_signal == 'neutral':
            contributions['Inflazione'] = 10
        elif 2 <= inflation_value <= 3:
            contributions['Inflazione'] = 5
    
    elif scenario == 'DOLLARO FORTE':
        if dxy_signal == 'strong-dollar':
            contributions['DXY'] = 50
        elif dxy_value > 102:
            contributions['DXY'] = 30
        
        if unemployment_signal == 'strong-economy':
            contributions['Disoccupazione'] = 15
        elif unemployment_value < 5:
            contributions['Disoccupazione'] = 8
        
        if vix_value < 20:
            contributions['VIX'] = 15
        
        if yield_value > 1:
            contributions['Yield Curve'] = 10
        
        if inflation_value < 4:
            contributions['Inflazione'] = 10
    
    elif scenario == 'NEUTRALE':
        # Per il neutrale, distribuiamo il punteggio in modo più uniforme
        total_neutral = 60  # Punteggio base neutrale
        contributions['Base/Neutral'] = total_neutral
        
        # Bonus per indicatori normali
        normal_count = 0
        if 15 <= vix_value <= 25: 
            contributions['VIX'] = 8
            normal_count += 1
        if 2 <= inflation_value <= 3: 
            contributions['Inflazione'] = 8
            normal_count += 1
        if 4 <= unemployment_value <= 6: 
            contributions['Disoccupazione'] = 8
            normal_count += 1
        if 0.5 <= yield_value <= 2: 
            contributions['Yield Curve'] = 8
            normal_count += 1
        if 95 <= dxy_value <= 105: 
            contributions['DXY'] = 8
            normal_count += 1
    
    return contributions

def create_allocation_pie_chart(allocation: Dict[str, float], title: str) -> go.Figure:
    """Crea un grafico a torta per l'allocazione"""
    fig = px.pie(
        values=list(allocation.values()),
        names=list(allocation.keys()),
        title=title,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        textfont=dict(size=10)
    )
    fig.update_layout(
        height=400,
        showlegend=True,
        legend=dict(
            orientation="v", 
            yanchor="middle", 
            y=0.5, 
            xanchor="left", 
            x=1.05,
            font=dict(size=10)
        ),
        title=dict(
            font=dict(size=16),
            x=0.5,
            xanchor='center'
        )
    )
    return fig

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
    
    scenario_probabilities = analyzer.get_scenario_probabilities(indicators)
    recommended_allocation = analyzer.get_recommended_allocation(scenario_probabilities)
    
    return indicators, scenario_probabilities, recommended_allocation

def main():
    st.title("🎯 PAC Dynamic Rebalancing Dashboard")
    st.markdown("---")
    
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
        indicators, scenario_probabilities, recommended_allocation = get_current_data(collector, analyzer, 90)
    
    # Layout principale con scenari e allocazione
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 🎯 Probabilità Scenari")
        st.plotly_chart(create_scenario_probability_chart(scenario_probabilities, indicators), use_container_width=True)

    with col2:
        st.markdown("#### 💼 Allocazione Consigliata")
        st.plotly_chart(
            create_allocation_pie_chart(recommended_allocation, "Allocazione Ottimizzata"), 
            use_container_width=True
        )

    st.markdown("---")
    
    # Sezione indicatori compatta
    st.markdown("#### 📊 Indicatori Economici Attuali")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        vix_data = indicators['vix']
        if pd.notna(vix_data.get('current', 0)):
            level_color = {'CRITICO': '🔴', 'ALTO': '🟠', 'MEDIO': '🟡', 'BASSO': '🟢'}.get(vix_data['level'], '⚪')
            st.metric("VIX", f"{vix_data['current']:.1f}", f"{level_color} {vix_data['level']}", delta_color="off")
    
    with col2:
        inf_data = indicators['inflation']
        if pd.notna(inf_data.get('current', 0)):
            level_color = {'CRITICO': '🔴', 'ELEVATO': '🟠', 'TARGET': '🟢', 'BASSO': '🟡'}.get(inf_data['level'], '⚪')
            st.metric("Inflazione", f"{inf_data['current']:.1f}%", f"{level_color} {inf_data['level']}", delta_color="off")
    
    with col3:
        unemp_data = indicators['unemployment']
        if pd.notna(unemp_data.get('current', 0)):
            level_color = {'CRITICO': '🔴', 'ELEVATO': '🟠', 'NORMALE': '🟡', 'BASSO': '🟢'}.get(unemp_data['level'], '⚪')
            st.metric("Disoccupazione", f"{unemp_data['current']:.1f}%", f"{level_color} {unemp_data['level']}", delta_color="off")
    
    with col4:
        yield_data = indicators['yield_curve']
        if pd.notna(yield_data.get('current', 0)):
            level_color = {'INVERTITO': '🔴', 'PIATTO': '🟠', 'NORMALE': '🟢'}.get(yield_data['level'], '⚪')
            st.metric("10Y-2Y", f"{yield_data['current']:.2f}%", f"{level_color} {yield_data['level']}", delta_color="off")
    
    with col5:
        dxy_data = indicators['dxy']
        if pd.notna(dxy_data.get('current', 0)):
            level_color = {'FORTE': '🟠', 'DEBOLE': '🔴', 'NEUTRALE': '🟡'}.get(dxy_data['level'], '⚪')
            st.metric("DXY", f"{dxy_data['current']:.1f}", f"{level_color} {dxy_data['level']}", delta_color="off")

    # Tendina con soglie e significati
    with st.expander("📋 Soglie e Logica di Scoring degli Scenari"):
        st.markdown("""
        **🎯 Sistema di Scoring Multi-Scenario**
        
        Il dashboard utilizza un sistema di scoring avanzato che assegna punti a ciascuno scenario basandosi sui valori degli indicatori economici. Ogni scenario può ottenere un punteggio da 0 a 100, che viene poi convertito in probabilità.
        
        **🔥 VIX (Volatility Index)**
        - 🟢 **BASSO** (<15): Mercati calmi, favorisce CRESCITA FORTE
        - 🟡 **MEDIO** (15-25): Volatilità normale
        - 🟠 **ALTO** (25-35): Favorisce RISK-OFF e RECESSIONE
        - 🔴 **CRITICO** (>35): Forte segnale per RISK-OFF
        
        **📈 Inflazione YoY**
        - 🟡 **BASSO** (<2%): Può indicare RECESSIONE o deflazione
        - 🟢 **TARGET** (2-3%): Favorisce CRESCITA FORTE e NEUTRALE
        - 🟠 **ELEVATO** (3-5%): Segnale principale per STAGFLAZIONE
        - 🔴 **CRITICO** (>5%): Forte indicatore di STAGFLAZIONE
        
        **👥 Disoccupazione**
        - 🟢 **BASSO** (<4%): Segnale principale per CRESCITA FORTE
        - 🟡 **NORMALE** (4-6%): Supporta scenario NEUTRALE
        - 🟠 **ELEVATO** (6-8%): Favorisce STAGFLAZIONE e RISK-OFF
        - 🔴 **CRITICO** (>8%): Forte indicatore di RECESSIONE
        
        **📊 Yield Curve (10Y-2Y)**
        - 🔴 **INVERTITO** (<0%): Segnale principale di RECESSIONE
        - 🟠 **PIATTO** (0-0.5%): Supporta RISK-OFF e cautela
        - 🟢 **NORMALE** (>0.5%): Favorisce CRESCITA FORTE
        
        **💵 DXY (Dollar Index)**
        - 🔴 **DEBOLE** (<95): Riduce probabilità DOLLARO FORTE
        - 🟡 **NEUTRALE** (95-105): Scenario equilibrato
        - 🟠 **FORTE** (>105): Segnale principale per DOLLARO FORTE
        
        **🧮 Logica di Calcolo:**
        - Ogni scenario accumula punti in base alla forza dei segnali
        - I punti vengono normalizzati in probabilità (somma = 100%)
        - L'allocazione finale è una media pesata di tutti gli scenari
        - Il sistema evita decisioni binarie, fornendo transizioni graduali
        """)

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

    # Sezione di confronto scenari
    with st.expander("🔍 Confronto Allocazioni per Scenario", expanded=False):
        st.markdown("##### Allocazioni per ogni scenario puro:")
        
        scenario_tabs = st.tabs(list(analyzer.scenarios.keys()))
        
        for i, (scenario_name, scenario_allocation) in enumerate(analyzer.scenarios.items()):
            with scenario_tabs[i]:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.plotly_chart(
                        create_allocation_pie_chart(scenario_allocation, f"Scenario {scenario_name}"),
                        use_container_width=True
                    )
                
                with col2:
                    # Confronto con allocazione base
                    if scenario_name != 'NEUTRALE':
                        diff_df = pd.DataFrame([
                            {
                                'Asset': asset,
                                'Base': analyzer.base_allocation.get(asset, 0),
                                'Scenario': scenario_allocation.get(asset, 0),
                                'Differenza': scenario_allocation.get(asset, 0) - analyzer.base_allocation.get(asset, 0)
                            }
                            for asset in set(list(analyzer.base_allocation.keys()) + list(scenario_allocation.keys()))
                        ])
                        diff_df = diff_df.sort_values('Differenza', key=abs, ascending=False)
                        
                        st.markdown(f"**Differenze vs Base:**")
                        for _, row in diff_df.iterrows():
                            if abs(row['Differenza']) > 0.1:  # Solo differenze significative
                                color = "🟢" if row['Differenza'] > 0 else "🔴"
                                st.write(f"{color} {row['Asset']}: {row['Differenza']:+.1f}%")

    st.markdown(f"*Ultimo aggiornamento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

if __name__ == "__main__":
    main()
