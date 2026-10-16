import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

bist30 = ["AKBNK.IS", "ALARK.IS", "ASELS.IS", "ASTOR.IS", "BIMAS.IS", "BRSAN.IS", "CIMSA.IS", "CWENE.IS", "DOAS.IS", "EKGYO.IS", "ENKAI.IS", "EREGL.IS", "EUPWR.IS", "FROTO.IS", "GARAN.IS", "GUBRF.IS", "HEKTS.IS", "ISCTR.IS", "KCHOL.IS", "KONTR.IS", "KOZAL.IS", "KRDMD.IS", "ODAS.IS", "PGSUS.IS", "SAHOL.IS", "SASA.IS", "SISE.IS", "TCELL.IS", "THYAO.IS", "TUPRS.IS", "YKBNK.IS"]

def hesapla_rsi(df, periyot):
    temp_df = df.copy()
    delta = temp_df['Close'].diff()
    kazanc = delta.where(delta > 0, 0)
    kayip = -delta.where(delta < 0, 0)
    rs = kazanc.ewm(alpha=1/periyot, adjust=False).mean() / (kayip.ewm(alpha=1/periyot, adjust=False).mean() + 1e-10)
    temp_df['RSI'] = 100 - (100 / (1 + rs))
    temp_df['Sinyal'] = 0
    temp_df.loc[temp_df['RSI'] < 30, 'Sinyal'] = 1 
    temp_df.loc[temp_df['RSI'] > 70, 'Sinyal'] = -1
    return temp_df

def hesapla_macd(df, hizli, yavas, sinyal_periyot=9):
    temp_df = df.copy()
    macd_line = temp_df['Close'].ewm(span=hizli, adjust=False).mean() - temp_df['Close'].ewm(span=yavas, adjust=False).mean()
    signal_line = macd_line.ewm(span=sinyal_periyot, adjust=False).mean()
    temp_df['MACD_Durum'] = np.where(macd_line > signal_line, 1, -1)
    temp_df['Sinyal'] = temp_df['MACD_Durum'].diff() / 2
    return temp_df

def hesapla_supertrend(df, period, multiplier):
    temp_df = df.copy()
    temp_df['High-Low'] = temp_df['High'] - temp_df['Low']
    temp_df['High-PrevClose'] = np.abs(temp_df['High'] - temp_df['Close'].shift(1))
    temp_df['Low-PrevClose'] = np.abs(temp_df['Low'] - temp_df['Close'].shift(1))
    temp_df['TR'] = temp_df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    temp_df['ATR'] = temp_df['TR'].rolling(period).mean()

    hl2 = (temp_df['High'] + temp_df['Low']) / 2
    temp_df['Basic_UB'] = hl2 + (multiplier * temp_df['ATR'])
    temp_df['Basic_LB'] = hl2 - (multiplier * temp_df['ATR'])
    
    final_ub = [temp_df['Basic_UB'].iloc[0] if not pd.isna(temp_df['Basic_UB'].iloc[0]) else 0.0]
    final_lb = [temp_df['Basic_LB'].iloc[0] if not pd.isna(temp_df['Basic_LB'].iloc[0]) else 0.0]
    supertrend = [0.0]; trend = [1]

    for i in range(1, len(temp_df)):
        if pd.isna(temp_df['Basic_UB'].iloc[i]):
            final_ub.append(final_ub[i-1]); final_lb.append(final_lb[i-1]); trend.append(1); supertrend.append(0.0)
            continue
        if temp_df['Basic_UB'].iloc[i] < final_ub[i-1] or temp_df['Close'].iloc[i-1] > final_ub[i-1]:
            final_ub.append(temp_df['Basic_UB'].iloc[i])
        else: final_ub.append(final_ub[i-1])
            
        if temp_df['Basic_LB'].iloc[i] > final_lb[i-1] or temp_df['Close'].iloc[i-1] < final_lb[i-1]:
            final_lb.append(temp_df['Basic_LB'].iloc[i])
        else: final_lb.append(final_lb[i-1])
            
        if supertrend[i-1] == final_ub[i-1] and temp_df['Close'].iloc[i] <= final_ub[i]:
            trend.append(-1); supertrend.append(final_ub[i])
        elif supertrend[i-1] == final_ub[i-1] and temp_df['Close'].iloc[i] > final_ub[i]:
            trend.append(1); supertrend.append(final_lb[i])
        elif supertrend[i-1] == final_lb[i-1] and temp_df['Close'].iloc[i] >= final_lb[i]:
            trend.append(1); supertrend.append(final_lb[i])
        elif supertrend[i-1] == final_lb[i-1] and temp_df['Close'].iloc[i] < final_lb[i]:
            trend.append(-1); supertrend.append(final_ub[i])
        else:
            trend.append(trend[i-1]); supertrend.append(supertrend[i-1])

    temp_df['ST_Trend'] = trend
    temp_df['Sinyal'] = temp_df['ST_Trend'].diff() / 2
    return temp_df

def backtest_trailing_stop(df, trailing_oran):
    baslangic = 10000.0; nakit = baslangic; hisse_adedi = 0; komisyon = 0.002; zirve_fiyat = 0.0
    for _, satir in df.iterrows():
        fiyat = satir['Close']; sinyal = satir['Sinyal']
        if pd.isna(sinyal): continue
        if hisse_adedi > 0:
            if fiyat > zirve_fiyat: zirve_fiyat = fiyat
            if fiyat <= zirve_fiyat * (1 - trailing_oran):
                satis = hisse_adedi * fiyat; nakit = satis - (satis * komisyon); hisse_adedi = 0.0; zirve_fiyat = 0.0
                continue
        if sinyal == 1 and nakit > 0:
            zirve_fiyat = fiyat
            hisse_adedi = (nakit - (nakit * komisyon)) / fiyat
            nakit = 0.0
        elif sinyal == -1 and hisse_adedi > 0:
            satis = hisse_adedi * fiyat; nakit = satis - (satis * komisyon); hisse_adedi = 0.0; zirve_fiyat = 0.0
    son_bakiye = nakit + (hisse_adedi * df['Close'].iloc[-1])
    return (((son_bakiye - baslangic) / baslangic) * 100)

rapor = []
test_stop_oranlari = [0.05, 0.08, 0.12]

for hisse in bist30:
    try:
        df = yf.Ticker(hisse).history(period="2y")
        df.dropna(subset=['Close', 'High', 'Low'], inplace=True)
        if len(df) < 50: continue

        en_iyi_skor = -999; en_iyi_isim = ""; en_iyi_param = ""; en_iyi_stop = ""
        
        for stop_oran in test_stop_oranlari:
            for p in [7, 10, 14]:
                test_df = hesapla_rsi(df, p)
                skor = backtest_trailing_stop(test_df, stop_oran)
                if skor > en_iyi_skor: en_iyi_skor = skor; en_iyi_isim = "RSI"; en_iyi_param = f"Periyot: {p}"; en_iyi_stop = f"%{int(stop_oran*100)}"
            for h, y in [(8, 21), (12, 26)]:
                test_df = hesapla_macd(df, h, y)
                skor = backtest_trailing_stop(test_df, stop_oran)
                if skor > en_iyi_skor: en_iyi_skor = skor; en_iyi_isim = "MACD"; en_iyi_param = f"Hızlı: {h}, Yavaş: {y}"; en_iyi_stop = f"%{int(stop_oran*100)}"
            for p, c in [(10, 3.0), (14, 2.0)]:
                test_df = hesapla_supertrend(df, p, c)
                skor = backtest_trailing_stop(test_df, stop_oran)
                if skor > en_iyi_skor: en_iyi_skor = skor; en_iyi_isim = "SuperTrend"; en_iyi_param = f"Periyot: {p}, Çarpan: {c}"; en_iyi_stop = f"%{int(stop_oran*100)}"

        # Güncel günün sinyalini son satırdan çek
        son_satir = test_df.iloc[-1]
        guncel_sinyal = "TUT / NÖTR ⚪"
        if son_satir['Sinyal'] == 1: guncel_sinyal = "AL 🟢"
        elif son_satir['Sinyal'] == -1: guncel_sinyal = "SAT 🔴"

        # Sadece "AL" veya "SAT" üreten hisseleri listeye ekle
        if guncel_sinyal != "TUT / NÖTR ⚪":
            rapor.append(f"📌 <b>{hisse}</b>\nFiyat: {round(son_satir['Close'], 2)} TL\nStrateji: {en_iyi_isim} ({en_iyi_param}) | İzleyen Stop: {en_iyi_stop}\nDurum: <b>{guncel_sinyal}</b>\n")
            
    except Exception as e:
        pass

# TELEGRAM'A GÖNDERME KISMI
mesaj = "🤖 <b>QuantBIST Günlük Otonom Tarama Sonuçları</b>\n\n"
if rapor:
    mesaj += "<i>Algoritmaların yapay zeka optimizasyonuna göre bugünkü aksiyonlar:</i>\n\n"
    mesaj += "\n".join(rapor)
else:
    mesaj += "<i>Bugün BİST30'da algoritmik bir işlem sinyali (AL/SAT) bulunamadı. Nakitte/Mevcut pozisyonlarda bekliyoruz.</i>"

url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
payload = {"chat_id": TELEGRAM_CHAT_ID, "text": mesaj, "parse_mode": "HTML"}
requests.post(url, json=payload)
