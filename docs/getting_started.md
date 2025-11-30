# Getting Started (MATLAB / Octave)

Bu dal yalnizca MATLAB / GNU Octave akisi icin sadeleştirildi. Tum komutlar proje kok dizininden (`arrhytmia_detection/`) calismalidir.

## Ön koşullar
- MATLAB R2021b+ **veya** GNU Octave 6+ (`io` paketi yüklü)
- PTB-XL veri seti `dataset/physionet.org/files/ptb-xl/1.0.3/` altinda (en azindan `ptbxl_database.csv` ve `records100/` veya `records500/`)
- Opsiyonel: WFDB Toolbox ile `rdsamp` (sinyal onizlemeleri icin)

Klasor yapisi:
```
dataset/
  physionet.org/files/ptb-xl/1.0.3/
    ptbxl_database.csv
    records100/
    records500/
    scp_statements.csv
```
- Dataset farkli bir yerdeyse komutlardan once `export PTBXL_BASE=/diger/yol/ptb-xl/1.0.3` seklinde ortam degiskeni tanimlayabilirsiniz.
- Komutlar, yukaridaki yol veya PTBXL_BASE ayari yoksa `ptbxl_autodetect_base` ile otomatik olarak dataset dizinini tarar ve eksik (ptbxl_database.csv, scp_statements.csv, records100/records500) ogeleri ayrintili raporlar.

Octave kullaniyorsaniz `io` paketini ilk seferde kurun:
```octave
pkg install -forge io
pkg load io
```

## Hızlı analiz (metadata)
```bash
octave --persist --eval "addpath('octave'); pkg load io; ptbxl_data_analysis"
```
- Konsola veri ozeti yazar (kayıt/hasta sayısı, yas araligi, cinsiyet dagilimi, eksik degerler).
- `analysis_results/` altina yas dagilimi, cinsiyet, arrhythmia dagilimi ve SCP co-occurrence PNG'leri kaydeder.

## Dashboard + sinyal önizleme
```bash
octave --persist --eval "graphics_toolkit qt; addpath('octave'); pkg load io; ptbxl_dashboard"
```
- Yas/cinsiyet/arrhythmia filtreleriyle interaktif EDA.
- Menu 12'de sinyal onizlemesi icin WFDB Toolbox tercih edilir; yoksa Python `wfdb` modulu varsa otomatik denenir, aksi halde bu ozellik pasif kalir.

## Basit model
```bash
octave --persist --eval "addpath('octave'); pkg load io; ptbxl_simple_model"
```
- Sadece metadata (yas, cinsiyet, boy, kilo) ile `NORM` vs non-`NORM` ayrimi icin lojistik regresyon egitir, dogruluk ve 2x2 confusion matrix raporlar.

## Diger faydali komutlar
- `ptbxl_basic_eda`: temel dagilimlar ve normal/anormal ozetleri.
- `ptbxl_advanced_plots`: arrhythmia yas istatistikleri, cinsiyet dagilimi, SCP co-occurrence heatmap.
- `ptbxl_patient_eda`: hasta bazli kayit sayisi ve arrhythmia prevalansi ozeti.

## Not
- Sunumlar: https://presentations.bariscanatakli.com/ptbxl/ (erisim yoksa yerelden `presentation.html` dosyasini acabilirsiniz; harici font/ikon bagimliligi yok).
- Python egitim/test scriptleri bu dalda kaldirildi; metadata-temelli MATLAB/Octave akislari destekleniyor. Python `wfdb` yalnizca sinyal onizleme icin opsiyonel bir bagimlilik olarak kullaniliyor.
