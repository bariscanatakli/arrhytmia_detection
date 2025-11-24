# Getting Started

Uygulamayi hizli sekilde calistirmak icin gerekli adimlar bu dokumanda ozetlenmistir. Tum komutlar proje kok dizininden (`arrhytmia_detection/`) calismalidir.

## On kosullar
- Python 3.8+ ve `pip`
- PTB-XL veri seti `dataset/physionet.org/files/ptb-xl/1.0.3/` altinda (en azindan `records100/` veya `records500/` klasorleri)
- Opsiyonel: Jupyter Notebook, GNU Octave 6+ (`io` paketi yüklü)

## Kurulum
```bash
pip install -r requirements.txt
```
Veri seti sisteminizde yoksa PhysioNet uzerinden indirip `dataset/physionet.org/files/ptb-xl/1.0.3/` altina cikartin. Yapinin asagidaki gibi oldugunu kontrol edin:
```
dataset/
  physionet.org/files/ptb-xl/1.0.3/
    records100/
    records500/
    ptbxl_database.csv
```

## Model egitme
GPU varsa otomatik kullanilir; aksi halde CPU uzerinde calisir.
```bash
python train_model.py
```
Egitim sonrasi en iyi agirliklar `best_model.keras` dosyasina yazilir.

## Hizi test
Rastgele ornekler uzerinde model testi:
```bash
python model_test/random_test.py
```
Komut, metrikleri ve ornek tahminleri konsola basar.

## Notebook ile EDA
Jupyter kullanarak veri kesif analizi yapabilirsiniz:
```bash
jupyter notebook data_analysis.ipynb
```
Notebook PTB-XL metadata ve ornek sinyal cizimlerini icerir.

## Octave ile metadata analizi (GPU gerekmez)
Octave tarafinda yalnizca metadata kullanilir; sinyal dosyalari okunmaz, bu nedenle hizli calisir. Ilk kez kullaniyorsaniz `io` paketini yukleyin:
```octave
pkg install -forge io
pkg load io
```
Ardindan proje kokunden:
```bash
octave --persist --eval "addpath('octave'); pkg load io; ptbxl_data_analysis"
```
Bu komut konsola veri ozeti yazar ve `analysis_results/` altina PNG grafikler uretir (yas dagilimi, cinsiyet, arrhythmia dagilimi, co-occurrence heatmap vb.).

### Dashboard + sinyal onizleme
Interaktif dashboard icin:
```bash
octave --persist --eval "graphics_toolkit qt; addpath('octave'); pkg load io; ptbxl_dashboard"
```
- WFDB sinyal onizlemesi (menu 12) icin tercih edilen yol WFDB Toolbox + `rdsamp`. Bu yoksa otomatik olarak Python `wfdb` paketi (requirements.txt) ile sinyali okuyup cizer; dataset yolunun dogru oldugundan emin olun.
- Python yolunuzda `python` yoksa `python3` ile de calisir (dashboard otomatik dener). Python tarafinda `wfdb` modulu yoksa `pip3 install wfdb` (veya `pip install -r requirements.txt`) calistirin.
- Menu 12'de opsiyonel olarak Hasta ID kutusuna hasta kimligini girerek dogrudan o hastanin kaydini cizdirebilirsiniz (yas/arrhythmia filtreleri ile birlikte).

## SSS / ipuclari
- Komutlar veri setine dogrudan bagli; dizin farkliysa `train_model.py` ve `octave/*.m` icindeki `dataset` yollarini guncelleyin.
- Egitime baslamadan once yeterli disk alaniniz oldugundan emin olun (PTB-XL tam hali ~5GB).
- Octave GUI acilmiyorsa yine de PNG dosyalari `analysis_results/` altinda olusur.
