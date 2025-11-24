# Agent Notes for `arrhytmia_detection`

Bu repo, PTB-XL ECG veri seti ile aritmi tespiti üzerine Python tabanlı bir derin öğrenme projesi ve bunun üzerine eklenmiş bir GNU Octave/Matlab veri analizi katmanı içerir.

## Genel Yapı

- Python tarafı:
  - `train_model.py`: PTB-XL sinyallerini `wfdb` ile okuyup 1D CNN+LSTM modeli eğitir.
  - `model_test/random_test.py`: Eğitilmiş modeli rastgele örnekler üzerinde test eder.
  - `data_analysis.ipynb`: Pandas + Matplotlib ile veri seti EDA (summary, class distribution, örnek sinyal) yapar.
- Veri:
  - PTB-XL dataset’i `dataset/physionet.org/files/ptb-xl/1.0.3/` altında.
  - Hem `records100` (100 Hz) hem `records500` (500 Hz) kayıtları mevcut.
- Octave/Matlab tarafı:
  - `octave/` klasörü tamamen metadata ve sinyal tabanlı EDA için; model eğitimi burada yapılmıyor.

## Octave Metadata Analizi

Ana fonksiyonlar:

- `octave/ptbxl_read_metadata.m`
  - `ptbxl_database.csv` dosyasını okur (varsayılan path: `../dataset/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv`).
  - Dönüş yapısı:
    - `metadata.patient_id`, `metadata.age`, `metadata.sex`, `metadata.height`, `metadata.weight`
    - `metadata.scp_codes_raw` (string olarak orijinal `scp_codes` kolonu)
    - `metadata.rows`, `metadata.headers`, `metadata.num_records`
  - `pkg load io` gerektirir; `io` paketini Octave’da elle kurmak gerekir.

- `octave/ptbxl_basic_eda.m`
  - Hızlı EDA ve temel grafikler üretir (ve PNG olarak kaydeder):
    - Yaş dağılımı (age ≤ 89 filtresi ile).
    - Cinsiyet dağılımı.
    - Normal / anormal (scp_codes içinde `NORM` var / yok) dağılımı.
    - Normal vs anormal yaş dağılımı.
    - 10 arrhythmia sınıfı (SR, AFIB, STACH, SARRH, PVC, PAC, AFLT, SBRAD, SVTAC, NORM) için kayıt sayıları.
  - En sık SCP kodlarını sadece konsola metin olarak yazar; okunması zor olan bar grafiği bilerek kaldırıldı.
  - Tüm figürler hem ekran figürü hem de `analysis_results/octave_*.png` olarak üretilir.

- `octave/ptbxl_advanced_plots.m`
  - Metadata’dan daha “kompleks” ama hala model bağımsız grafikler üretir:
    - Arrhythmia sınıflarına göre yaş istatistikleri (ortalama ± std, bar+errorbar):
      - Çıktı: `octave_arrhythmia_age_stats.png`
    - Arrhythmia sınıfları için cinsiyet dağılımı (stacked bar):
      - Çıktı: `octave_arrhythmia_sex_distribution.png`
    - Seçilmiş SCP kodları (AFIB, STACH, SARRH, PVC, PAC, AFLT, SBRAD, SVTAC, NORM, LVH) için co-occurrence heatmap:
      - Çıktı: `octave_scp_cooccurrence_heatmap.png`
  - Yardımcı fonksiyonlar:
    - `octave/ptbxl_cooccurrence_matrix.m`
    - `octave/create_or_show_figure.m`

- `octave/ptbxl_data_analysis.m`
  - Python `data_analysis.ipynb`’e benzer, üst düzey bir akış:
    - Age ≤ 89 filtresi ile dataset özetini konsola basar (kayıt/hasta sayısı, yaş aralığı, cinsiyet dağılımı, eksik hücre sayısı, dosya boyutu).
    - Ardından `ptbxl_basic_eda` ve `ptbxl_advanced_plots` fonksiyonlarını sırasıyla çağırır.
  - Tipik kullanım:
    - Proje kökünden:
      - `octave --persist --eval "addpath('octave'); pkg load io; ptbxl_data_analysis"`
    - veya `octave/` klasöründen:
      - `octave --persist --eval "pkg load io; ptbxl_data_analysis"`

## Tasarım Kararları ve Kısıtlar

- Octave tarafında **extra paket bağımlılığını minimumda tutmak** için:
  - `nanmean`/`nanmedian` gibi `statistics` paketine ait fonksiyonlar kullanılmıyor; bunun yerine NaN filtrelenip `mean`/`median` çalıştırılıyor.
  - `boxplot` yerine bar + errorbar (ortalama ± std) kullanılıyor.
  - Histogram için `histogram` yerine Octave’ın yerleşik `hist` fonksiyonu kullanılıyor.
  - SCP kodları bar grafiği kaldırıldı; top 10 SCP kodu sadece konsola yazılıyor (eksende isimler okunmuyordu).
- Tüm grafikler GUI açılamasa bile (WSL/X11 sorunları) kullanılabilsin diye **PNG’ye otomatik kaydediliyor**. Kullanıcı daha sonra `analysis_results/*.png` dosyalarını normal bir görüntüleyici ile açıyor.
- `ptbxl_basic_eda` ve `ptbxl_advanced_plots` sadece metadata kullanıyor; sinyal (ECG waveform) tarafına dokunmuyor. Sinyal işleme için ayrı fonksiyonlar planlandı ama bu oturumda henüz eklenmedi.

## WFDB / Sinyal İşleme ile İlgili Notlar

- Kullanıcı, WFDB C kütüphanesini kaynak koddan kurdu:
  - `WFDB` ortam değişkeni `~/opt/wfdb`’ye işaret ediyor.
  - `wfdb-config --version` çıktısı 10.7.0.
- MATLAB/Octave WFDB Toolbox için:
  - Eski `https://physionet.org/physiotools/matlab/wfdb-app-toolbox.tar.gz` URL’si artık 404 veriyor.
  - Doğru yol: Toolbox zip/tar.gz dosyasını tarayıcıdan indirip WSL’e manuel kopyalamak, sonra:
    - `unzip ... -d ~/opt/wfdb-toolbox`
    - Octave içinde:
      - `addpath(genpath('~/opt/wfdb-toolbox'));`
      - `setenv('WFDBROOT', getenv('WFDB'));`
      - `wfdbloadlib`
  - Bu kurulum tamamlandığında `rdsamp` vb. fonksiyonlar ile PTB-XL sinyalleri okunabilir hale gelecek.
- Gelecek iş için planlanan sinyal fonksiyonları (henüz yok):
  - `ptbxl_plot_12lead(record_path_or_index)` – 12 lead’i 3x4 grid halinde çizmek, başlıkta yaş/cinsiyet/tanı göstermek.
  - `ptbxl_plot_normal_vs_afib` – normal ve AFIB kayıtlarını aynı lead’de karşılaştırmak.
  - Basit filtre ve zoom fonksiyonları (örneğin QRS ve dalga şekillerini vurgulamak).

## Tarz ve Beklenti

- Değişiklikler yaparken:
  - Mevcut Octave fonksiyonlarının isim ve yapılarını koru; yeni analizler için yeni `.m` dosyaları ekle.
  - Mümkünse ekstra Octave paketlerine bağımlı olma (özellikle `statistics` gibi Forge paketleri).
  - Grafikleri hem figure olarak üret, hem `analysis_results` altına PNG kaydet.
  - Gereksiz veya okunması zor grafiklerden kaçın; her grafiğin “yorumlayabileceğin net bir mesajı” olsun (ör. yaş dağılımı, sınıf dengesi, cinsiyet dağılımı, co-occurrence gibi).
- Sinyal tarafına geçerken:
  - Önce birkaç temsilci kayıt için görsel olarak güçlü figürler (12-lead grid, normal vs AFIB/PVC) üretmek öncelikli.
  - Tam otomatik rapor üretimi yerine, iyi seçilmiş 3–5 örnek kayıt üzerinde detaylı grafikler daha değerli.
- Bu repo üzerinde sonraki geliştirmeler için öncelik **GNU Octave / Matlab kodu**dur; Python tarafındaki eğitim/test kodlarına dokunma, yeni analizleri mümkün olduğunca `octave/` altına ekle.

## Gelecek Geliştirmeler (TODO fikirleri)

- Octave dashboard (`ptbxl_dashboard.m`) üzerinde:
  - Yas araligi (min/max) ve cinsiyet (kadin/erkek/tumu) filtrelerini ekleyerek tum grafiklerin filtrelenmis versiyonlarini goster.
  - Secilebilir arrhythmia sinif filtresini (SR, AFIB, PVC, vb.) mevcut yas/cinsiyet ve co-occurrence grafiklerine uygula (dashboard'da temel filtre desteği eklendi, ileri versiyonlar icin genisletilebilir).
  - Grafikten filtreye link: arrhythmia sinif dagilimi grafiginde bir bara tiklandiginda ilgili arrhythmia filtresini otomatik sec (Octave GUI sinirlari nedeniyle henuz saglam degil, ileride daha iyi event destegiyle yeniden denenebilir).
  - "Reset filtre" butonu ekleyerek yas araligini 0-89'a, arrhythmia filtresini "(hepsi)"ne tek adimda dondur (dashboard'da eklendi: "Filtreyi sifirla").
  - Birden fazla arrhythmia sinifini ayni anda secmeye izin veren (multi-select) bir filtre arayuzu ekle (dashboard'da listbox ile temel destek eklendi; co-occurrence ve yas/cinsiyet grafigi bu filtreyi dikkate aliyor).
  - Birden fazla arrhythmia sinifini ayni anda secmeye izin veren (multi-select) bir filtre arayuzu ekle.
- Metadata EDA tarafinda:
  - `patient_id` bazinda hasta ozetleri (kisi basina kayit sayisi, arrhythmia prevalansi) eklendi (`ptbxl_patient_eda.m`), bunu gelistirerek dashboard'a da baglayabilirsin.
  - Eksik verinin (ozellikle weight) demografik dagilimlar uzerindeki etkisi icin konsol ozetleri eklendi; gerekirse bunlar gorsel hale getirilebilir.
- Label analizinde:
  - `scp_statements.csv` ile ust duzey kategori gruplari (MI, STTC, HYPERTROPHY, CONDUCTION, vb.) tanimlama altyapisi eklendi (`ptbxl_scp_categories.m`) ve diagnostic class dagilimi hem ileri grafiklerde hem dashboard'da kullaniliyor; co-occurrence'in kategori bazli versiyonu ileride eklenebilir.
- Sinyal tarafinda (WFDB hazir oldugunda):
  - Secilen kayit icin 12-lead ECG viewer (3x4 grid) ve normal vs AFIB/PVC karsilastirma figurleri ekle.
  - Basit filtre/zoom araclari ile QRS ve dalga sekillerini vurgulayan gorseller uret.
- Raporlama tarafinda:
  - Dashboard uzerinden secili 3-4 kritik grafigi ve ilgili filtre bilgilerini tek bir PDF/PNG kolaj halinde disariya aktaran "snapshot" export modu ekle.
