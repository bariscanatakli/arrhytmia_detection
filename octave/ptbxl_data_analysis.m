function ptbxl_data_analysis(base_directory)
% PTBXL_DATA_ANALYSIS  Python'daki data_analysis.ipynb benzeri bir ozet + EDA calistirir.
%
%   PTBXL_DATA_ANALYSIS()
%   PTBXL_DATA_ANALYSIS(base_directory)
%
%   Yaptigi islemler (konsol + PNG ciktisi):
%     - PTB-XL metadata ozetini yazar (kayit, hasta, yas araligi, cinsiyet dagilimi, eksik deger sayisi, dosya boyutu)
%     - ptbxl_basic_eda fonksiyonunu cagirarak temel dagilim grafiklerini uretir
%     - ptbxl_advanced_plots ile ileri duzey grafikler (boxplot, cinsiyet dagilimi, co-occurrence heatmap) uretir
%
%   Notlar:
%     - Sadece metadata (ptbxl_database.csv) uzerinden calisir; derin ogrenme modelini kullanmaz.
%     - Grafikler `analysis_results/` klasorune PNG olarak kaydedilir.

    if nargin < 1 || isempty(base_directory)
        base_directory = fullfile('..', 'dataset', 'physionet.org', 'files', 'ptb-xl', '1.0.3');
    end

    metadata_file_path = fullfile(base_directory, 'ptbxl_database.csv');

    fprintf('PTB-XL data analysis (GNU Octave) basliyor...\n');
    fprintf('Metadata dosyasi: %s\n\n', metadata_file_path);

    % Metadata'yi oku
    metadata = ptbxl_read_metadata(base_directory);

    % Yas filtresi (Python notebook: df = df[df.age <= 89])
    age_filter_mask = metadata.age <= 89 & ~isnan(metadata.age);

    total_records = sum(age_filter_mask);

    % Unique patient sayisi (NaN olmayanlar icin)
    filtered_patient_ids = metadata.patient_id(age_filter_mask & ~isnan(metadata.patient_id));
    unique_patients = numel(unique(filtered_patient_ids));

    % Yas araligi
    filtered_ages = metadata.age(age_filter_mask);
    min_age_filtered = min(filtered_ages);
    max_age_filtered = max(filtered_ages);

    % Cinsiyet dagilimi
    filtered_sex = metadata.sex(age_filter_mask & ~isnan(metadata.sex));
    num_male = sum(filtered_sex == 1);
    num_female = sum(filtered_sex == 0);

    % Dosya boyutu (MB)
    file_info = dir(metadata_file_path);
    if ~isempty(file_info)
        data_size_mb = file_info.bytes / (1024 * 1024);
    else
        data_size_mb = NaN;
    end

    % Eksik deger sayisi (tum hucreler uzerinden basit sayim)
    all_cells = metadata.rows;
    missing_count = 0;
    [num_rows, num_cols] = size(all_cells);
    for r_index = 1:num_rows
        for c_index = 1:num_cols
            if isempty(all_cells{r_index, c_index})
                missing_count = missing_count + 1;
            end
        end
    end

    % Ozet rapor (Python Cell 3'e benzer)
    fprintf('--- PTB-XL Dataset Ozeti (age <= 89 filtresi ile) ---\n');
    fprintf('Toplam Kayit        : %d\n', total_records);
    fprintf('Esiz Hasta Sayisi   : %d\n', unique_patients);
    fprintf('Yas Araligi         : %.0f-%.0f\n', min_age_filtered, max_age_filtered);
    fprintf('Cinsiyet Dagilimi   : Erkek=%d, Kadin=%d\n', num_male, num_female);
    fprintf('Kayit Suresi        : 10 saniye\n');
    fprintf('Ornekleme Hizlari   : 500 Hz (orijinal), 100 Hz (downsampled)\n');
    fprintf('Lead Sayisi         : 12\n');
    if ~isnan(data_size_mb)
        fprintf('CSV Dosya Boyutu    : %.2f MB\n', data_size_mb);
    else
        fprintf('CSV Dosya Boyutu    : (hesaplanamadi)\n');
    end
    fprintf('Eksik Hucre Sayisi  : %d\n', missing_count);
    fprintf('Ogrenci             : Baris Can Atakli - 210717014\n');
    fprintf('-----------------------------------------------------\n\n');

    % Ayrintili EDA ve grafikler
    fprintf('Temel dagilim grafikleri icin ptbxl_basic_eda cagiriliyor...\n');
    ptbxl_basic_eda(base_directory);

    fprintf('\nHasta-bazli ve missingness EDA icin ptbxl_patient_eda cagiriliyor...\n');
    ptbxl_patient_eda(base_directory);

    fprintf('\nIleri duzey grafikler icin ptbxl_advanced_plots cagiriliyor...\n');
    ptbxl_advanced_plots(base_directory);
end
