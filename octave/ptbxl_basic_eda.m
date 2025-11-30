function ptbxl_basic_eda(base_directory)
% PTBXL_BASIC_EDA  PTB-XL dataseti icin temel veri analizi ve gorsellestirme.
%
%   PTBXL_BASIC_EDA()
%   PTBXL_BASIC_EDA(base_directory)
%
%   Ornek kullanim:
%       cd octave
%       ptbxl_basic_eda
%
%   Bu fonksiyon:
%     - Yas dagilimini inceler (histogram)
%     - Cinsiyet dagilimini hesaplar
%     - Normal / anormal etiket dagilimini tahmini olarak cikartir
%       (scp_codes icinde 'NORM' gecmesine gore)
%
%   Not: Etiketleme, sadece 'NORM' kodunun varligina gore yapilan
%   basit bir tanimdir ve klinik anlamda tam dogru olmayabilir. Ama
%   veri bilimi calismalari icin baslangic noktasi saglar.

    if nargin < 1 || isempty(base_directory)
        base_directory = fullfile('..', 'dataset', 'physionet.org', 'files', 'ptb-xl', '1.0.3');
    end

    fprintf('PTB-XL metadata okunuyor...\n');
    metadata = ptbxl_read_metadata(base_directory);

    fprintf('Toplam kayit sayisi: %d\n', metadata.num_records);

    age_values = metadata.age;
    sex_values = metadata.sex;

    % Basit yas istatistikleri (statistics paketine gerek kalmadan)
    % Python data_analysis.ipynb ile uyumlu olmak icin
    % yaslari 89 ve altina sinirliyoruz (PTB-XL paper'daki gibi).
    valid_age_mask = ~isnan(age_values) & age_values <= 89;
    valid_ages = age_values(valid_age_mask);
    if isempty(valid_ages)
        fprintf('Uyari: yas kolonu tamamen bos veya okunamadi, yas istatistikleri atlandi.\n');
        mean_age_value = NaN;
        median_age_value = NaN;
        min_age_value = NaN;
        max_age_value = NaN;
    else
        mean_age_value = mean(valid_ages);
        median_age_value = median(valid_ages);
        min_age_value = min(valid_ages);
        max_age_value = max(valid_ages);

        fprintf('Yas istatistikleri (NaN hariç):\n');
        fprintf('  Ortalama: %.2f\n', mean_age_value);
        fprintf('  Medyan  : %.2f\n', median_age_value);
        fprintf('  Min     : %.2f\n', min_age_value);
        fprintf('  Max     : %.2f\n', max_age_value);
    end

    % Cinsiyet dagilimi (0: kadin, 1: erkek)
    valid_sex_mask = ~isnan(sex_values);
    female_count = sum(sex_values(valid_sex_mask) == 0);
    male_count = sum(sex_values(valid_sex_mask) == 1);

    fprintf('Cinsiyet dagilimi:\n');
    fprintf('  Kadin: %d\n', female_count);
    fprintf('  Erkek: %d\n', male_count);
    total_valid_sex = female_count + male_count;
    if total_valid_sex > 0
        female_ratio = 100 * female_count / total_valid_sex;
        male_ratio = 100 * male_count / total_valid_sex;
        fprintf('  (Kadin: %.1f %%, Erkek: %.1f %%)\n', female_ratio, male_ratio);
    end

    % Basit normal / anormal ayrimi (scp_codes icinde 'NORM' gecmesine gore)
    normal_mask = build_normal_label_mask(metadata.scp_codes_raw);
    num_normal_records = sum(normal_mask);
    num_abnormal_records = metadata.num_records - num_normal_records;

    fprintf('Etiket dagilimi (yaklasik):\n');
    fprintf('  Normal  : %d\n', num_normal_records);
    fprintf('  Anormal : %d\n', num_abnormal_records);

    % Eksik deger oranlari (age, sex, weight)
    missing_age_ratio = 100 * sum(isnan(metadata.age)) / metadata.num_records;
    missing_sex_ratio = 100 * sum(isnan(metadata.sex)) / metadata.num_records;
    missing_weight_ratio = 100 * sum(isnan(metadata.weight)) / metadata.num_records;

    fprintf('Eksik deger oranlari:\n');
    fprintf('  age   : %.2f %%\n', missing_age_ratio);
    fprintf('  sex   : %.2f %%\n', missing_sex_ratio);
    fprintf('  weight: %.2f %%\n', missing_weight_ratio);

    % Gorsellestirmeler icin cikti klasorunu hazirla
    % Cikti klasoru: repo kokundeki analysis_results
    script_dir = fileparts(mfilename('fullpath'));
    output_dir = fullfile(script_dir, '..', 'analysis_results');
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % Gorsellestirmeler

    % 1) Yas dagilimi
    if ~isempty(valid_ages)
        create_or_show_figure(1);
        hist(valid_ages, 40);
        xlabel('Yas');
        ylabel('Kayit sayisi');
        title('PTB-XL yas dagilimi');
        print(fullfile(output_dir, 'octave_age_distribution.png'), '-dpng');
    end

    % 2) Cinsiyet dagilimi
    create_or_show_figure(2);
    bar([female_count, male_count]);
    set(gca, 'XTickLabel', {'Kadin', 'Erkek'});
    ylabel('Kayit sayisi');
    title('PTB-XL cinsiyet dagilimi');
    print(fullfile(output_dir, 'octave_sex_distribution.png'), '-dpng');

    % 3) Normal / anormal dagilimi
    create_or_show_figure(3);
    bar([num_normal_records, num_abnormal_records]);
    set(gca, 'XTickLabel', {'Normal', 'Anormal'});
    ylabel('Kayit sayisi');
    title('PTB-XL normal / anormal dagilimi (tahmini)');
    print(fullfile(output_dir, 'octave_normal_abnormal_distribution.png'), '-dpng');

    % 4) En sik SCP kodlari (metin olarak, sayi + yüzde)
    [scp_code_list, scp_counts] = ptbxl_label_frequencies(metadata.scp_codes_raw);
    if ~isempty(scp_code_list)
        [sorted_counts, sort_indices] = sort(scp_counts, 'descend');
        sorted_codes = scp_code_list(sort_indices);
        top_k = min(10, numel(sorted_codes));
        top_codes = sorted_codes(1:top_k);
        top_counts = sorted_counts(1:top_k);

        fprintf('En sik SCP kodlari (ilk %d):\n', top_k);
        for code_index = 1:top_k
            percentage = 100 * top_counts(code_index) / metadata.num_records;
            fprintf('  %s: %d kayit (%.2f %%)\n', top_codes{code_index}, top_counts(code_index), percentage);
        end
    end

    % 5) Yas dagilimi: normal vs anormal
    if ~isempty(valid_ages)
        age_normal = age_values(normal_mask & ~isnan(age_values));
        age_abnormal = age_values(~normal_mask & ~isnan(age_values));

        create_or_show_figure(6);
        hist(age_normal, 40);
        hold on;
        hist(age_abnormal, 40);
        hold off;
        legend({'Normal', 'Anormal'});
        xlabel('Yas');
        ylabel('Kayit sayisi');
        title('Yas dagilimi: normal vs anormal');
        print(fullfile(output_dir, 'octave_age_normal_vs_abnormal.png'), '-dpng');
    end

    % 6) Arrhythmia odakli sinif dagilimi (Python modelindeki 10 sinif)
    arrhythmia_classes = {'SR', 'AFIB', 'STACH', 'SARRH', 'PVC', 'PAC', 'AFLT', 'SBRAD', 'SVTAC', 'NORM'};
    arrhythmia_counts = zeros(numel(arrhythmia_classes), 1);
    for c_index = 1:numel(arrhythmia_classes)
        code = arrhythmia_classes{c_index};
        existing_index = find(strcmp(scp_code_list, code));
        if ~isempty(existing_index)
            arrhythmia_counts(c_index) = scp_counts(existing_index);
        else
            arrhythmia_counts(c_index) = 0;
        end
    end

    fprintf('\nArrhythmia odakli sinif dagilimi (Python modelindeki 10 sinif icin kayit sayilari):\n');
    for c_index = 1:numel(arrhythmia_classes)
        percentage = 100 * arrhythmia_counts(c_index) / metadata.num_records;
        fprintf('  %s: %d kayit (%.2f %%)\n', arrhythmia_classes{c_index}, arrhythmia_counts(c_index), percentage);
    end

    create_or_show_figure(7);
    bar(arrhythmia_counts);
    set(gca, 'XTick', 1:numel(arrhythmia_classes), 'XTickLabel', arrhythmia_classes);
    set(gca, 'TickLabelInterpreter', 'none');  % Etiketleri duz metin olarak goster
    xtickangle(45);
    ylabel('Kayit sayisi');
    title('Arrhythmia odakli secilmis SCP kodlarinin dagilimi');
    print(fullfile(output_dir, 'octave_arrhythmia_class_distribution.png'), '-dpng');

    fprintf('\nGorseller, mevcut Octave figur pencerelerinde olusturuldu ve PNG olarak kaydedildi.\n');
    fprintf('Olusan dosyalar: %s/*.png\n', output_dir);
end

function normal_mask = build_normal_label_mask(scp_codes_raw)
% BUILD_NORMAL_LABEL_MASK  scp_codes stringlerine gore normal maskesi uretir.
%
%   Basit bir yaklasim olarak, 'NORM' kodunu iceren kayitlari
%   "normal" kabul ediyoruz.

    num_records = numel(scp_codes_raw);
    normal_mask = false(num_records, 1);

    for record_index = 1:num_records
        codes_string = scp_codes_raw{record_index};
        if isempty(codes_string)
            continue;
        end

        % 'NORM' ifadesini ariyoruz (tek tirnaklar dahil veya hariç olabilir)
        contains_norm = ~isempty(strfind(codes_string, 'NORM')); %#ok<STREMP>
        if contains_norm
            normal_mask(record_index) = true;
        end
    end
end
