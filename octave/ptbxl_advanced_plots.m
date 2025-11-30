function ptbxl_advanced_plots(base_directory)
% PTBXL_ADVANCED_PLOTS  PTB-XL icin ileri duzey grafikler (model olmadan).
%
%   - Arrhythmia siniflari icin yas dagilim boxplot'u
%   - Arrhythmia siniflari icin cinsiyet dagilimi bar grafigi
%   - Secilmis SCP kodlari icin birlikte-gorunme (co-occurrence) heatmap
%
%   Ornek kullanim:
%       cd octave
%       pkg load io
%       ptbxl_advanced_plots

    if nargin < 1 || isempty(base_directory)
        base_directory = fullfile('..', 'dataset', 'physionet.org', 'files', 'ptb-xl', '1.0.3');
    end

    metadata_file_path = fullfile(base_directory, 'ptbxl_database.csv');
    fprintf('Ileri duzey grafikler icin metadata okunuyor: %s\n', metadata_file_path);

    metadata = ptbxl_read_metadata(base_directory);

    % Cikti klasoru: repo kokundeki analysis_results
    script_dir = fileparts(mfilename('fullpath'));
    output_dir = fullfile(script_dir, '..', 'analysis_results');
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % Arrhythmia odakli 10 sinif
    arrhythmia_classes = {'SR', 'AFIB', 'STACH', 'SARRH', 'PVC', 'PAC', 'AFLT', 'SBRAD', 'SVTAC', 'NORM'};

    %% 1) Arrhythmia siniflari icin yas dagilimi (ortalama + std, errorbar)
    fprintf('1) Arrhythmia siniflari icin yas istatistikleri (ortalama + std) olusturuluyor...\n');

    age_values = metadata.age;
    valid_age_mask = ~isnan(age_values) & age_values <= 89;

    class_age_data = cell(numel(arrhythmia_classes), 1);

    pattern = '''([A-Z0-9_/]+)''\s*:\s*([0-9.]+)';

    num_records = metadata.num_records;
    for record_index = 1:num_records
        if ~valid_age_mask(record_index)
            continue;
        end
        codes_string = metadata.scp_codes_raw{record_index};
        if isempty(codes_string)
            continue;
        end
        tokens = regexp(codes_string, pattern, 'tokens');
        if isempty(tokens)
            continue;
        end

        present_codes = {};
        for token_index = 1:numel(tokens)
            token_pair = tokens{token_index};
            code = token_pair{1};
            present_codes{end + 1} = code;
        end

        for class_index = 1:numel(arrhythmia_classes)
            if any(strcmp(present_codes, arrhythmia_classes{class_index}))
                class_age_data{class_index}(end + 1) = age_values(record_index); %#ok<AGROW>
            end
        end
    end

    % Her sinif icin ortalama ve standart sapma
    num_classes = numel(arrhythmia_classes);
    mean_ages = NaN(num_classes, 1);
    std_ages = NaN(num_classes, 1);

    for class_index = 1:num_classes
        ages_for_class = class_age_data{class_index};
        if isempty(ages_for_class)
            continue;
        end
        mean_ages(class_index) = mean(ages_for_class);
        std_ages(class_index) = std(ages_for_class);
    end

    create_or_show_figure(20);
    cla;
    bar(1:num_classes, mean_ages);
    hold on;
    errorbar(1:num_classes, mean_ages, std_ages, '.k');
    hold off;
    set(gca, 'XTick', 1:num_classes, 'XTickLabel', arrhythmia_classes);
    set(gca, 'TickLabelInterpreter', 'none');
    xlabel('Arrhythmia sinifi (SCP kodu)');
    ylabel('Yas (ortalama ± std)');
    title('Arrhythmia siniflarina gore yas istatistikleri');
    print(fullfile(output_dir, 'octave_arrhythmia_age_stats.png'), '-dpng');

    %% 2) Arrhythmia siniflari icin cinsiyet dagilimi
    fprintf('2) Arrhythmia siniflari icin cinsiyet dagilimi bar grafigi olusturuluyor...\n');

    sex_values = metadata.sex;
    valid_sex_mask = ~isnan(sex_values);

    male_counts = zeros(numel(arrhythmia_classes), 1);
    female_counts = zeros(numel(arrhythmia_classes), 1);

    for record_index = 1:num_records
        if ~valid_sex_mask(record_index)
            continue;
        end
        codes_string = metadata.scp_codes_raw{record_index};
        if isempty(codes_string)
            continue;
        end
        tokens = regexp(codes_string, pattern, 'tokens');
        if isempty(tokens)
            continue;
        end

        present_codes = {};
        for token_index = 1:numel(tokens)
            token_pair = tokens{token_index};
            code = token_pair{1};
            present_codes{end + 1} = code;
        end

        is_male = (sex_values(record_index) == 1);

        for class_index = 1:numel(arrhythmia_classes)
            if any(strcmp(present_codes, arrhythmia_classes{class_index}))
                if is_male
                    male_counts(class_index) = male_counts(class_index) + 1;
                else
                    female_counts(class_index) = female_counts(class_index) + 1;
                end
            end
        end
    end

    create_or_show_figure(21);
    bar_data = [female_counts, male_counts];
    bar(bar_data, 'stacked');
    set(gca, 'XTick', 1:numel(arrhythmia_classes), 'XTickLabel', arrhythmia_classes);
    set(gca, 'TickLabelInterpreter', 'none');
    xlabel('Arrhythmia sinifi');
    ylabel('Kayit sayisi');
    legend({'Kadin', 'Erkek'}, 'Location', 'northwest');
    title('Arrhythmia siniflarinda cinsiyet dagilimi');
    print(fullfile(output_dir, 'octave_arrhythmia_sex_distribution.png'), '-dpng');

    %% 3) Yas grubu ve cinsiyet kirilimlari (tum dataset)
    fprintf('3) Yas grubu ve cinsiyet kirilim grafigi olusturuluyor...\n');

    valid_age_sex_mask = ~isnan(age_values) & age_values <= 89 & ~isnan(sex_values);
    age_filtered = age_values(valid_age_sex_mask);
    sex_filtered = sex_values(valid_age_sex_mask);

    num_records_age_sex = numel(age_filtered);
    age_group_counts_female = zeros(4, 1);
    age_group_counts_male = zeros(4, 1);

    for idx = 1:num_records_age_sex
        a = age_filtered(idx);
        s = sex_filtered(idx);
        if a < 40
            g = 1;
        elseif a < 60
            g = 2;
        elseif a < 80
            g = 3;
        else
            g = 4;
        end
        if s == 1
            age_group_counts_male(g) = age_group_counts_male(g) + 1;
        else
            age_group_counts_female(g) = age_group_counts_female(g) + 1;
        end
    end

    age_group_labels = {'<40', '40-59', '60-79', '>=80'};

    create_or_show_figure(22);
    bar_age_sex = [age_group_counts_female, age_group_counts_male];
    bar(bar_age_sex, 'stacked');
    set(gca, 'XTick', 1:4, 'XTickLabel', age_group_labels);
    ylabel('Kayit sayisi');
    legend({'Kadin', 'Erkek'}, 'Location', 'northwest');
    title('Yas gruplarina gore cinsiyet dagilimi (age <= 89)');
    print(fullfile(output_dir, 'octave_age_group_sex_distribution.png'), '-dpng');

    %% 4) Secilmis SCP kodlari icin co-occurrence heatmap
    fprintf('4) Secilmis SCP kodlari icin co-occurrence heatmap olusturuluyor...\n');

    % Siklik olarak onemli ve aritmi ile ilgili bazilarini secelim:
    selected_codes = {'AFIB', 'STACH', 'SARRH', 'PVC', 'PAC', 'AFLT', 'SBRAD', 'SVTAC', 'NORM', 'LVH'};

    co_matrix = ptbxl_cooccurrence_matrix(metadata.scp_codes_raw, selected_codes);

    % Co-occurrence icin ozet tablo (en yuksek Jaccard degere sahip ilk 10 kod ciftini yazdir)
    num_codes = numel(selected_codes);
    diag_counts = diag(co_matrix);
    pair_labels = {};
    pair_counts = [];
    pair_jaccard = [];
    for i = 1:num_codes
        for j = i+1:num_codes
            intersection = co_matrix(i, j);
            if intersection <= 0
                continue;
            end
            union_ij = diag_counts(i) + diag_counts(j) - intersection;
            if union_ij > 0
                jacc = intersection / union_ij;
            else
                jacc = 0;
            end
            pair_labels{end + 1} = sprintf('%s - %s', selected_codes{i}, selected_codes{j}); %#ok<AGROW>
            pair_counts(end + 1) = intersection; %#ok<AGROW>
            pair_jaccard(end + 1) = jacc; %#ok<AGROW>
        end
    end

    if ~isempty(pair_labels)
        [sorted_jaccard, sort_indices] = sort(pair_jaccard, 'descend'); %#ok<ASGLU>
        top_k = min(10, numel(sort_indices));
        fprintf('\nCo-occurrence ozet tablosu (en yuksek Jaccard''a sahip ilk %d kod cifti):\n', top_k);
        fprintf('  Kod Cifti        | Birlikte Kayit | Jaccard\n');
        fprintf('  -----------------+----------------+--------\n');
        for idx = 1:top_k
            pair_index = sort_indices(idx);
            fprintf('  %-16s | %14d | %6.3f\n', pair_labels{pair_index}, pair_counts(pair_index), pair_jaccard(pair_index));
        end
    end

    create_or_show_figure(23);
    imagesc(co_matrix);
    colorbar;
    axis equal tight;
    set(gca, 'XTick', 1:numel(selected_codes), 'XTickLabel', selected_codes);
    set(gca, 'YTick', 1:numel(selected_codes), 'YTickLabel', selected_codes);
    set(gca, 'TickLabelInterpreter', 'none');
    xlabel('SCP kodu');
    ylabel('SCP kodu');
    title('Secilmis SCP kodlari icin birlikte-gorunme matrisi');
    print(fullfile(output_dir, 'octave_scp_cooccurrence_heatmap.png'), '-dpng');

    % Ayrica Jaccard benzerligi ile normalize edilmis bir heatmap uret
    jaccard_matrix = zeros(num_codes, num_codes);
    for i = 1:num_codes
        for j = 1:num_codes
            intersection = co_matrix(i, j);
            union_ij = diag_counts(i) + diag_counts(j) - intersection;
            if union_ij > 0
                jaccard_matrix(i, j) = intersection / union_ij;
            else
                jaccard_matrix(i, j) = 0;
            end
        end
    end

    create_or_show_figure(24);
    imagesc(jaccard_matrix, [0, 1]);
    colorbar;
    axis equal tight;
    set(gca, 'XTick', 1:num_codes, 'XTickLabel', selected_codes);
    set(gca, 'YTick', 1:num_codes, 'YTickLabel', selected_codes);
    set(gca, 'TickLabelInterpreter', 'none');
    xlabel('SCP kodu');
    ylabel('SCP kodu');
    title('Secilmis SCP kodlari icin birlikte-gorunme (Jaccard) matrisi');
    print(fullfile(output_dir, 'octave_scp_cooccurrence_heatmap_normalized.png'), '-dpng');

    %% 5) Diagnostic class (scp_statements.csv) dagilimi
    fprintf('5) Diagnostic class (scp_statements.csv) dagilimi olusturuluyor...\n');
    try
        categories = ptbxl_scp_categories(base_directory);
        [scp_code_list, scp_counts] = ptbxl_label_frequencies(metadata.scp_codes_raw);

        code_to_class = containers.Map();
        for idx = 1:numel(categories.codes)
            code_to_class(categories.codes{idx}) = categories.diagnostic_class{idx};
        end

        class_names = {};
        class_counts = [];
        for idx = 1:numel(scp_code_list)
            code = scp_code_list{idx};
            if isKey(code_to_class, code)
                cls = code_to_class(code);
            else
                cls = '';
            end
            if isempty(cls)
                continue;
            end
            existing = find(strcmp(class_names, cls));
            if isempty(existing)
                class_names{end + 1} = cls; %#ok<AGROW>
                class_counts(end + 1) = scp_counts(idx); %#ok<AGROW>
            else
                class_counts(existing) = class_counts(existing) + scp_counts(idx);
            end
        end

        if ~isempty(class_names)
            [sorted_counts, sort_idx] = sort(class_counts, 'descend');
            sorted_names = class_names(sort_idx);

            create_or_show_figure(25);
            bar(sorted_counts);
            set(gca, 'XTick', 1:numel(sorted_names), 'XTickLabel', sorted_names);
            set(gca, 'TickLabelInterpreter', 'none');
            xtickangle(45);
            ylabel('Kayit sayisi (en az bir kod ile)');
            title('Diagnostic class dagilimi (scp_statements.csv)');
            print(fullfile(output_dir, 'octave_diagnostic_class_distribution.png'), '-dpng');
        end
    catch err
        fprintf('Diagnostic class dagilimi olusturulurken hata: %s\n', err.message);
    end

    fprintf('\nIleri duzey grafikler olusturuldu ve %s altina kaydedildi.\n', output_dir);
end
