function ptbxl_dashboard(base_directory)
% PTBXL_DASHBOARD  PTB-XL metadata icin basit bir etkileşimli GUI.
%
%   PTBXL_DASHBOARD()
%   PTBXL_DASHBOARD(base_directory)
%
%   Sol ustte grafik secimi icin bir popup menusu,
%   sag tarafta ise secili grafigi gosteren bir alan bulunur.
%
%   Gosterilen gorseller:
%     1) Dataset ozet metni (age <= 89 filtresi ile)
%     2) Yas dagilimi (age <= 89)
%     3) Cinsiyet dagilimi
%     4) Normal / anormal dagilimi
%     5) Arrhythmia odakli 10 sinifin kayit dagilimi
%     6) Yas gruplarina gore cinsiyet dagilimi
%     7) Arrhythmia siniflarina gore yas (ortalama ± std)
%     8) Arrhythmia siniflarinda cinsiyet dagilimi
%     9) Secilmis SCP kodlari icin co-occurrence (ham sayi)
%    10) Secilmis SCP kodlari icin co-occurrence (Jaccard)

    if nargin < 1 || isempty(base_directory)
        base_directory = fullfile('..', 'dataset', 'physionet.org', 'files', 'ptb-xl', '1.0.3');
    end

    metadata_file_path = fullfile(base_directory, 'ptbxl_database.csv');
    fprintf('PTB-XL dashboard icin metadata okunuyor: %s\n', metadata_file_path);

    metadata = ptbxl_read_metadata(base_directory);

    f = figure('Name', 'PTB-XL Metadata Dashboard', ...
               'NumberTitle', 'off', ...
               'Units', 'normalized', ...
               'Position', [0.1 0.1 0.8 0.8]);

    plot_names = { ...
        '1) Dataset ozeti', ...
        '2) Yas dagilimi', ...
        '3) Cinsiyet dagilimi', ...
        '4) Normal / anormal dagilimi', ...
        '5) Arrhythmia sinif dagilimi', ...
        '6) Yas grubu x cinsiyet dagilimi', ...
        '7) Arrhythmia yas istatistikleri', ...
        '8) Arrhythmia cinsiyet dagilimi', ...
        '9) Co-occurrence (ham sayi)', ...
        '10) Co-occurrence (Jaccard)', ...
        '11) Diagnostic class dagilimi', ...
        '12) 12-lead ECG onizleme (WFDB)'};

    % Arrhythmia sinif filtresi icin secenekler
    arrhythmia_filter_options = {'(hepsi)', 'SR', 'AFIB', 'STACH', 'SARRH', 'PVC', 'PAC', 'AFLT', 'SBRAD', 'SVTAC', 'NORM'};
    current_arrhythmia_filter_label = arrhythmia_filter_options{1};
    current_arrhythmia_codes = {};
    current_age_min = 0;
    current_age_max = 89;
    current_patient_id = NaN;

    uicontrol('Parent', f, ...
              'Style', 'text', ...
              'Units', 'normalized', ...
              'Position', [0.03 0.9 0.26 0.05], ...
              'String', 'Gosterilecek grafigi secin:', ...
              'HorizontalAlignment', 'left');

    popup_plot = uicontrol('Parent', f, ...
                           'Style', 'popupmenu', ...
                           'Units', 'normalized', ...
                           'Position', [0.03 0.84 0.26 0.05], ...
                           'String', plot_names, ...
                           'Callback', @on_plot_selection_changed);

    % Arrhythmia sinif filtresi icin listbox
    uicontrol('Parent', f, ...
              'Style', 'text', ...
              'Units', 'normalized', ...
              'Position', [0.03 0.78 0.26 0.05], ...
              'String', 'Arrhythmia filtresi (bazı grafige uygulanir):', ...
              'HorizontalAlignment', 'left');

    popup_class = uicontrol('Parent', f, ...
                            'Style', 'listbox', ...
                            'Units', 'normalized', ...
                            'Position', [0.03 0.64 0.26 0.13], ...
                            'String', arrhythmia_filter_options, ...
                            'Min', 0, 'Max', numel(arrhythmia_filter_options), ...
                            'Value', 1, ...
                            'Callback', @on_class_selection_changed);

    % Yas araligi filtresi icin edit kutulari
    uicontrol('Parent', f, ...
              'Style', 'text', ...
              'Units', 'normalized', ...
              'Position', [0.03 0.58 0.26 0.04], ...
              'String', 'Yas araligi (min / max):', ...
              'HorizontalAlignment', 'left');

    edit_age_min = uicontrol('Parent', f, ...
                             'Style', 'edit', ...
                             'Units', 'normalized', ...
                             'Position', [0.03 0.54 0.12 0.045], ...
                             'String', num2str(current_age_min), ...
                             'Callback', @on_age_min_changed);

    edit_age_max = uicontrol('Parent', f, ...
                             'Style', 'edit', ...
                             'Units', 'normalized', ...
                             'Position', [0.17 0.54 0.12 0.045], ...
                             'String', num2str(current_age_max), ...
                             'Callback', @on_age_max_changed);

    % Hasta ID filtresi
    uicontrol('Parent', f, ...
              'Style', 'text', ...
              'Units', 'normalized', ...
              'Position', [0.03 0.52 0.26 0.04], ...
              'String', 'Hasta ID (opsiyonel):', ...
              'HorizontalAlignment', 'left');

    edit_patient_id = uicontrol('Parent', f, ...
                                'Style', 'edit', ...
                                'Units', 'normalized', ...
                                'Position', [0.03 0.48 0.26 0.045], ...
                                'String', '', ...
                                'Callback', @on_patient_id_changed);

    % Yeniden ciz ve filtre reset butonlari
    uicontrol('Parent', f, ...
              'Style', 'pushbutton', ...
              'Units', 'normalized', ...
              'Position', [0.03 0.42 0.26 0.05], ...
              'String', 'Yeniden ciz', ...
              'Callback', @on_redraw_clicked);

    uicontrol('Parent', f, ...
              'Style', 'pushbutton', ...
              'Units', 'normalized', ...
              'Position', [0.03 0.36 0.26 0.05], ...
              'String', 'Filtreyi sifirla', ...
              'Callback', @on_reset_clicked);

    % Ozet export (metin) butonu
    uicontrol('Parent', f, ...
              'Style', 'pushbutton', ...
              'Units', 'normalized', ...
              'Position', [0.03 0.30 0.26 0.05], ...
              'String', 'Ozetini kaydet', ...
              'Callback', @on_export_clicked);

    % PNG export butonu
    uicontrol('Parent', f, ...
              'Style', 'pushbutton', ...
              'Units', 'normalized', ...
              'Position', [0.03 0.24 0.26 0.05], ...
              'String', 'PNG olarak kaydet', ...
              'Callback', @on_export_png_clicked);

    % Grafik aciklamasi icin metin alani (alt tarafta)
    info_text = uicontrol('Parent', f, ...
                          'Style', 'text', ...
                          'Units', 'normalized', ...
                          'Position', [0.03 0.05 0.26 0.18], ...
                          'String', '', ...
                          'HorizontalAlignment', 'left');

    on_plot_selection_changed(popup_plot, []);

    function on_redraw_clicked(~, ~)
        on_plot_selection_changed(popup_plot, []);
    end

    function on_class_selection_changed(src, ~)
        idx = get(src, 'Value');
        if isempty(idx) || any(idx == 1)
            set(src, 'Value', 1);
            current_arrhythmia_codes = {};
            current_arrhythmia_filter_label = arrhythmia_filter_options{1};
        else
            selected = arrhythmia_filter_options(idx);
            current_arrhythmia_codes = selected;
            current_arrhythmia_filter_label = strjoin(selected, '+');
        end
        on_plot_selection_changed(popup_plot, []);
    end

    function on_reset_clicked(~, ~)
        % Yas araligini ve arrhythmia filtresini varsayilana dondur
        current_age_min = 0;
        current_age_max = 89;
        set(edit_age_min, 'String', num2str(current_age_min));
        set(edit_age_max, 'String', num2str(current_age_max));

        current_arrhythmia_codes = {};
        current_arrhythmia_filter_label = arrhythmia_filter_options{1};
        set(popup_class, 'Value', 1);

        current_patient_id = NaN;
        set(edit_patient_id, 'String', '');

        on_plot_selection_changed(popup_plot, []);
    end

    function on_export_clicked(~, ~)
        selected_index = get(popup_plot, 'Value');
        plot_label = plot_names{selected_index};

        output_dir = fullfile('..', 'analysis_results');
        if ~exist(output_dir, 'dir')
            mkdir(output_dir);
        end
        output_file = fullfile(output_dir, 'ptbxl_dashboard_summary.txt');

        fid = fopen(output_file, 'a');
        if fid == -1
            fprintf('Ozet dosyasi acilamadi: %s\n', output_file);
            return;
        end

        timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
        fprintf(fid, '--- PTB-XL Dashboard Ozet (%s) ---\n', timestamp);
        fprintf(fid, 'Grafik     : %s\n', plot_label);
        fprintf(fid, 'Arrhythmia : %s\n', current_arrhythmia_filter_label);
        fprintf(fid, 'Yas araligi: %.1f - %.1f\n', current_age_min, current_age_max);
        if isfinite(current_patient_id)
            fprintf(fid, 'Hasta ID   : %.0f\n', current_patient_id);
        end

        info_lines = get(info_text, 'String');
        if ischar(info_lines)
            info_lines = cellstr(info_lines);
        end
        fprintf(fid, 'Aciklama:\n');
        for k = 1:numel(info_lines)
            fprintf(fid, '  %s\n', info_lines{k});
        end
        fprintf(fid, '\n');
        fclose(fid);

        fprintf('Dashboard ozeti kaydedildi: %s\n', output_file);
    end

    function on_arrhythmia_bar_click(~, ~)
        cp = get(gca, 'CurrentPoint');
        xclick = cp(1, 1);

        arrhythmia_classes = {'SR', 'AFIB', 'STACH', 'SARRH', 'PVC', ...
                              'PAC', 'AFLT', 'SBRAD', 'SVTAC', 'NORM'};
        x_centers = 1:numel(arrhythmia_classes);
        [~, idx] = min(abs(x_centers - xclick));
        if idx < 1 || idx > numel(arrhythmia_classes)
            return;
        end

        selected_code = arrhythmia_classes{idx};

        option_index = find(strcmp(arrhythmia_filter_options, selected_code), 1);
        if isempty(option_index)
            return;
        end

        current_arrhythmia_codes = {selected_code};
        current_arrhythmia_filter_label = selected_code;
        set(popup_class, 'Value', option_index);

        on_plot_selection_changed(popup_plot, []);
    end

    function on_export_png_clicked(~, ~)
        selected_index = get(popup_plot, 'Value');
        plot_label = plot_names{selected_index};

        output_dir = fullfile('..', 'analysis_results');
        if ~exist(output_dir, 'dir')
            mkdir(output_dir);
        end

        safe_label = regexprep(plot_label, '[^A-Za-z0-9]+', '_');
        safe_filter = regexprep(current_arrhythmia_filter_label, '[^A-Za-z0-9]+', '_');
        output_file = fullfile(output_dir, sprintf('ptbxl_dashboard_%s_%s_%.0f_%.0f.png', ...
            safe_label, safe_filter, current_age_min, current_age_max));

        f_exp = figure('Visible', 'off');
        ax_exp = axes('Parent', f_exp);

        switch selected_index
            case 1
                plot_dataset_summary(ax_exp, metadata, metadata_file_path);
            case 2
                plot_age_distribution(ax_exp, metadata, current_arrhythmia_codes, current_age_min, current_age_max);
            case 3
                plot_sex_distribution(ax_exp, metadata, current_arrhythmia_codes, current_age_min, current_age_max);
            case 4
                plot_normal_abnormal_distribution(ax_exp, metadata, current_age_min, current_age_max);
            case 5
                plot_arrhythmia_class_distribution(ax_exp, metadata, current_age_min, current_age_max);
            case 6
                plot_age_group_sex_distribution(ax_exp, metadata, current_arrhythmia_codes, current_age_min, current_age_max);
            case 7
                plot_arrhythmia_age_stats(ax_exp, metadata, current_age_min, current_age_max);
            case 8
                plot_arrhythmia_sex_distribution(ax_exp, metadata, current_age_min, current_age_max);
            case 9
                plot_cooccurrence_counts(ax_exp, metadata, current_arrhythmia_codes, current_age_min, current_age_max);
            case 10
                plot_cooccurrence_jaccard(ax_exp, metadata, current_arrhythmia_codes, current_age_min, current_age_max);
            case 11
                plot_diagnostic_class_distribution(ax_exp, metadata, metadata_file_path, current_age_min, current_age_max);
            case 12
                plot_ecg_signal_preview(ax_exp, metadata, base_directory, current_arrhythmia_codes, current_age_min, current_age_max, current_patient_id);
        end

        % Basliga filtre bilgilerini ekle
        title_obj = get(ax_exp, 'Title');
        orig_title = get(title_obj, 'String');
        if ischar(orig_title)
            lines = {orig_title};
        elseif iscell(orig_title)
            lines = orig_title;
        else
            lines = {''};
        end
        if selected_index ~= 1
            lines{end+1} = sprintf('Yas: %.1f-%.1f, Arrhythmia: %s', ...
                                   current_age_min, current_age_max, current_arrhythmia_filter_label);
        end
        title(ax_exp, lines);

        print(f_exp, output_file, '-dpng');
        close(f_exp);

        fprintf('Dashboard PNG kaydedildi: %s\n', output_file);
    end

    function on_age_min_changed(src, ~)
        val = str2double(get(src, 'String'));
        if ~isfinite(val)
            val = current_age_min;
        end
        current_age_min = val;
        if current_age_min > current_age_max
            current_age_max = current_age_min;
            set(edit_age_max, 'String', num2str(current_age_max));
        end
        on_plot_selection_changed(popup_plot, []);
    end

    function on_age_max_changed(src, ~)
        val = str2double(get(src, 'String'));
        if ~isfinite(val)
            val = current_age_max;
        end
        current_age_max = val;
        if current_age_max < current_age_min
            current_age_min = current_age_max;
            set(edit_age_min, 'String', num2str(current_age_min));
        end
        on_plot_selection_changed(popup_plot, []);
    end

    function on_patient_id_changed(src, ~)
        val = str2double(get(src, 'String'));
        if ~isfinite(val)
            current_patient_id = NaN;
            set(src, 'String', '');
        else
            current_patient_id = val;
        end
        on_plot_selection_changed(popup_plot, []);
    end

    function update_info_text(selected_index)
        % Kisa ve satirlara bolunmus aciklamalar (cell array)
        switch selected_index
            case 1
                lines = { ...
                    'Dataset ozeti:', ...
                    'Kayit ve hasta sayisi, yas/cinsiyet dagilimi,', ...
                    'dosya boyutu ve eksik hucre sayisini gosterir.', ...
                    '(age <= 89 filtresi ile).'};
            case 2
                lines = { ...
                    'Yas dagilimi:', ...
                    'age <= 89 icin kayit bazli yas histogrami.', ...
                    'Arrhythmia filtresi varsa sadece o sinifi', ...
                    'iceren kayitlar kullanilir.'};
            case 3
                lines = { ...
                    'Cinsiyet dagilimi:', ...
                    'Kayit bazinda Kadin/Erkek sayilari.', ...
                    'Arrhythmia filtresi varsa sadece o sinifi', ...
                    'iceren kayitlar kullanilir.'};
            case 4
                lines = { ...
                    'Normal / Anormal dagilimi:', ...
                    'scp_codes icinde "NORM" gecmesine gore', ...
                    'yaklasik normal/anormal kayit sayilari.', ...
                    '(tum dataset uzerinden).'};
            case 5
                lines = { ...
                    'Arrhythmia sinif dagilimi:', ...
                    'SR, AFIB, PVC vb. 10 hedef sinif icin,', ...
                    'en az bir kod iceren kayit sayilari.', ...
                    '(multi-label, tum dataset uzerinden).'};
            case 6
                lines = { ...
                    'Yas grubu x cinsiyet:', ...
                    'age <= 89 icin 4 yas grubunda', ...
                    '(<40, 40-59, 60-79, >=80) Kadin/Erkek', ...
                    'dagilimini gosterir.'};
            case 7
                lines = { ...
                    'Arrhythmia yas istatistikleri:', ...
                    'Her hedef sinif icin yas ortalamasi ve', ...
                    'standart sapmasi (bar + errorbar).', ...
                    'Sadece o sinifi iceren kayitlar kullanilir.'};
            case 8
                lines = { ...
                    'Arrhythmia cinsiyet dagilimi:', ...
                    'Her hedef sinif icin Kadin/Erkek sayilari,', ...
                    'stacked bar olarak gosterilir.'};
            case 9
                lines = { ...
                    'Co-occurrence (ham sayi):', ...
                    'Secilmis SCP kodlari (AFIB, PVC vb.) icin,', ...
                    'her kod ciftinin ayni kayitta kac kez', ...
                    'birlikte goruldugunu heatmap olarak gosterir.'};
            case 10
                lines = { ...
                    'Co-occurrence (Jaccard):', ...
                    'Secilmis SCP kodlari icin intersection / union', ...
                    'seklinde 0-1 araliginda Jaccard benzerligi.', ...
                    'Yuksek degerler daha sik birlikte gorulme demektir.'};
            case 11
                lines = { ...
                    'Diagnostic class dagilimi:', ...
                    'scp_statements.csv icindeki diagnostic_class', ...
                    'alanina gore ust kategori bazinda,', ...
                    'en az bir kod iceren kayit sayilari.'};
            case 12
                lines = { ...
                    '12-lead ECG onizleme:', ...
                    'WFDB rdsamp ile ham sinyalden ilk ~2000', ...
                    'ornek cizilir. Arrhythmia filtresi varsa', ...
                    'ilgili kayitlardan ilk bulunan secilir.', ...
                    'WFDB Toolbox yüklü degilse bilgilendirme yapar.'};
            otherwise
                lines = {''};
        end

        if ~isempty(current_arrhythmia_codes) && any(selected_index == [2, 3, 6, 9, 10, 12])
            lines{end+1} = ''; %#ok<AGROW>
            lines{end+1} = sprintf('Aktif arrhythmia filtresi: %s', current_arrhythmia_filter_label); %#ok<AGROW>
        end

        if isfinite(current_patient_id) && selected_index == 12
            lines{end+1} = ''; %#ok<AGROW>
            lines{end+1} = sprintf('Hasta filtresi: %.0f', current_patient_id); %#ok<AGROW>
        end

        if selected_index ~= 1
            lines{end+1} = ''; %#ok<AGROW>
            lines{end+1} = sprintf('Aktif yas araligi: %.1f - %.1f', current_age_min, current_age_max); %#ok<AGROW>
        end

        set(info_text, 'String', lines);
    end

    function on_plot_selection_changed(src, ~)
        selected_index = get(src, 'Value');

        % Var olan axes'leri sil ve yeni bir tanesini olustur
        existing_axes = findobj(f, 'Type', 'axes');
        delete(existing_axes);
        ax = axes('Parent', f, ...
                  'Units', 'normalized', ...
                  'Position', [0.32 0.12 0.65 0.8]);

        % Hangi grafiklerin arrhythmia filtresi kullandigini kontrol et
        uses_arrhythmia_filter = any(selected_index == [2, 3, 6, 9, 10, 12]);
        if uses_arrhythmia_filter
            set(popup_class, 'Enable', 'on');
        else
            set(popup_class, 'Enable', 'off');
        end

        update_info_text(selected_index);

        switch selected_index
            case 1
                plot_dataset_summary(ax, metadata, metadata_file_path);
            case 2
                plot_age_distribution(ax, metadata, current_arrhythmia_codes, current_age_min, current_age_max);
            case 3
                plot_sex_distribution(ax, metadata, current_arrhythmia_codes, current_age_min, current_age_max);
            case 4
                plot_normal_abnormal_distribution(ax, metadata, current_age_min, current_age_max);
            case 5
                plot_arrhythmia_class_distribution(ax, metadata, current_age_min, current_age_max);
                hbars = findobj(ax, 'Type', 'bar');
                if ~isempty(hbars)
                    set(hbars, 'ButtonDownFcn', @on_arrhythmia_bar_click);
                end
            case 6
                plot_age_group_sex_distribution(ax, metadata, current_arrhythmia_codes, current_age_min, current_age_max);
            case 7
                plot_arrhythmia_age_stats(ax, metadata, current_age_min, current_age_max);
            case 8
                plot_arrhythmia_sex_distribution(ax, metadata, current_age_min, current_age_max);
            case 9
                plot_cooccurrence_counts(ax, metadata, current_arrhythmia_codes, current_age_min, current_age_max);
            case 10
                plot_cooccurrence_jaccard(ax, metadata, current_arrhythmia_codes, current_age_min, current_age_max);
            case 11
                plot_diagnostic_class_distribution(ax, metadata, metadata_file_path, current_age_min, current_age_max);
            case 12
                plot_ecg_signal_preview(ax, metadata, base_directory, current_arrhythmia_codes, current_age_min, current_age_max, current_patient_id);
        end
    end
end

function plot_dataset_summary(ax, metadata, metadata_file_path)
    axes(ax);
    axis(ax, 'off');

    age_filter_mask = metadata.age <= 89 & ~isnan(metadata.age);
    total_records = sum(age_filter_mask);

    filtered_patient_ids = metadata.patient_id(age_filter_mask & ~isnan(metadata.patient_id));
    unique_patients = numel(unique(filtered_patient_ids));

    filtered_ages = metadata.age(age_filter_mask);
    min_age_filtered = min(filtered_ages);
    max_age_filtered = max(filtered_ages);

    filtered_sex = metadata.sex(age_filter_mask & ~isnan(metadata.sex));
    num_male = sum(filtered_sex == 1);
    num_female = sum(filtered_sex == 0);

    file_info = dir(metadata_file_path);
    if ~isempty(file_info)
        data_size_mb = file_info.bytes / (1024 * 1024);
    else
        data_size_mb = NaN;
    end

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

    lines = {};
    lines{end + 1} = 'PTB-XL Dataset Ozeti (age <= 89 filtresi ile)'; %#ok<AGROW>
    lines{end + 1} = sprintf('Toplam Kayit      : %d', total_records);
    lines{end + 1} = sprintf('Esiz Hasta Sayisi : %d', unique_patients);
    lines{end + 1} = sprintf('Yas Araligi       : %.0f-%.0f', min_age_filtered, max_age_filtered);
    lines{end + 1} = sprintf('Cinsiyet Dagilimi : Erkek=%d, Kadin=%d', num_male, num_female);
    lines{end + 1} = 'Kayit Suresi      : 10 saniye';
    lines{end + 1} = 'Ornekleme Hizlari : 500 Hz (orijinal), 100 Hz (downsampled)';
    lines{end + 1} = 'Lead Sayisi       : 12';
    if ~isnan(data_size_mb)
        lines{end + 1} = sprintf('CSV Boyutu        : %.2f MB', data_size_mb);
    else
        lines{end + 1} = 'CSV Boyutu        : (hesaplanamadi)';
    end
    lines{end + 1} = sprintf('Eksik Hucre Sayisi: %d', missing_count);

    y = 0.95;
    for i = 1:numel(lines)
        text(0.01, y, lines{i}, 'Parent', ax, 'FontName', 'monospace');
        y = y - 0.07;
    end
end

function plot_age_distribution(ax, metadata, selected_codes, age_min, age_max)
    axes(ax);

    age_values = metadata.age;
    if nargin < 4 || ~isfinite(age_min)
        age_min = 0;
    end
    if nargin < 5 || ~isfinite(age_max)
        age_max = 89;
    end
    valid_age_mask = ~isnan(age_values) & age_values >= age_min & age_values <= age_max;
    if nargin >= 3 && ~isempty(selected_codes)
        class_mask = build_multiclass_presence_mask(metadata.scp_codes_raw, selected_codes);
        valid_age_mask = valid_age_mask & class_mask;
    end
    valid_ages = age_values(valid_age_mask);
    if isempty(valid_ages)
        text(0.5, 0.5, 'Yas verisi bulunamadi', 'HorizontalAlignment', 'center');
        axis off;
        return;
    end

    hist(valid_ages, 40);
    xlabel('Yas');
    ylabel('Kayit sayisi');
    title('PTB-XL yas dagilimi (age <= 89)');
end

function plot_sex_distribution(ax, metadata, selected_codes, age_min, age_max)
    axes(ax);

    sex_values = metadata.sex;
    age_values = metadata.age;
    if nargin < 4 || ~isfinite(age_min)
        age_min = 0;
    end
    if nargin < 5 || ~isfinite(age_max)
        age_max = 89;
    end
    valid_sex_mask = ~isnan(sex_values) & ~isnan(age_values) & age_values >= age_min & age_values <= age_max;
    if nargin >= 3 && ~isempty(selected_codes)
        class_mask = build_multiclass_presence_mask(metadata.scp_codes_raw, selected_codes);
        valid_sex_mask = valid_sex_mask & class_mask;
    end
    female_count = sum(sex_values(valid_sex_mask) == 0);
    male_count = sum(sex_values(valid_sex_mask) == 1);

    bar([female_count, male_count]);
    set(gca, 'XTickLabel', {'Kadin', 'Erkek'});
    ylabel('Kayit sayisi');
    title('PTB-XL cinsiyet dagilimi');
end

function plot_normal_abnormal_distribution(ax, metadata, age_min, age_max)
    axes(ax);

    age_values = metadata.age;
    if nargin < 3 || ~isfinite(age_min)
        age_min = 0;
    end
    if nargin < 4 || ~isfinite(age_max)
        age_max = 89;
    end
    age_mask = ~isnan(age_values) & age_values >= age_min & age_values <= age_max;

    normal_mask_all = build_normal_label_mask(metadata.scp_codes_raw);
    normal_mask = normal_mask_all & age_mask;
    valid_mask = age_mask;

    num_normal_records = sum(normal_mask);
    num_abnormal_records = sum(valid_mask) - num_normal_records;

    bar([num_normal_records, num_abnormal_records]);
    set(gca, 'XTickLabel', {'Normal', 'Anormal'});
    ylabel('Kayit sayisi');
    title('Normal / anormal dagilimi (NORM heuristigi)');
end

function plot_arrhythmia_class_distribution(ax, metadata, age_min, age_max)
    axes(ax);

    age_values = metadata.age;
    if nargin < 3 || ~isfinite(age_min)
        age_min = 0;
    end
    if nargin < 4 || ~isfinite(age_max)
        age_max = 89;
    end
    age_mask = ~isnan(age_values) & age_values >= age_min & age_values <= age_max;

    scp_filtered = metadata.scp_codes_raw(age_mask);
    [scp_code_list, scp_counts] = ptbxl_label_frequencies(scp_filtered);

    arrhythmia_classes = {'SR', 'AFIB', 'STACH', 'SARRH', 'PVC', 'PAC', 'AFLT', 'SBRAD', 'SVTAC', 'NORM'};
    arrhythmia_counts = zeros(numel(arrhythmia_classes), 1);
    for c_index = 1:numel(arrhythmia_classes)
        code = arrhythmia_classes{c_index};
        existing_index = find(strcmp(scp_code_list, code));
        if ~isempty(existing_index)
            arrhythmia_counts(c_index) = scp_counts(existing_index);
        end
    end

    bar(arrhythmia_counts);
    set(gca, 'XTick', 1:numel(arrhythmia_classes), 'XTickLabel', arrhythmia_classes);
    set(gca, 'TickLabelInterpreter', 'none');
    xtickangle(45);
    ylabel('Kayit sayisi');
    title('Arrhythmia odakli secilmis SCP kodlarinin dagilimi');
end

function plot_age_group_sex_distribution(ax, metadata, selected_codes, age_min, age_max)
    axes(ax);

    age_values = metadata.age;
    sex_values = metadata.sex;
    if nargin < 4 || ~isfinite(age_min)
        age_min = 0;
    end
    if nargin < 5 || ~isfinite(age_max)
        age_max = 89;
    end
    valid_age_sex_mask = ~isnan(age_values) & age_values >= age_min & age_values <= age_max & ~isnan(sex_values);
    if nargin >= 3 && ~isempty(selected_codes)
        class_mask = build_multiclass_presence_mask(metadata.scp_codes_raw, selected_codes);
        valid_age_sex_mask = valid_age_sex_mask & class_mask;
    end
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

    bar_age_sex = [age_group_counts_female, age_group_counts_male];
    bar(bar_age_sex, 'stacked');
    set(gca, 'XTick', 1:4, 'XTickLabel', age_group_labels);
    ylabel('Kayit sayisi');
    legend({'Kadin', 'Erkek'}, 'Location', 'northwest');
    title('Yas gruplarina gore cinsiyet dagilimi (age <= 89)');
end

function plot_arrhythmia_age_stats(ax, metadata, age_min, age_max)
    axes(ax);

    arrhythmia_classes = {'SR', 'AFIB', 'STACH', 'SARRH', 'PVC', 'PAC', 'AFLT', 'SBRAD', 'SVTAC', 'NORM'};
    age_values = metadata.age;
    if nargin < 3 || ~isfinite(age_min)
        age_min = 0;
    end
    if nargin < 4 || ~isfinite(age_max)
        age_max = 89;
    end
    valid_age_mask = ~isnan(age_values) & age_values >= age_min & age_values <= age_max;

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

    bar(1:num_classes, mean_ages);
    hold on;
    errorbar(1:num_classes, mean_ages, std_ages, '.k');
    hold off;
    set(gca, 'XTick', 1:num_classes, 'XTickLabel', arrhythmia_classes);
    set(gca, 'TickLabelInterpreter', 'none');
    xlabel('Arrhythmia sinifi (SCP kodu)');
    ylabel('Yas (ortalama ± std)');
    title('Arrhythmia siniflarina gore yas istatistikleri');
end

function plot_arrhythmia_sex_distribution(ax, metadata, age_min, age_max)
    axes(ax);

    arrhythmia_classes = {'SR', 'AFIB', 'STACH', 'SARRH', 'PVC', 'PAC', 'AFLT', 'SBRAD', 'SVTAC', 'NORM'};

    sex_values = metadata.sex;
    age_values = metadata.age;
    if nargin < 3 || ~isfinite(age_min)
        age_min = 0;
    end
    if nargin < 4 || ~isfinite(age_max)
        age_max = 89;
    end
    valid_sex_mask = ~isnan(sex_values) & ~isnan(age_values) & age_values >= age_min & age_values <= age_max;

    male_counts = zeros(numel(arrhythmia_classes), 1);
    female_counts = zeros(numel(arrhythmia_classes), 1);

    pattern = '''([A-Z0-9_/]+)''\s*:\s*([0-9.]+)';
    num_records = metadata.num_records;
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

    bar_data = [female_counts, male_counts];
    bar(bar_data, 'stacked');
    set(gca, 'XTick', 1:numel(arrhythmia_classes), 'XTickLabel', arrhythmia_classes);
    set(gca, 'TickLabelInterpreter', 'none');
    xlabel('Arrhythmia sinifi');
    ylabel('Kayit sayisi');
    legend({'Kadin', 'Erkek'}, 'Location', 'northwest');
    title('Arrhythmia siniflarinda cinsiyet dagilimi');
end

function plot_cooccurrence_counts(ax, metadata, selected_codes, age_min, age_max)
    axes(ax);

    selected_codes = {'AFIB', 'STACH', 'SARRH', 'PVC', 'PAC', 'AFLT', 'SBRAD', 'SVTAC', 'NORM', 'LVH'};

    scp_codes_raw = metadata.scp_codes_raw;
    age_values = metadata.age;
    if nargin < 4 || ~isfinite(age_min)
        age_min = 0;
    end
    if nargin < 5 || ~isfinite(age_max)
        age_max = 89;
    end
    base_mask = ~isnan(age_values) & age_values >= age_min & age_values <= age_max;
    if nargin >= 3 && ~isempty(selected_codes)
        class_mask = build_multiclass_presence_mask(scp_codes_raw, selected_codes);
        base_mask = base_mask & class_mask;
    end
    scp_codes_raw = scp_codes_raw(base_mask);

    co_matrix = ptbxl_cooccurrence_matrix(scp_codes_raw, selected_codes);

    imagesc(co_matrix);
    colorbar;
    axis equal tight;
    set(gca, 'XTick', 1:numel(selected_codes), 'XTickLabel', selected_codes);
    set(gca, 'YTick', 1:numel(selected_codes), 'YTickLabel', selected_codes);
    set(gca, 'TickLabelInterpreter', 'none');
    xlabel('SCP kodu');
    ylabel('SCP kodu');
    title('Secilmis SCP kodlari icin birlikte-gorunme (ham sayi)');
end

function plot_cooccurrence_jaccard(ax, metadata, selected_codes, age_min, age_max)
    axes(ax);

    selected_codes = {'AFIB', 'STACH', 'SARRH', 'PVC', 'PAC', 'AFLT', 'SBRAD', 'SVTAC', 'NORM', 'LVH'};

    scp_codes_raw = metadata.scp_codes_raw;
    age_values = metadata.age;
    if nargin < 4 || ~isfinite(age_min)
        age_min = 0;
    end
    if nargin < 5 || ~isfinite(age_max)
        age_max = 89;
    end
    base_mask = ~isnan(age_values) & age_values >= age_min & age_values <= age_max;
    if nargin >= 3 && ~isempty(selected_codes)
        class_mask = build_multiclass_presence_mask(scp_codes_raw, selected_codes);
        base_mask = base_mask & class_mask;
    end
    scp_codes_raw = scp_codes_raw(base_mask);

    co_matrix = ptbxl_cooccurrence_matrix(scp_codes_raw, selected_codes);

    num_codes = numel(selected_codes);
    diag_counts = diag(co_matrix);
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

    imagesc(jaccard_matrix, [0, 1]);
    colorbar;
    axis equal tight;
    set(gca, 'XTick', 1:num_codes, 'XTickLabel', selected_codes);
    set(gca, 'YTick', 1:num_codes, 'YTickLabel', selected_codes);
    set(gca, 'TickLabelInterpreter', 'none');
    xlabel('SCP kodu');
    ylabel('SCP kodu');
    title('Secilmis SCP kodlari icin birlikte-gorunme (Jaccard)');
end

function normal_mask = build_normal_label_mask(scp_codes_raw)
    num_records = numel(scp_codes_raw);
    normal_mask = false(num_records, 1);
    for record_index = 1:num_records
        codes_string = scp_codes_raw{record_index};
        if isempty(codes_string)
            continue;
        end
        contains_norm = ~isempty(strfind(codes_string, 'NORM')); %#ok<STREMP>
        if contains_norm
            normal_mask(record_index) = true;
        end
    end
end

function plot_diagnostic_class_distribution(ax, metadata, metadata_file_path, age_min, age_max)
    axes(ax);

    base_directory = fileparts(metadata_file_path);
    try
        categories = ptbxl_scp_categories(base_directory);

        age_values = metadata.age;
        if nargin < 4 || ~isfinite(age_min)
            age_min = 0;
        end
        if nargin < 5 || ~isfinite(age_max)
            age_max = 89;
        end
        age_mask = ~isnan(age_values) & age_values >= age_min & age_values <= age_max;
        scp_filtered = metadata.scp_codes_raw(age_mask);

        [scp_code_list, scp_counts] = ptbxl_label_frequencies(scp_filtered);

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

        if isempty(class_names)
            text(0.5, 0.5, 'Diagnostic class bilgisi bulunamadi', 'HorizontalAlignment', 'center');
            axis off;
            return;
        end

        [sorted_counts, sort_idx] = sort(class_counts, 'descend');
        sorted_names = class_names(sort_idx);

        bar(sorted_counts);
        set(gca, 'XTick', 1:numel(sorted_names), 'XTickLabel', sorted_names);
        set(gca, 'TickLabelInterpreter', 'none');
        xtickangle(45);
        ylabel('Kayit sayisi (en az bir kod ile)');
        title('Diagnostic class dagilimi (scp_statements.csv)');
    catch err
        cla(ax);
        text(0.5, 0.5, sprintf('Hata: %s', err.message), 'HorizontalAlignment', 'center');
        axis off;
    end
end

function plot_ecg_signal_preview(ax, metadata, base_directory, selected_codes, age_min, age_max, patient_id)
    axes(ax);
    cla(ax);

    if nargin < 5 || ~isfinite(age_min)
        age_min = 0;
    end
    if nargin < 6 || ~isfinite(age_max)
        age_max = 89;
    end

    % Kayit yollarini bul
    record_paths = {};
    if isfield(metadata, 'filename_hr') && ~isempty(metadata.filename_hr)
        record_paths = metadata.filename_hr;
    elseif isfield(metadata, 'filename_lr') && ~isempty(metadata.filename_lr)
        record_paths = metadata.filename_lr;
    end
    if isempty(record_paths)
        text(0.5, 0.5, 'filename_hr / filename_lr sutunlari bulunamadi.', 'HorizontalAlignment', 'center');
        axis off;
        return;
    end

    age_values = metadata.age;
    valid_mask = ~isnan(age_values) & age_values >= age_min & age_values <= age_max;
    if nargin >= 4 && ~isempty(selected_codes)
        class_mask = build_multiclass_presence_mask(metadata.scp_codes_raw, selected_codes);
        valid_mask = valid_mask & class_mask;
    end
    if nargin >= 7 && isfinite(patient_id)
        valid_mask = valid_mask & (metadata.patient_id == patient_id);
    end

    candidate_indices = find(valid_mask);
    chosen_index = NaN;
    record_name = '';
    for idx = candidate_indices'
        rel_path = record_paths{idx};
        if isempty(rel_path)
            continue;
        end
        candidate = fullfile(base_directory, rel_path);
        if exist([candidate '.hea'], 'file') || exist([candidate '.dat'], 'file')
            record_name = candidate;
            chosen_index = idx;
            break;
        end
    end

    if isnan(chosen_index)
        text(0.5, 0.5, 'Filtreye uyan WFDB kaydi bulunamadi.', 'HorizontalAlignment', 'center');
        axis off;
        return;
    end

    [ok, sig, Fs, lead_names, err_msg, source_label] = read_wfdb_signal(record_name, 2000);
    if ~ok
        text(0.5, 0.55, 'Sinyal okunamadi.', 'HorizontalAlignment', 'center');
        text(0.5, 0.45, err_msg, 'HorizontalAlignment', 'center');
        axis off;
        return;
    end

    max_leads = size(sig, 2);
    max_samples = size(sig, 1);

    t = (0:max_samples-1) / Fs;
    colors = lines(max_leads);
    hold on;
    for k = 1:max_leads
        plot(t, sig(:, k), 'Color', colors(k, :));
    end
    hold off;
    grid on;
    xlabel('Time (s)');
    ylabel('Amplitude (phys units)');
    legend(lead_names, 'Location', 'eastoutside');

    age_val = metadata.age(chosen_index);
    if isnan(age_val)
        age_str = 'NA';
    else
        age_str = sprintf('%.0f', age_val);
    end
    sex_val = metadata.sex(chosen_index);
    if isnan(sex_val)
        sex_str = 'NA';
    elseif sex_val == 1
        sex_str = 'Male';
    else
        sex_str = 'Female';
    end

    codes_str = metadata.scp_codes_raw{chosen_index};
    diag_codes = {};
    if ~isempty(codes_str)
        tokens = regexp(codes_str, '''([A-Z0-9_/]+)''\s*:\s*([0-9.]+)', 'tokens');
        for t_index = 1:numel(tokens)
            diag_codes{end + 1} = tokens{t_index}{1}; %#ok<AGROW>
        end
    end
    if isempty(diag_codes)
        diag_str = 'N/A';
    else
        diag_str = strjoin(unique(diag_codes), ', ');
    end

    if nargin >= 7 && isfinite(patient_id)
        patient_line = sprintf('Patient ID: %.0f', patient_id);
    else
        patient_line = sprintf('Patient ID: %.0f', metadata.patient_id(chosen_index));
    end

    title_lines = { ...
        sprintf('12-lead ECG onizleme | Fs=%.0f Hz | ilk %d ornek (%s)', Fs, max_samples, source_label), ...
        sprintf('Record: %s', record_paths{chosen_index}), ...
        patient_line, ...
        sprintf('Age: %s | Sex: %s | Diagnosis: %s', age_str, sex_str, diag_str)};
    title(title_lines, 'Interpreter', 'none');
end

function [ok, sig, Fs, lead_names, err_msg, source_label] = read_wfdb_signal(record_name, max_samples)
    % read_wfdb_signal  WFDB kaydini rdsamp veya Python wfdb ile okur.
    ok = false;
    sig = [];
    Fs = NaN;
    lead_names = {};
    err_msg = '';
    source_label = 'n/a';

    % 1) Octave rdsamp (WFDB Toolbox) varsa onu dene
    if exist('rdsamp', 'file')
        try
            [sig, Fs, ~, siginfo] = rdsamp(record_name);
            source_label = 'Octave rdsamp';
        catch
            try
                [sig, Fs] = rdsamp(record_name);
                siginfo = [];
                source_label = 'Octave rdsamp';
            catch err
                err_msg = sprintf('rdsamp hatasi: %s', err.message);
                sig = [];
            end
        end

        if ~isempty(sig)
            max_leads = min(12, size(sig, 2));
            sig = sig(1:min(max_samples, size(sig, 1)), 1:max_leads);
            lead_names = derive_lead_names(siginfo, max_leads);
            Fs = Fs;
            ok = true;
            return;
        end
    end

    % 2) Python wfdb fallback (requirements.txt icinde wfdb var)
    tmp_csv = [tempname(), '.csv'];
    py_exec = find_python_exec();
    if isempty(py_exec)
        err_msg = 'Python (python/python3) bulunamadi.';
        return;
    end

    % wfdb modulu mevcut mu?
    status_check = system(sprintf('%s -c "import wfdb" 2> /dev/null', py_exec));
    if status_check ~= 0
        err_msg = sprintf('%s icin Python wfdb paketi yok. pip install wfdb', py_exec);
        return;
    end

    py_cmd = sprintf([ ...
        'python - <<''PY''\n' ...
        'import wfdb, numpy as np\n' ...
        'record = r''%s''\n' ...
        'max_samples = %d\n' ...
        'sig, fields = wfdb.rdsamp(record, sampfrom=0, sampto=max_samples)\n' ...
        'if isinstance(fields, dict):\n' ...
        '    fs = float(fields.get("fs", 0.0))\n' ...
        '    names = list(fields.get("sig_name", []) or [])\n' ...
        'else:\n' ...
        '    fs = float(getattr(fields, "fs", 0.0))\n' ...
        '    names = list(getattr(fields, "sig_name", []) or [])\n' ...
        'if not names:\n' ...
        '    names = [f"Lead{i+1}" for i in range(sig.shape[1])]\n' ...
        'header = "#fs,{:.6f}\\n#leads,".format(fs) + ",".join(names)\n' ...
        'np.savetxt(r"%s", sig, delimiter=",", header=header, comments="")\n' ...
        'PY'], record_name, max_samples, tmp_csv);

    status = system(strrep(py_cmd, 'python', py_exec));
    if status ~= 0
        err_msg = sprintf('%s wfdb ile okuma basarisiz.', py_exec);
        return;
    end
    if ~exist(tmp_csv, 'file')
        err_msg = 'Python wfdb cikti dosyasi bulunamadi.';
        return;
    end

    fid = fopen(tmp_csv, 'r');
    if fid == -1
        err_msg = 'Geçici dosya acilamadi.';
        return;
    end
    line1 = fgetl(fid);
    line2 = fgetl(fid);
    fclose(fid);

    if ischar(line1) && strncmp(line1, '#fs,', 4)
        Fs = str2double(strrep(line1, '#fs,', ''));
    end
    if ischar(line2) && strncmp(line2, '#leads,', 7)
        lead_names = strsplit(strrep(line2, '#leads,', ''), ',');
    end

    try
        sig = dlmread(tmp_csv, ',', 2, 0);
    catch
        sig = [];
    end

    delete(tmp_csv);

    if isempty(sig)
        err_msg = 'Sinyal verisi bos (Python wfdb).';
        return;
    end

    max_leads = min(12, size(sig, 2));
    sig = sig(:, 1:max_leads);
    if isempty(lead_names)
        lead_names = derive_lead_names([], max_leads);
    end

    source_label = 'Python wfdb';
    ok = true;
end

function py_exec = find_python_exec()
    candidates = {'python', 'python3'};
    py_exec = '';
    for c = 1:numel(candidates)
        cmd = sprintf('%s --version', candidates{c});
        status = system(cmd);
        if status == 0
            py_exec = candidates{c};
            return;
        end
    end
end

function lead_names = derive_lead_names(siginfo, max_leads)
    lead_names = cell(1, max_leads);
    if ~isempty(siginfo)
        if isstruct(siginfo) && isfield(siginfo, 'Description')
            for k = 1:max_leads
                lead_names{k} = strtrim(siginfo(k).Description);
            end
        elseif isstruct(siginfo) && isfield(siginfo, 'SignalName')
            for k = 1:max_leads
                lead_names{k} = strtrim(siginfo(k).SignalName);
            end
        end
    end
    for k = 1:max_leads
        if isempty(lead_names{k})
            lead_names{k} = sprintf('Lead %d', k);
        end
    end
end

function class_mask = build_multiclass_presence_mask(scp_codes_raw, target_codes)
    % BUILD_MULTICLASS_PRESENCE_MASK  Bir veya daha fazla arrhythmia kodu icin kayit maskesi.
    %
    %   class_mask(i) = true  <=>  kayit i icin scp_codes_raw{i} stringinde
    %   target_codes listesinden (SR, AFIB, PVC, vb.) en az bir kod bulunur.
    %
    %   Performans icin, hedeflenen arrhythmia siniflari icin N x K boyutlu
    %   bir boolean matris (arr_presence) once parse edilip persistent olarak
    %   saklanir; filtre degistikce yalnizca bu matris uzerinde vektorel
    %   islemler yapilir.

    num_records = numel(scp_codes_raw);
    class_mask = false(num_records, 1);

    if isempty(target_codes)
        return;
    end
    if ischar(target_codes)
        target_codes = {target_codes};
    end

    % Hedef arrhythmia siniflari (dashboard'da kullanilan 10 sinif)
    arrhythmia_classes = {'SR', 'AFIB', 'STACH', 'SARRH', 'PVC', ...
                          'PAC', 'AFLT', 'SBRAD', 'SVTAC', 'NORM'};
    num_classes = numel(arrhythmia_classes);

    persistent cached_raw arr_presence;

    if isempty(cached_raw) || ~isequal(size(cached_raw), size(scp_codes_raw))
        % Ilk cagri veya farkli boyutta scp_codes_raw: boolean matrisi yeniden olustur
        arr_presence = false(num_records, num_classes);
        pattern = '''([A-Z0-9_/]+)''\s*:\s*([0-9.]+)';

        for record_index = 1:num_records
            codes_string = scp_codes_raw{record_index};
            if isempty(codes_string)
                continue;
            end
            tokens = regexp(codes_string, pattern, 'tokens');
            if isempty(tokens)
                continue;
            end
            for token_index = 1:numel(tokens)
                token_pair = tokens{token_index};
                code = token_pair{1};
                class_idx = find(strcmp(arrhythmia_classes, code), 1);
                if ~isempty(class_idx)
                    arr_presence(record_index, class_idx) = true;
                end
            end
        end

        cached_raw = scp_codes_raw;
    end

    % Hedef siniflari ilgili sutunlara esle
    [~, col_indices] = ismember(target_codes, arrhythmia_classes);
    col_indices = col_indices(col_indices > 0);
    if isempty(col_indices)
        class_mask = false(num_records, 1);
        return;
    end

    class_mask = any(arr_presence(:, col_indices), 2);
end
