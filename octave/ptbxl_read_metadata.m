function metadata = ptbxl_read_metadata(base_directory)
% PTBXL_READ_METADATA  PTB-XL meta verisini GNU Octave icin okur.
%
%   metadata = PTBXL_READ_METADATA()
%   metadata = PTBXL_READ_METADATA(base_directory)
%
%   Geri donus:
%     metadata.headers           : Hucre dizisi, sutun basliklari
%     metadata.rows              : Ham hucre dizisi verisi (stringler)
%     metadata.num_records       : Kayit sayisi
%     metadata.patient_id        : Hasta ID (double, NaN dahil)
%     metadata.age               : Yas (double, NaN dahil)
%     metadata.sex               : Cinsiyet (double, 0/1, NaN dahil)
%     metadata.height            : Boy (double, NaN dahil)
%     metadata.weight            : Kilo (double, NaN dahil)
%     metadata.scp_codes_raw     : SCP kod stringleri (hucre dizisi)
%     metadata.filename_hr       : 500 Hz WFDB kayit yolu (hucre)
%     metadata.filename_lr       : 100 Hz WFDB kayit yolu (hucre)
%
%   Notlar:
%   - Bu fonksiyon, PTB-XL'in orijinal CSV dosyasi olan
%     ptbxl_database.csv dosyasini okur.
%   - CSV parsing icin Octave'in "io" paketinden csv2cell fonksiyonuna
%     ihtiyac duyar.
%
%   Ornek kullanim:
%       metadata = ptbxl_read_metadata();
%       mean_age = nanmean(metadata.age);
%
%   Baris'in arrhythmia_detection projesi icin GNU Octave desteği.

    if nargin < 1 || isempty(base_directory)
        base_directory = fullfile('..', 'dataset', 'physionet.org', 'files', 'ptb-xl', '1.0.3');
    end

    metadata_file_path = fullfile(base_directory, 'ptbxl_database.csv');

    if ~exist(metadata_file_path, 'file')
        error(['PTB-XL metadata dosyasi bulunamadi: ', metadata_file_path, ...
               '\nDataset klasor yapisinin README''de anlatildigi gibi oldugundan emin olun.']);
    end

    % io paketini yuklemeyi dene
    try
        pkg('load', 'io');
    catch
        error(['Octave "io" paketi yuklenemedi.\n', ...
               'Lutfen once su komutlari calistir:\n', ...
               '  pkg install -forge io   (gerekirse)\n', ...
               '  pkg load io']);
    end

    % CSV'yi oku
    csv_cell = csv2cell(metadata_file_path);
    if isempty(csv_cell)
        error('ptbxl_database.csv bos gorunuyor.');
    end

    header_row = csv_cell(1, :);
    data_rows = csv_cell(2:end, :);

    metadata.headers = header_row;
    metadata.rows = data_rows;
    metadata.num_records = size(data_rows, 1);

    % Ilgili sutun indekslerini bul
    patient_id_index = find(strcmp(header_row, 'patient_id'));
    age_index = find(strcmp(header_row, 'age'));
    sex_index = find(strcmp(header_row, 'sex'));
    height_index = find(strcmp(header_row, 'height'));
    weight_index = find(strcmp(header_row, 'weight'));
    scp_codes_index = find(strcmp(header_row, 'scp_codes'));
    filename_hr_index = find(strcmp(header_row, 'filename_hr'));
    filename_lr_index = find(strcmp(header_row, 'filename_lr'));

    if isempty(patient_id_index) || isempty(age_index) || isempty(sex_index) || isempty(height_index) || ...
       isempty(weight_index) || isempty(scp_codes_index)
        error('Beklenen sutunlardan biri bulunamadi (patient_id, age, sex, height, weight, scp_codes).');
    end

    % Hucreleri sayisal vektorlere cevir
    patient_id_values = convert_column_to_double(data_rows(:, patient_id_index));
    age_values = convert_column_to_double(data_rows(:, age_index));
    sex_values = convert_column_to_double(data_rows(:, sex_index));
    height_values = convert_column_to_double(data_rows(:, height_index));
    weight_values = convert_column_to_double(data_rows(:, weight_index));

    scp_codes_raw = data_rows(:, scp_codes_index);
    if ~isempty(filename_hr_index)
        filename_hr = data_rows(:, filename_hr_index);
    else
        filename_hr = {};
    end
    if ~isempty(filename_lr_index)
        filename_lr = data_rows(:, filename_lr_index);
    else
        filename_lr = {};
    end

    metadata.patient_id = patient_id_values;
    metadata.age = age_values;
    metadata.sex = sex_values;
    metadata.height = height_values;
    metadata.weight = weight_values;
    metadata.scp_codes_raw = scp_codes_raw;
    metadata.filename_hr = filename_hr;
    metadata.filename_lr = filename_lr;
end

function numeric_vector = convert_column_to_double(column_cells)
% CONVERT_COLUMN_TO_DOUBLE  Hucre sutununu double vektore cevirir (boslar icin NaN).

    num_rows = numel(column_cells);
    numeric_vector = NaN(num_rows, 1);

    for row_index = 1:num_rows
        cell_value = column_cells{row_index};
        if isempty(cell_value)
            numeric_value = NaN;
        elseif isnumeric(cell_value)
            numeric_value = cell_value;
        elseif ischar(cell_value)
            numeric_value = str2double(cell_value);
            if isnan(numeric_value)
                numeric_value = NaN;
            end
        else
            % Diger tipler icin, stringe cevirip tekrar dene
            try
                numeric_value = str2double(char(cell_value));
            catch
                numeric_value = NaN;
            end
        end
        numeric_vector(row_index) = numeric_value;
    end
end
