function categories = ptbxl_scp_categories(base_directory)
% PTBXL_SCP_CATEGORIES  scp_statements.csv dosyasindan SCP kod kategorilerini okur.
%
%   categories = PTBXL_SCP_CATEGORIES(base_directory)
%
%   Donus yapisi:
%     categories.codes             : SCP kodlari (hucre dizisi)
%     categories.diagnostic_class  : Her kod icin diagnostic_class (hucre dizisi, bos olabilir)
%
%   Not: Sadece metadata icin kullanilir; ek paket gerektirmez (csv2cell, io).

    if nargin < 1 || isempty(base_directory)
        base_directory = fullfile('..', 'dataset', 'physionet.org', 'files', 'ptb-xl', '1.0.3');
    end

    statements_path = fullfile(base_directory, 'scp_statements.csv');
    if ~exist(statements_path, 'file')
        error('scp_statements.csv bulunamadi: %s', statements_path);
    end

    try
        pkg('load', 'io');
    catch
        error('Octave "io" paketi (csv2cell icin) yuklenemedi.');
    end

    csv_cell = csv2cell(statements_path);
    if isempty(csv_cell)
        error('scp_statements.csv bos gorunuyor.');
    end

    header_row = csv_cell(1, :);
    data_rows = csv_cell(2:end, :);

    % PTB-XL orijinal dosyasinda ilk kolon SCP kodu ama header bos;
    % bazi surumlerde "scp_code" olarak isimlendirilebilir. Ikisine de hazir ol.
    code_index = find(strcmp(header_row, 'scp_code'));
    if isempty(code_index)
        code_index = 1;
    end
    diag_class_index = find(strcmp(header_row, 'diagnostic_class'));

    if isempty(diag_class_index)
        error('scp_statements.csv icinde beklenen sutun bulunamadi: diagnostic_class');
    end

    num_rows = size(data_rows, 1);
    codes = cell(num_rows, 1);
    diagnostic_class = cell(num_rows, 1);

    for r = 1:num_rows
        codes{r} = data_rows{r, code_index};
        diagnostic_class{r} = data_rows{r, diag_class_index};
        if isempty(diagnostic_class{r})
            diagnostic_class{r} = '';
        end
    end

    categories.codes = codes;
    categories.diagnostic_class = diagnostic_class;
end
