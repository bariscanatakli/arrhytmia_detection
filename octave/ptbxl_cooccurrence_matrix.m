function co_matrix = ptbxl_cooccurrence_matrix(scp_codes_raw, selected_codes)
% PTBXL_COOCCURRENCE_MATRIX  Secilen SCP kodlari icin birlikte-gorunme matrisi.
%
%   co_matrix = PTBXL_COOCCURRENCE_MATRIX(scp_codes_raw, selected_codes)
%
%   scp_codes_raw   : ptbxl_read_metadata'dan gelen scp_codes_raw hucre dizisi
%   selected_codes  : ornegin {'AFIB','PVC','NORM',...} seklinde kod listesi
%
%   co_matrix(i,j), i ve j kodlarinin ayni kayitta kac kez birlikte
%   gorundugunu gosterir.

    num_codes = numel(selected_codes);
    co_matrix = zeros(num_codes, num_codes);

    pattern = '''([A-Z0-9_/]+)''\s*:\s*([0-9.]+)';

    num_records = numel(scp_codes_raw);
    for record_index = 1:num_records
        codes_string = scp_codes_raw{record_index};
        if isempty(codes_string)
            continue;
        end

        tokens = regexp(codes_string, pattern, 'tokens');
        if isempty(tokens)
            continue;
        end

        present = false(num_codes, 1);
        for token_index = 1:numel(tokens)
            token_pair = tokens{token_index};
            code = token_pair{1};
            idx = find(strcmp(selected_codes, code));
            if ~isempty(idx)
                present(idx) = true;
            end
        end

        for i = 1:num_codes
            if ~present(i)
                continue;
            end
            for j = 1:num_codes
                if present(j)
                    co_matrix(i, j) = co_matrix(i, j) + 1;
                end
            end
        end
    end
end

