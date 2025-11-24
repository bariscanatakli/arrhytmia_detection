function [scp_code_list, scp_counts] = ptbxl_label_frequencies(scp_codes_raw)
% PTBXL_LABEL_FREQUENCIES  SCP kodlarinin frekans dagilimini hesaplar.
%
%   [scp_code_list, scp_counts] = PTBXL_LABEL_FREQUENCIES(scp_codes_raw)
%
%   Her kayit icin, scp_codes stringinden kodlari cikarir ve her kodun
%   kac farkli kayitta gorundugunu sayar. Bir kayit icinde ayni kod
%   birden fazla gecse bile yalnizca bir kez sayilir.

    num_records = numel(scp_codes_raw);
    scp_code_list = {};
    scp_counts = [];

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

        seen_codes = {};

        for token_index = 1:numel(tokens)
            token_pair = tokens{token_index};
            code = token_pair{1};

            if any(strcmp(seen_codes, code))
                continue;
            end
            seen_codes{end + 1} = code;

            existing_index = find(strcmp(scp_code_list, code));
            if isempty(existing_index)
                scp_code_list{end + 1} = code;
                scp_counts(end + 1) = 1;
            else
                scp_counts(existing_index) = scp_counts(existing_index) + 1;
            end
        end
    end
end

