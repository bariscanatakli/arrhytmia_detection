function ptbxl_patient_eda(base_directory)
% PTBXL_PATIENT_EDA  PTB-XL icin hasta-bazli ve missingness EDA'si.
%
%   PTBXL_PATIENT_EDA()
%   PTBXL_PATIENT_EDA(base_directory)
%
%   Konsola su ozetleri yazar:
%     - Kişi basina kayit sayisi dagilimi (min / medyan / max)
%     - En az bir AFIB/PVC/NORM kaydi olan hasta oranlari
%     - Weight dolu vs eksik gruplar icin yas ve cinsiyet ozetleri

    if nargin < 1 || isempty(base_directory)
        base_directory = fullfile('..', 'dataset', 'physionet.org', 'files', 'ptb-xl', '1.0.3');
    end

    metadata = ptbxl_read_metadata(base_directory);

    patient_ids = metadata.patient_id;
    valid_patient_mask = ~isnan(patient_ids);
    patient_ids = patient_ids(valid_patient_mask);

    unique_ids = unique(patient_ids);
    num_patients = numel(unique_ids);

    counts_per_patient = zeros(num_patients, 1);
    for idx = 1:num_patients
        counts_per_patient(idx) = sum(patient_ids == unique_ids(idx));
    end

    fprintf('--- Hasta-bazli EDA ---\n');
    fprintf('Toplam hasta sayisi (patient_id != NaN): %d\n', num_patients);
    fprintf('Kisi basina kayit sayisi (min / median / max): %d / %.1f / %d\n', ...
            min(counts_per_patient), median(counts_per_patient), max(counts_per_patient));

    scp_raw = metadata.scp_codes_raw(valid_patient_mask);

    afib_patient_mask = false(num_patients, 1);
    pvc_patient_mask = false(num_patients, 1);
    norm_patient_mask = false(num_patients, 1);
    pattern = '''([A-Z0-9_/]+)''\s*:\s*([0-9.]+)';

    for idx = 1:num_patients
        pid = unique_ids(idx);
        records_for_patient = scp_raw(patient_ids == pid);
        for r_index = 1:numel(records_for_patient)
            codes_string = records_for_patient{r_index};
            if isempty(codes_string)
                continue;
            end
            tokens = regexp(codes_string, pattern, 'tokens');
            if isempty(tokens)
                continue;
            end
            present_codes = cellfun(@(t) t{1}, tokens, 'UniformOutput', false);
            if any(strcmp(present_codes, 'AFIB'))
                afib_patient_mask(idx) = true;
            end
            if any(strcmp(present_codes, 'PVC'))
                pvc_patient_mask(idx) = true;
            end
            if any(strcmp(present_codes, 'NORM'))
                norm_patient_mask(idx) = true;
            end
        end
    end

    fprintf('En az bir AFIB kaydi olan hastalar  : %d (%.2f %%)\n', ...
            sum(afib_patient_mask), 100 * sum(afib_patient_mask) / num_patients);
    fprintf('En az bir PVC kaydi olan hastalar   : %d (%.2f %%)\n', ...
            sum(pvc_patient_mask), 100 * sum(pvc_patient_mask) / num_patients);
    fprintf('En az bir NORM kaydi olan hastalar  : %d (%.2f %%)\n', ...
            sum(norm_patient_mask), 100 * sum(norm_patient_mask) / num_patients);

    fprintf('\n--- Missingness (weight) analizi ---\n');
    weight_values = metadata.weight;
    age_values = metadata.age;
    sex_values = metadata.sex;

    has_weight = ~isnan(weight_values) & ~isnan(age_values) & ~isnan(sex_values);
    missing_weight = isnan(weight_values) & ~isnan(age_values) & ~isnan(sex_values);

    fprintf('Weight mevcut olan kayitlar         : %d (%.2f %%)\n', ...
            sum(has_weight), 100 * sum(has_weight) / metadata.num_records);
    fprintf('Weight eksik olan kayitlar          : %d (%.2f %%)\n', ...
            sum(missing_weight), 100 * sum(missing_weight) / metadata.num_records);

    if any(has_weight)
        ages_has = age_values(has_weight);
        sexes_has = sex_values(has_weight);
        fprintf('  Weight mevcut grupta yas (ort/med): %.2f / %.2f\n', mean(ages_has), median(ages_has));
        fprintf('  Weight mevcut grupta cinsiyet (E/K): %d / %d\n', ...
                sum(sexes_has == 1), sum(sexes_has == 0));
    end

    if any(missing_weight)
        ages_missing = age_values(missing_weight);
        sexes_missing = sex_values(missing_weight);
        fprintf('  Weight eksik grupta yas (ort/med) : %.2f / %.2f\n', mean(ages_missing), median(ages_missing));
        fprintf('  Weight eksik grupta cinsiyet (E/K): %d / %d\n', ...
                sum(sexes_missing == 1), sum(sexes_missing == 0));
    end

    fprintf('------------------------------------\n\n');
end

