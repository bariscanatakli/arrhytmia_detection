function ptbxl_simple_model(base_directory)
% PTBXL_SIMPLE_MODEL  PTB-XL meta verisi uzerinde basit lojistik regresyon.
%
%   PTBXL_SIMPLE_MODEL()
%   PTBXL_SIMPLE_MODEL(base_directory)
%
%   Bu ornek, sadece metadata (yas, cinsiyet, kilo) kullanarak
%   kaydin "normal" olup olmadigini tahmin etmeye calisir.
%   Klinik olarak sinirli ama veri bilimi acisindan guzel bir baslangic
%   ornegidir.

    if nargin < 1 || isempty(base_directory)
        base_directory = fullfile('..', 'dataset', 'physionet.org', 'files', 'ptb-xl', '1.0.3');
    end

    fprintf('PTB-XL metadata okunuyor (model icin)...\n');
    metadata = ptbxl_read_metadata(base_directory);

    normal_mask = build_normal_label_mask(metadata.scp_codes_raw);

    % Ozellik matrisi: yas, cinsiyet, kilo
    % PTB-XL'de boy (height) cok eksik oldugu icin buraya dahil edilmiyor.
    feature_matrix = [metadata.age, metadata.sex, metadata.weight];

    % Eksik degerleri olan satirlari cikar
    valid_row_mask = all(~isnan(feature_matrix), 2);
    cleaned_feature_matrix = feature_matrix(valid_row_mask, :);
    cleaned_label_vector = double(normal_mask(valid_row_mask));

    fprintf('Gecerli ornek sayisi (eksiksiz metadata): %d\n', numel(cleaned_label_vector));

    % Ozellikleri normalize et (ortalama 0, standart sapma 1)
    [normalized_feature_matrix, feature_means, feature_stds] = normalize_features(cleaned_feature_matrix);

    % Bias terimi ekle
    num_valid_examples = size(normalized_feature_matrix, 1);
    bias_column = ones(num_valid_examples, 1);
    design_matrix = [bias_column, normalized_feature_matrix];

    % Egitim / test bolunmesi
    random_seed = 42;
    rand('seed', random_seed); %#ok<RAND>
    random_permutation = randperm(num_valid_examples);

    train_ratio = 0.8;
    num_train = floor(train_ratio * num_valid_examples);

    train_indices = random_permutation(1:num_train);
    test_indices = random_permutation(num_train + 1:end);

    design_matrix_train = design_matrix(train_indices, :);
    label_vector_train = cleaned_label_vector(train_indices);

    design_matrix_test = design_matrix(test_indices, :);
    label_vector_test = cleaned_label_vector(test_indices);

    fprintf('Egitim ornek sayisi: %d\n', numel(label_vector_train));
    fprintf('Test ornek sayisi   : %d\n', numel(label_vector_test));

    % Modeli egit
    learning_rate = 0.1;
    num_iterations = 200;
    fprintf('Lojistik regresyon egitiliyor...\n');
    [parameter_vector, loss_history] = train_logistic_regression(design_matrix_train, label_vector_train, learning_rate, num_iterations);

    % Kayıp grafiği
    create_or_show_figure(10);
    plot(1:num_iterations, loss_history, 'LineWidth', 2);
    xlabel('Iterasyon');
    ylabel('Kayıp (loss)');
    title('Egitim kayip egirisi (lojistik regresyon)');

    % Test seti performansi
    predicted_probabilities_test = 1 ./ (1 + exp(-(design_matrix_test * parameter_vector)));
    predicted_labels_test = predicted_probabilities_test >= 0.5;

    accuracy_value = mean(predicted_labels_test == label_vector_test);

    fprintf('Test dogrulugu: %.4f\n', accuracy_value);

    % Confusion matrix
    confusion_matrix = compute_confusion_matrix(label_vector_test, predicted_labels_test);
    fprintf('Karışıklık matrisi [TN FP; FN TP]:\n');
    disp(confusion_matrix);

    % Model parametrelerini yazdir
    fprintf('Bias + ozellik agirliklari:\n');
    feature_names = {'bias', 'age', 'sex', 'weight'};
    for parameter_index = 1:numel(feature_names)
        fprintf('  %s: %.4f\n', feature_names{parameter_index}, parameter_vector(parameter_index));
    end

    fprintf('\nModel, sadece metadata uzerine kurulmus basit bir ornektir.\n');
    fprintf('Gercek arrhythmia tespiti icin sinyal tabanli ozellikler eklenmelidir.\n');
end

function normal_mask = build_normal_label_mask(scp_codes_raw)
% BUILD_NORMAL_LABEL_MASK  scp_codes stringlerine gore normal maskesi uretir.

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

function [normalized_matrix, feature_means, feature_stds] = normalize_features(feature_matrix)
% NORMALIZE_FEATURES  Ozellikleri ortalama 0, std 1 olacak sekilde olceklendirir.

    feature_means = mean(feature_matrix, 1);
    feature_stds = std(feature_matrix, 0, 1);

    feature_stds(feature_stds == 0) = 1.0; % sabit ozellik olmasin

    normalized_matrix = (feature_matrix - feature_means) ./ feature_stds;
end

function confusion_matrix = compute_confusion_matrix(true_labels, predicted_labels)
% COMPUTE_CONFUSION_MATRIX  2x2 karışıklık matrisi döndürür.

    true_negative = sum((true_labels == 0) & (predicted_labels == 0));
    false_positive = sum((true_labels == 0) & (predicted_labels == 1));
    false_negative = sum((true_labels == 1) & (predicted_labels == 0));
    true_positive = sum((true_labels == 1) & (predicted_labels == 1));

    confusion_matrix = [true_negative, false_positive; false_negative, true_positive];
end

function create_or_show_figure(figure_id)
% CREATE_OR_SHOW_FIGURE  Var olan figur penceresini getirir veya yenisini acar.

    if ishghandle(figure_id)
        figure(figure_id);
        clf;
    else
        figure(figure_id);
    end
end
