function [parameter_vector, loss_history] = train_logistic_regression(feature_matrix, label_vector, learning_rate, num_iterations)
% TRAIN_LOGISTIC_REGRESSION  Basit lojistik regresyon egitimi (gradient descent).
%
%   [parameter_vector, loss_history] =
%       TRAIN_LOGISTIC_REGRESSION(feature_matrix, label_vector, learning_rate, num_iterations)
%
%   feature_matrix  : [numenek x num_ozellik] seklinde matris
%   label_vector    : [numenek x 1] seklinde 0/1 etiket vektoru
%   learning_rate   : Adim buyuklugu (or. 0.1)
%   num_iterations  : Iterasyon sayisi (or. 200)

    num_examples = size(feature_matrix, 1);
    num_features = size(feature_matrix, 2);

    parameter_vector = zeros(num_features, 1);
    loss_history = zeros(num_iterations, 1);

    for iteration_index = 1:num_iterations
        linear_output = feature_matrix * parameter_vector;
        predicted_probabilities = 1 ./ (1 + exp(-linear_output));

        gradient_vector = (1 / num_examples) * (feature_matrix' * (predicted_probabilities - label_vector));
        parameter_vector = parameter_vector - learning_rate * gradient_vector;

        % Kayıp (binary cross-entropy)
        epsilon = 1e-10;
        clipped_predictions = min(max(predicted_probabilities, epsilon), 1 - epsilon);
        loss_value = (-1 / num_examples) * ( ...
            label_vector' * log(clipped_predictions) + ...
            (1 - label_vector)' * log(1 - clipped_predictions) ...
        );

        loss_history(iteration_index) = loss_value;
    end
end

