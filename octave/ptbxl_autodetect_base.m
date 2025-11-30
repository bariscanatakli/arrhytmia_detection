function info = ptbxl_autodetect_base(base_directory)
% PTBXL_AUTODETECT_BASE  PTB-XL dataset yolunu otomatik tespit eder.
%
%   info = PTBXL_AUTODETECT_BASE()
%   info = PTBXL_AUTODETECT_BASE(base_directory)
%
%   Donus yapisi:
%     info.base_path         : Secilen temel PTB-XL dizini
%     info.has_database      : ptbxl_database.csv bulundu mu
%     info.has_statements    : scp_statements.csv bulundu mu
%     info.has_records100    : records100/ dizini bulundu mu
%     info.has_records500    : records500/ dizini bulundu mu
%     info.ready             : ptbxl_database.csv + scp_statements.csv + (records100 veya records500) var mi
%     info.missing           : Eksik ogelerin hucre dizisi
%
%   Arama sirasi:
%     1) Fonksiyona verilen base_directory (varsa)
%     2) Ortam degiskeni PTBXL_BASE (varsa)
%     3) Calisma dizinine gore dataset/physionet.org/files/ptb-xl/1.0.3/
%     4) Bu dosyanin konumuna gore ../dataset/physionet.org/files/ptb-xl/1.0.3/
%     5) Calisma dizininin bir ustu ../dataset/physionet.org/files/ptb-xl/1.0.3/

    candidates = {};

    if nargin >= 1 && ~isempty(base_directory)
        candidates{end + 1} = base_directory;
    end

    env_base = getenv('PTBXL_BASE');
    if ~isempty(env_base)
        candidates{end + 1} = env_base;
    end

    cwd_default = fullfile(pwd, 'dataset', 'physionet.org', 'files', 'ptb-xl', '1.0.3');
    candidates{end + 1} = cwd_default;

    script_dir = fileparts(mfilename('fullpath'));
    script_default = fullfile(script_dir, '..', 'dataset', 'physionet.org', 'files', 'ptb-xl', '1.0.3');
    candidates{end + 1} = script_default;

    parent_default = fullfile(pwd, '..', 'dataset', 'physionet.org', 'files', 'ptb-xl', '1.0.3');
    candidates{end + 1} = parent_default;

    % Duz layout (dataset/ptbxl_database.csv) icin ek adaylar
    flat_repo_default = fullfile(pwd, 'dataset');
    candidates{end + 1} = flat_repo_default;

    flat_script_default = fullfile(script_dir, '..', 'dataset');
    candidates{end + 1} = flat_script_default;

    flat_parent_default = fullfile(pwd, '..', 'dataset');
    candidates{end + 1} = flat_parent_default;

    % Ayni yolu iki kez denememek icin benzersiz hale getir
    candidates = unique(candidates);

    best_status = init_status('');
    for c = 1:numel(candidates)
        status = inspect_candidate(candidates{c});
        if status.ready
            info = status;
            return;
        end
        % Hangisi daha cok dosya tutturduysa onu sakla
        if status.score > best_status.score
            best_status = status;
        end
    end

    info = best_status;
end

function status = inspect_candidate(base_path)
    status = init_status(base_path);

    database_path = fullfile(base_path, 'ptbxl_database.csv');
    statements_path = fullfile(base_path, 'scp_statements.csv');
    records100_path = fullfile(base_path, 'records100');
    records500_path = fullfile(base_path, 'records500');

    status.has_database = exist(database_path, 'file') == 2;
    status.has_statements = exist(statements_path, 'file') == 2;
    status.has_records100 = exist(records100_path, 'dir') == 7;
    status.has_records500 = exist(records500_path, 'dir') == 7;

    status.ready = status.has_database && status.has_statements && ...
                   (status.has_records100 || status.has_records500);

    status.missing = {};
    if ~status.has_database
        status.missing{end + 1} = 'ptbxl_database.csv';
    end
    if ~status.has_statements
        status.missing{end + 1} = 'scp_statements.csv';
    end
    if ~(status.has_records100 || status.has_records500)
        status.missing{end + 1} = 'records100/ veya records500/';
    end

    status.score = status.has_database + status.has_statements + ...
                   status.has_records100 + status.has_records500;
end

function status = init_status(base_path)
    status.base_path = base_path;
    status.has_database = false;
    status.has_statements = false;
    status.has_records100 = false;
    status.has_records500 = false;
    status.ready = false;
    status.missing = {};
    status.score = -Inf;
end
