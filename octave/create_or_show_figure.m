function create_or_show_figure(figure_id)
% CREATE_OR_SHOW_FIGURE  Var olan figur penceresini getirir veya yenisini acar.
%
%   Bu helper hem temel EDA hem de ileri duzey
%   grafik fonksiyonlari tarafindan kullanilir.

    if ishghandle(figure_id)
        figure(figure_id);
        clf;
    else
        figure(figure_id);
    end
end

