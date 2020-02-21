clear;

%% read input and set arguments
output_figure_dir = './figures/';

P = csvread('data/test_P.csv');
csvs = dir('h');

for i = 1:length(csvs)
    if strcmp(csvs(i).name, '.') || strcmp(csvs(i).name, '..') || contains(csvs(i).name,'final')
        continue;
    end
    file_name = csvs(i).name;
    h = csvread(strcat('./h/',file_name));
    ind = int64(readmatrix(strcat('./ind/',file_name)));
    ind = ind + 1;
    X = readmatrix(strcat('./volP/',file_name));
    numP = size(P,1);
    dim = size(P,2);
    bat_size = 20000;
    num_X = length(ind);
    num_bat = num_X/bat_size;

    %% compute colors
    for ii = 1:num_bat
        x_bat = X((ii-1)*bat_size+1:ii*bat_size,:);
        ind_bat = ind((ii-1)*bat_size+1:ii*bat_size);
    end

    %% plot results
    cmap = hsv(numP);
    c = cmap(ind,:);

    f = figure('visible','off');
    hold on
    for ii = 1:num_bat
        x_bat = X((ii-1)*bat_size+1:ii*bat_size,:);
        x_bat = x_bat - .5;
        scatter(x_bat(:,1),x_bat(:,2),5,c((ii-1)*bat_size+1:ii*bat_size,:),'filled');
    end

    C = strsplit(file_name,'.');
    name = C{1};
    im_id = str2num(name);
    im_name = num2str(im_id, '%04d');
    
    saveas(f, strcat(output_figure_dir,im_name),'png');
    close(f)

    fprintf('%i/%i...\n',i, length(csvs))
end
