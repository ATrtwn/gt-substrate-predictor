function PredictAcceptorInteraction_DT(interactionFile, query, outFile)

Data = read_acceptor_interaction_file(interactionFile);
%thres = (Data.X(:, [5, 7:20])>0);
%Data.X(:, [5, 7:20]) = thres;
thres = (Data.X(:,5) > 0);
Data.X(:,5) = thres;
%query.family   % scalar: 1..4
query.logP;     % scalar
query.area;    % scalar
query.vol;      % scalar
query.cooh;     % 0/1
query.numOH;    % scalar
% query.groups_raw  % 1×N vector, only the family-specific groups

% groups = zeros(1,14);   % all zeros initially

%switch query.family
    %case 1   % flavonoids
        %groups(1:6) = query.groups_raw(:)';
    %case 2   % cm
        %groups(7:8) = query.groups_raw(:)';
    %case 3   % ck
        %groups(9:11) = query.groups_raw(:)';
    %case 4   % cn
        %groups(12:14) = query.groups_raw(:)';
    %otherwise
        %error('Unknown family value');
%end

X_new = [ ...
 %query.family, ... % scalar
 query.logP, ...
 query.area, ...
 query.vol, ...
 query.cooh, ... % 0/1
 query.numOH, ...
 %groups ... % 1×14 binary vector
];

% train all of classification trees
[dum, numE] = size(Data.E);
for enz=1:numE,
  L = Data.E(:,enz);
  I = find(L<2);
  % t = fitctree(Data.X(I,:), L(I) , 'CategoricalPredictors', [1, 5, 7:20], 'SplitCriterion', 'deviance', 'MinLeafSize', 1);
  t = fitctree(Data.X(I,:), L(I) , 'CategoricalPredictors', [1, 5], 'SplitCriterion', 'deviance', 'MinLeafSize', 1);
  %treedisp(t ,'names', handles.Data.Xnames)
  %[sfit, node, cnames] = treeval(t,handles.Data.X(I,:));      % find assigned class numbers
  %prate = sum(str2num(char(cnames))==L(I))/length(I)*100;   % com
  class = predict(t, X_new);

    if class == 1
        cl = 'Yes';
        s_next = [char(Data.Enames{enz}), '     ', cl];
    else
        cl = 'No';
        s_next = [char(Data.Enames{enz}), '                 ', cl];
    end

    S{enz,1} = s_next;
    %fprintf(s_next);
    %fprintf('\n');
end

% ---- AUTO-SAVE PREDICTION TO CSV ----
% Assume query_name is passed or set manually
if ~isfield(query, 'name')
    query_name = 'UnknownSubstrate';
else
    query_name = query.name;
end

% Open file for writing
fid = fopen(outFile, 'w');
if fid == -1
    error('Cannot open file %s for writing', outFile);
end

% write substrate name as header
fprintf(fid, 'Substrate: %s\n', query_name);

% Write each enzyme prediction
for i = 1:numel(S)
    fprintf(fid, '%s\n', S{i});
end

% Close file
fclose(fid);
% ------------------------------------

