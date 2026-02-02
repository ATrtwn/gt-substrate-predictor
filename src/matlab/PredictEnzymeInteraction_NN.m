function PredictEnzymeInteraction_NN(interactionFile, seqsFile, newSeqFile, outFile)

Mols = read_enzyme_interaction_file(interactionFile);
[numM, numE] = size(Mols.X);

PS = fastaread(seqsFile);
numPS = length(PS);

newPS = fastaread(newSeqFile);
numNewPS = length(newPS);

Map = get_Fasta_Data_Map(PS, Mols.Enames);

% find the nearest neighbour using swalign
 for i =1:numPS,
          sw = swalign(PS(i).Sequence, newPS(1).Sequence);
          S(i) = sw;
 end

 [sorted, inds] = sort(S, 'descend');
 % Print top 10
fprintf('Top 10 nearest enzymes for query %s:\n', newPS(1).Header);
fprintf('Rank\tEnzyme_ID\tScore\n');
for k = 1:min(10, length(sorted))
    fprintf('%d\t%d\t\t%.4f\n', k, inds(k), sorted(k));
end
 enz_ind = find_indexes(Map, inds(1));  % get the interaction data index for the most similar protein

if (enz_ind > 0)

    %% Go through the data in Mols.X(:, enz_ind) and make a prediction of
    %% either 'yes', 'no' or preidction. State which enzyme is used for the
    %% prediction of each interaction.

    Pred{1,1} = ['Using nearest protein ', char(Mols.Enames{enz_ind(1)}), ' for prediction (Smith-Waterman alignment score ', num2str(S(inds(1))), ')' ];
    Pred{1,2} = [''];
    names = Mols.names{1};

    counter =3;
    for m=1:numM,
        s_next = 'Not found';
        val = Mols.X(m, enz_ind);
        switch val
            case 0
                class = 'No';
                s_next = [char(names(m)), '     ', class];
                Pred{counter,1} = s_next;
                counter = counter+1;
                %fprintf(s_next);
                %fprintf('\n');
            case 1
                class = 'Yes';
                %s_next = [char(Mols.names{m}), '                 ', class];
                s_next = [char(names(m)), '     ', class];
                Pred{counter,1} = s_next;
                counter = counter+1;
                %fprintf(s_next);
                %fprintf('\n');
        end
    end

    enz_ind2 = find_indexes(Map, inds(2));

    if (enz_ind2 > 0)
         Pred{counter,2} = [''];
         Pred{counter+1,1} = ['Using second nearest protein ', char(Mols.Enames{enz_ind2}), ' for prediction (Smith-Waterman alignment score ', num2str(S(inds(2))), ')' ];
         Pred{counter+2,2} = [''];
         counter  = counter +3;

              for m=1:numM,
                s_next = 'Not found';
                val = Mols.X(m, enz_ind);
                new_val = Mols.X(m, enz_ind2);
                switch val
                    case 2
                        switch new_val
                          case 0
                            class = 'No';
                            s_next = [char(names(m)), '     ', class];
                            Pred{counter,1} = s_next;
                            counter = counter+1;
                            %fprintf(s_next);
                            %fprintf('\n');
                          case 1
                            class = 'Yes';
                            %s_next = [char(Mols.names{m}), '                 ', class];
                            s_next = [char(names(m)), '     ', class];
                            Pred{counter,1} = s_next;
                            counter = counter+1;
                            %fprintf(s_next);
                            %fprintf('\n');
                          otherwise
                            class = 'Missing';
                            s_next = [char(names(m)), '     ', class];
                            Pred{counter,1} = s_next;
                            counter = counter + 1;
                            %fprintf(s_next);
                            %fprintf('\n');
                         end
                    case 3
                        switch new_val
                          case 0
                            class = 'No';
                            s_next = [char(names(m)), '     ', class];
                            Pred{counter,1} = s_next;
                            counter = counter+1;
                            %fprintf(s_next);
                            %fprintf('\n');
                          case 1
                            class = 'Yes';
                            %s_next = [char(Mols.names{m}), '                 ', class];
                            s_next = [char(names(m)), '     ', class];
                            Pred{counter,1} = s_next;
                            counter = counter+1;
                            %fprintf(s_next);
                            %fprintf('\n');
                        end
                end
            end
    end

% ---- AUTO-SAVE PREDICTION TO CSV ----
genelist = Pred;

% Choose output filename (safe default)
outname = outFile;

% Ensure cell array of strings
if ischar(genelist)
    genelist = cellstr(genelist);
end

% Write CSV
% Extract only the first column
pred_col = Pred(:,1);

% Filter out empty cells and non-char cells
pred_col_str = pred_col(~cellfun(@(x) isempty(x) || ~ischar(x), pred_col));

% Write to CSV
fid = fopen(outFile, 'w');
for i = 1:length(pred_col_str)
    fprintf(fid, '%s\n', pred_col_str{i});
end
fclose(fid);
% ------------------------------------

else
 disp(['No interaction data was found for the closest protein ', PS(inds(1)).Header])
end