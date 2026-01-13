function PredictEnzymeInteraction_NN(interactionFile, seqsFile, newSeqFile, outFile)

Mols = read_enzyme_interaction_file(interactionFile);
[numM, numE] = size(Mols.X);

PS = fastaread(seqsFile);
numPS = length(PS);

newPS = fastaread(newSeqFile);
numNewPS = length(newPS);

%% check that have data for all the enzymes that have fasta sequences for -
%% or just go down list and find nearest neighbour that do have sequences
%% for??

Map = get_Fasta_Data_Map(PS, Mols.Enames);

% find the nearest neighbour using swalign
 for i =1:numPS,
          sw = swalign(PS(i).Sequence, newPS(1).Sequence);
          S(i) = sw;
 end

 [sorted, inds] = sort(S, 'descend');
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
fid = fopen(outname, 'w');
for i = 1:size(genelist,1)
    row = genelist(i,:);
    row = row(~cellfun('isempty',row));
    fprintf(fid, '%s\n', strjoin(row, ','));
end
fclose(fid);
% ------------------------------------

 else
     msgbox(['No interaction data was found for the closest protein ', char( PS(inds(1)).Header ) ],'Unable to make prediction','modal')
 end