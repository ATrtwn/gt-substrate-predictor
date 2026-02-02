function M = get_Fasta_Data_Map(Fasta, Dnames)

M = [];
for i=1:length(Fasta)
  str1 = Fasta(i).Header;
  
  str2_ind = 0;
  
  for j=1:length(Dnames)
      same = strcmp(str1, Dnames{j});

      if (same)
          str2_ind = j;
          break;
      end
  end  
  M(end+1, :) = [i str2_ind];
end