function Data = read_enzyme_interaction_file(filename)
%
% Read the contents of the given file.
%   Print out each line from the file onto the screen.
%   
% Arguments:
%      
%         filename          (input)  name of disk file to read
%                   

k=0;

X = dlmread(filename, '\t', 2,2);
[NUmM, NumE]= size(X);

  % first, we try to open the file for reading
  fid = fopen(filename, 'r');
  if (fid == -1) 
    error('read_file: cannot open file for reading');
  end

 str = '%*s%*s'; 
 for i=1:(NumE),
    str  = [str, '%s'];
 end
 H = textscan(fid, str,1, 'delimiter', '\t');
 
 %fseek(fid, 0, -1);
 H2 = textscan(fid, str, 1, 'delimiter', '\t');
 
 fseek(fid, 0, -1);
 IDs = textscan(fid,'%s%*[^\n]', 'headerLines', 2, 'delimiter', '\t');
 
 fseek(fid, 0, -1);
 N = textscan(fid,'%*s%s%*[^\n]', 'headerLines', 2, 'delimiter', '\t');

 fclose(fid);
  
       
Data.X = X ; 
Data.ids= IDs{1};
Data.names = N ;
Data.Enames = H;
Data.Groups = H2;
Data.catidx = [];