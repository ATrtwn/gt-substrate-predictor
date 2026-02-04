function Data = read_acceptor_interaction_file(filename)

numX = 20;
%numE = 53;


k=0;
X = dlmread(filename, '\t', 2,2);
[numM, numXandE]= size(X);
catidx = [1];
numE = numXandE - numX;

  % first, we try to open the file for reading
  fid = fopen(filename, 'r');
  if (fid == -1) 
    error('read_file: cannot open file for reading');
  end

 str = '%*s%*s'; 
 for i=1:(numX+numE),
    str  = [str, '%s'];
 end
 H = textscan(fid, str,1, 'delimiter', '\t');
 
 fseek(fid, 0, -1);
 IDs = textscan(fid,'%f%*[^\n]', 'headerLines', 1, 'delimiter', '\t');
 
 fseek(fid, 0, -1);
 N = textscan(fid,'%*d%s%*[^\n]', 'headerLines', 1, 'delimiter', '\t');
 
 fclose(fid);
  
X = dlmread(filename, '\t', 1,2);
       
Data.X = X(:, 1:numX);
Data.E = X(:, (numX+1):(numX+numE));
Data.ids= IDs{1};
Data.names = N{1} ;
%Data.molGroup = G;
Data.Xnames = H(1:numX) ;
Data.Enames = H((numX+1):end);
Data.catidx = catidx;