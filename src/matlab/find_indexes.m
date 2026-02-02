function [Ind, found ]= find_indexes(Map, inds)

next = 1;
for i=1:length(inds)
  dum =   find(Map(:, 1)==inds(i));
  val = Map(dum(1), 2);
  
  %if (val>0)
    Ind(next) = val;
    found(next) = inds(i);
    next = next +1;
  %end
end