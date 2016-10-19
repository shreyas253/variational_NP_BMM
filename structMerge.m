function structMerged = structMerge(struct1,struct2,struct3)
% STRUCTMERGE   % function A = catstruct(varargin)
%
% merges 2 or 3 structures such that the new one contains all fields from both
%
% INPUTS:
%        struct1,struct1 = structres
% OUTPUTS:
%        structMerged = merged struct
%
% By: Shreyas Seshadri (shreyas.seshadri@aalto.fi), last update: 11.10.2016

if isempty(struct3)
    names{1} = fieldnames(struct1);
    names{2} = fieldnames(struct2) ;
    values{1} = struct2cell(struct1);
    values{2} = struct2cell(struct2);

    names = cat(1,names{:}) ;    
    values = cat(1,values{:}) ;    

    structMerged = cell2struct(values, names);
else
    names{1} = fieldnames(struct1);
    names{2} = fieldnames(struct2);
    names{3} = fieldnames(struct3);
    values{1} = struct2cell(struct1);
    values{2} = struct2cell(struct2);
    values{3} = struct2cell(struct3);

    names = cat(1,names{:}) ;    
    values = cat(1,values{:}) ;    

    structMerged = cell2struct(values, names);
    
end